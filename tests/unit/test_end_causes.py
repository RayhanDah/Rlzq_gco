"""ADR-006 I3/I4 : cause de fin d'épisode et origine (kickoff / générique) à l'entraînement. Diagnostic seulement :
rien n'entre dans la récompense ni la perte (GCO strict). Simulateur factice de `test_env`."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from test_env import FakeSim, fake_obs, make_env, zero_actions

from env.batch_env import BatchEnv
from env.initial_states import InitialStateSampler
from models import PolicyNet
from rewards import RewardConfig, RewardFunction
from training import Collector
from training.logging import end_scalars

KEYS = ("ended", "end_goal", "end_no_touch", "end_max_time", "end_generic", "end_touched")


def test_causes_but_no_touch_duree_max() -> None:
    # Arrange : arène 0 but au step 2 ; arène 1 jamais touchée ; arène 2 touchée à chaque step
    env, sim, _ = make_env(n_envs=3, no_touch_s=1.0, max_s=3.0)
    sim.goals[(2, 0)] = ("blue", 1)
    for t in range(60):
        sim.touches[(t, 2)] = 0
    env.reset(0)
    seen: dict[str, list[tuple[int, int]]] = {"end_goal": [], "end_no_touch": [], "end_max_time": []}
    # Act
    for t in range(50):
        _, _, _, _, info = env.step(zero_actions(env))
        causes = info["end_goal"].astype(int) + info["end_no_touch"] + info["end_max_time"]
        assert info["ended"].dtype == np.bool_
        assert np.array_equal(causes, info["ended"].astype(int))  # exactement une cause par fin
        for k in seen:
            seen[k] += [(t, int(a)) for a in np.flatnonzero(info[k])]
    # Assert
    assert seen["end_goal"] == [(2, 0)]
    assert seen["end_no_touch"][0] == (14, 1)
    assert seen["end_max_time"][0] == (44, 2)


def test_priorite_but_puis_no_touch_puis_duree_max() -> None:
    """Seuils de `RewardConfig` via `RewardFunction`. No-touch et durée max vrais au même step : no_touch ;
    but et no_touch au même step : but (`gco.episode_end`, A6 : but => non tronqué)."""
    env, sim, _ = make_env(n_envs=2, no_touch_s=1.0, max_s=1.0)
    sim.goals[(14, 1)] = ("blue", 1)
    env.reset(0)
    for _ in range(15):
        _, _, _, _, info = env.step(zero_actions(env))
    assert info["end_no_touch"].tolist() == [True, False]
    assert info["end_max_time"].tolist() == [False, False]
    assert info["end_goal"].tolist() == [False, True]


@pytest.mark.parametrize(("p_kick", "generic"), [(1.0, False), (0.0, True)])
def test_origine_kickoff_ou_generique(p_kick: float, generic: bool) -> None:
    env, sim, _ = make_env(n_envs=2, p_kick=p_kick, no_touch_s=1.0, max_s=3.0)
    sim.touches[(0, 0)] = 0
    env.reset(0)
    for _ in range(15):
        _, _, _, _, info = env.step(zero_actions(env))
    assert info["ended"].tolist() == [False, True]  # arène 0 touchée au step 0 : pas encore finie
    assert info["end_generic"].tolist() == [False, generic]
    assert info["end_touched"].tolist() == [False, False]


def test_origine_est_celle_du_tirage_qui_a_demarre_l_episode() -> None:
    env, sim, _ = make_env(n_envs=1, p_kick=0.0, no_touch_s=1.0, max_s=3.0)
    sim.touches[(3, 0)] = 0
    env.reset(0)
    flags = []
    for t in range(60):
        if t == 14:
            env.set_p_kick(1.0)  # le tirage a lieu au reset, à la fin de l'épisode courant
        _, _, _, _, info = env.step(zero_actions(env))
        if info["ended"][0]:
            flags.append((t, bool(info["end_generic"][0]), bool(info["end_touched"][0])))
    assert flags[0] == (3 + 15, True, True)  # touché au step 3 puis 15 steps sans touche
    assert len(flags) >= 2 and all(not g for _, g, _ in flags[1:])


class _ResumableSim(FakeSim):
    def state_dict(self) -> dict[str, Any]:
        keep = ("_state", "_blue", "_orange", "_tick", "_touch", "n_steps")
        return copy.deepcopy({k: getattr(self, k) for k in keep})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for k, v in copy.deepcopy(dict(state)).items():
            setattr(self, k, v)


def _mk() -> BatchEnv:
    sim = _ResumableSim(4)
    for t in range(200):
        for a in range(4):
            if t % 11 == a:
                sim.touches[(t, a)] = 0
        if t % 17 == 3:
            sim.goals[(t, t % 4)] = ("blue", 1)
    reward = RewardFunction(RewardConfig(no_touch_timeout_s=1.0, max_episode_s=3.0))
    return BatchEnv(sim, reward, InitialStateSampler(2, 0.5, 0), build_obs=fake_obs, obs_dim=7)


def test_reprise_exacte_conserve_l_origine_en_cours() -> None:
    """Sans l'origine sauvegardée, les épisodes en cours à la reprise seraient comptés kickoff à tort."""
    ref = _mk()
    ref.reset(5)
    act = zero_actions(ref)
    for _ in range(20):
        ref.step(act)
    snap = ref.state_dict()
    assert snap["episode_generic"].any() and not snap["episode_generic"].all()  # le test discrimine
    res = _mk()
    res.reset(99)
    assert res.load_state_dict(snap) is None
    seen_end = False
    for _ in range(60):
        a, b = ref.step(act)[4], res.step(act)[4]
        for k in KEYS:
            assert np.array_equal(a[k], b[k]), k
        seen_end |= bool(a["ended"].any())
    assert seen_end


def test_ancien_checkpoint_sans_champs_origine_reste_chargeable() -> None:
    ref = _mk()
    ref.reset(5)
    for _ in range(5):
        ref.step(zero_actions(ref))
    snap = ref.state_dict()
    del snap["episode_generic"], snap["episode_touched"]
    res = _mk()
    res.reset(1)
    res.load_state_dict(snap)
    res.step(zero_actions(res))  # valeurs par défaut (kickoff, sans touche) : pas d'erreur


class _Env:
    """`BatchEnv` ; `strip=True` retire les clés I3/I4 de `info` (équivalent de l'env d'avant l'instrumentation)."""

    def __init__(self, strip: bool) -> None:
        self.env = _mk()
        self.strip = strip
        self.n_agents, self.obs_dim = self.env.n_agents, self.env.obs_dim

    def reset(self, seed: int) -> Any:
        return self.env.reset(seed)

    def step(self, actions: Any) -> Any:
        obs, rew, term, trunc, info = self.env.step(actions)
        if self.strip:
            info = {k: v for k, v in info.items() if k not in KEYS}
        return obs, rew, term, trunc, info


def test_collecteur_trajectoire_identique_avec_et_sans_compteurs() -> None:
    batches, counts = [], []
    for strip in (False, True):
        torch.manual_seed(0)
        env = _Env(strip)
        col = Collector(env, PolicyNet(env.obs_dim, 4, (8,)), 40, torch.device("cpu"), seed=3)
        batches.append([col.collect(0) for _ in range(3)])
        counts.append(dict(col.end_counts))
    for x, y in zip(batches[0], batches[1], strict=True):
        for f in ("obs", "act", "logp", "rew", "term", "trunc", "final_obs", "final_t", "final_agent", "last_obs"):
            assert np.array_equal(getattr(x, f), getattr(y, f)), f
    assert counts[1]["ended"] == 0 and counts[0]["ended"] > 0  # clés absentes : zéro, sans erreur
    c = counts[0]
    assert c["ended"] == c["goal"] + c["no_touch"] + c["max_time"] == c["kickoff"] + c["generic"]
    assert c["goals_kickoff"] + c["goals_generic"] == c["goal"] and c["touched"] <= c["ended"]


def test_end_scalars_noms() -> None:
    counts = {
        "ended": 5,
        "goal": 2,
        "no_touch": 2,
        "max_time": 1,
        "kickoff": 3,
        "generic": 2,
        "touched": 4,
        "goals_kickoff": 1,
        "goals_generic": 1,
    }
    assert end_scalars(counts) == {
        "env/ends_goal": 2.0,
        "env/ends_no_touch": 2.0,
        "env/ends_max_time": 1.0,
        "env/ends_kickoff": 3.0,
        "env/ends_generic": 2.0,
        "env/ends_touched": 4.0,
        "env/goals_kickoff": 1.0,
        "env/goals_generic": 1.0,
    }
