"""`BatchEnv` + RocketGPU CPU (build local, non versionné). Sauté si RocketGPU ou les meshes sont absents.

Environnement : `ROCKETGPU_PY_DIR` (dossier du module), `ROCKETGPU_COLLISION_MESH_DIR` ; optionnel `ROCKETGPU_DLL_DIR`.
Les hypothèses d'ADR-003 vérifiées ici sont consignées dans le rapport T2 (commit de ce fichier).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from env import BatchEnv, InitialStates, InitialStateSampler, RocketGPUSimulator, SimulatorUnavailableError
from obs.state import N_PADS, BatchState
from rewards import RewardConfig, RewardFunction

F32 = npt.NDArray[np.float32]
N_ENVS = 64
KICKOFFS = {(-2048, -2560), (-256, -3840), (0, -4608), (256, -3840), (2048, -2560)}
"""Positions du bleu (x, y) des 5 kickoffs officiels, relevées sur RocketGPU @f0508a1 (rapport T2)."""

try:
    _PROBE = RocketGPUSimulator(2)
    _SKIP: str | None = None
except (SimulatorUnavailableError, OSError) as exc:
    _SKIP = f"RocketGPU indisponible : {exc}"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


def trivial_obs(state: BatchState, prev: F32) -> F32:
    return np.concatenate([state.car_pos.reshape(-1, 3), prev], axis=1).astype(np.float32)


def random_table(seed: int = 0, n: int = 90) -> F32:
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1.0, 1.0, (n, 8)).astype(np.float32)
    t[:, 5:] = t[:, 5:] > 0
    t[0] = 0.0  # action 0 = inerte
    return t


def make_env(
    n_envs: int = N_ENVS, p_kick: float = 1.0, **reward_kw: Any
) -> tuple[BatchEnv, RocketGPUSimulator, InitialStateSampler]:
    sim = RocketGPUSimulator(n_envs, 2, 8)
    sampler = InitialStateSampler(2, p_kick, 0)
    env = BatchEnv(sim, trivial_obs, random_table(), RewardFunction(RewardConfig(**reward_kw)), sampler)
    return env, sim, sampler


def still_state(ball_pos: tuple[float, float, float], ball_vel: tuple[float, float, float]) -> InitialStates:
    """Une arène : voitures immobiles aux coins opposés, balle donnée (téléportation générique, rien de ciblé)."""
    one = np.ones((1, 2), dtype=np.uint8)
    zero = np.zeros((1, 2), dtype=np.uint8)
    car_pos = np.array([[[3000.0, -3000.0, 17.0], [-3000.0, 3000.0, 17.0]]], dtype=np.float32)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 2, 1))
    return InitialStates(
        ball_pos=np.array([ball_pos], dtype=np.float32),
        ball_vel=np.array([ball_vel], dtype=np.float32),
        ball_ang_vel=np.zeros((1, 3), dtype=np.float32),
        car_pos=car_pos,
        car_vel=np.zeros((1, 2, 3), dtype=np.float32),
        car_ang_vel=np.zeros((1, 2, 3), dtype=np.float32),
        car_quat=quat,
        car_boost=np.full((1, 2), 33.0, dtype=np.float32),
        car_on_ground=one,
        car_has_jumped=zero,
        car_has_double_jumped=zero,
        car_has_flipped=zero,
        car_air_time_since_jump=np.zeros((1, 2), dtype=np.float32),
    )


def test_etat_est_une_vue_sans_copie() -> None:
    sim = RocketGPUSimulator(4)
    st = sim.state()
    st.validate()
    assert all(not a.flags.owndata for a in (st.car_pos, st.ball_pos, st.pad_cooldown, st.car_quat))
    before = st.car_pos.copy()
    sim.step(np.tile(np.array([1, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32), (8, 1)))
    assert sim.state() is st and not np.array_equal(before, st.car_pos)  # la même vue reflète la mémoire du sim


def test_marche_aleatoire_valide_resets_et_5_kickoffs() -> None:
    env, sim, _ = make_env(max_episode_s=3.0)  # 3 s : plusieurs resets dans la fenêtre
    rng = np.random.default_rng(0)
    obs = env.reset(0)
    st = sim.state()
    st.validate()
    spots = {tuple(np.round(p).astype(int)) for p in st.car_pos[:, 0, :2]}
    n_done = 0
    for _ in range(300):
        obs, rew, term, trunc, info = env.step(rng.integers(0, 90, env.n_agents))
        st.validate()  # à chaque step
        assert obs.shape == (env.n_agents, 11)
        assert not (term & trunc).any() and (rew[~term] == 0).all()
        done = info["final_agent"][::2] // 2
        n_done += done.size
        if done.size:
            spots |= {tuple(np.round(p).astype(int)) for p in st.car_pos[done, 0, :2]}  # états après reset (kickoffs)
    assert n_done >= N_ENVS  # chaque arène a fini au moins une fois en moyenne (3 s = 45 steps sur 300)
    assert env.n_ambiguous_goals == 0
    assert spots == KICKOFFS  # exactement les 5 kickoffs officiels, tous produits
    assert np.isfinite(obs).all()


def test_etats_generiques_survivent_au_sim() -> None:
    env, sim, _ = make_env(p_kick=0.0, max_episode_s=3.0)  # 100 % états génériques
    rng = np.random.default_rng(1)
    env.reset(3)
    st = sim.state()
    st.validate()
    for _ in range(200):
        env.step(rng.integers(0, 90, env.n_agents))
        st.validate()
        assert np.abs(st.car_pos).max() < 6000.0 and np.abs(st.ball_pos[:, 2]).max() < 3000.0


def test_but_donne_plus_ou_moins_un_et_terminated() -> None:
    env, sim, _ = make_env()
    env.reset(0)
    # arène 0 : balle dans le filet bleu (y < 0) => orange marque ; arène 1 : filet orange => bleu marque
    sim.set_states(np.array([0]), still_state((0.0, -4700.0, 300.0), (0.0, -2500.0, 0.0)))
    sim.set_states(np.array([1]), still_state((0.0, 4700.0, 300.0), (0.0, 2500.0, 0.0)))
    idle = np.zeros(env.n_agents, dtype=np.int64)
    seen: dict[int, tuple[list[float], bool, bool]] = {}
    for _ in range(10):
        _, rew, term, trunc, info = env.step(idle)
        for a in (0, 1):
            if term[2 * a] and a not in seen:
                seen[a] = (rew[2 * a : 2 * a + 2].tolist(), bool(trunc[2 * a]), bool(info["goal_net"][a] >= 0))
    assert seen[0] == ([-1.0, 1.0], False, True)  # slot 0 = bleu encaisse, slot 1 = orange marque
    assert seen[1] == ([1.0, -1.0], False, True)
    assert sim.state().ball_pos[0, 2] > 0  # arène réinitialisée, balle revenue dans le terrain
    assert abs(float(sim.state().ball_pos[0, 1])) < 5000.0


def test_sans_touche_20s_tronque_recompense_nulle() -> None:
    env, _, _ = make_env()  # défauts de spec : 20 s sans touche, 120 s de durée
    env.reset(0)
    idle = np.zeros(env.n_agents, dtype=np.int64)  # voitures inertes : aucune touche possible
    for t in range(300):  # 20 s = 2400 ticks = 300 steps
        _, rew, term, trunc, info = env.step(idle)
        assert (rew == 0).all() and not term.any()
        if t < 299:
            assert not trunc.any()
    assert trunc.all()  # au 300e step : tronqué, jamais terminé
    assert info["final_obs"].shape[0] == env.n_agents
    assert (info["episode_ticks"] == 2400).all()


def test_pads_contigus_et_cooldown_en_secondes() -> None:
    """ADR-003 §2 : pads de l'arène e à l'indice e·34 + i, cooldown en secondes (4 s petit pad, 10 s grand)."""
    env, sim, _ = make_env(n_envs=8)
    env.reset(0)
    import rocketgpu_py as module  # déjà chargé par l'adaptateur

    locs = module.boost_pad_locations()
    for pad_i, expected in ((5, 4.0), (30, 10.0)):
        env.reset(0)
        p = locs[pad_i][0]
        s = still_state((3000.0, 0.0, 500.0), (0.0, 0.0, 0.0))
        s.car_pos[0, 0] = (p.x, p.y, 17.0)
        s.car_boost[0, 0] = 0.0
        sim.set_states(np.array([3]), s)
        sim.step(np.zeros((16, 8), dtype=np.float32))
        st = sim.state()
        assert np.argwhere(st.pad_active == 0).tolist() == [[3, pad_i]]
        cd = float(st.pad_cooldown[3, pad_i])
        assert expected - 0.2 < cd <= expected  # secondes : 4 s -> 3,94 après 8 ticks (1/15 s)
        assert float(st.car_boost[3, 0]) == (12.0 if expected == 4.0 else 100.0)
        assert st.pad_cooldown.shape == (8, N_PADS)
        assert float(np.delete(st.pad_cooldown, 3, axis=0).max()) == 0.0  # aucune autre arène touchée


def test_touche_detectee_au_kickoff_et_horodatage() -> None:
    """Après un kickoff `ball_hit_info` est invalide ; une touche porte un horodatage dans ]tick_avant, tick_après]."""
    sim = RocketGPUSimulator(16)
    ids = np.arange(16, dtype=np.int64)
    sim.reset_to_kickoff(ids, np.arange(1, 17, dtype=np.uint64))
    assert (sim.touch_ticks() == -1).all()
    ctrl = np.tile(np.array([1, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32), (32, 1))  # plein gaz
    n_touch = 0
    for _ in range(80):
        before = sim.env_ticks()
        sim.step(ctrl)
        stamp = sim.touch_ticks()
        new = stamp >= before[:, None]
        n_touch += int(new.sum())
        assert (stamp[new] < (before + 8)[:, None].repeat(2, axis=1)[new]).all()
    assert n_touch > 0


def test_determinisme_a_seed_fixe_avec_rocketgpu() -> None:
    def run() -> tuple[F32, F32]:
        env, _, _ = make_env(n_envs=16, p_kick=0.5, max_episode_s=2.0)
        rng = np.random.default_rng(4)
        obs = env.reset(11)
        for _ in range(60):
            obs, rew, _, _, _ = env.step(rng.integers(0, 90, env.n_agents))
        return obs.copy(), rew.copy()

    a, b = run(), run()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
