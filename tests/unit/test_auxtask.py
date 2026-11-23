"""Tâche auxiliaire « prochain but » (`training.auxtask`, ADR-010 D8) : étiquettes, rejeu, gradient (C2), régression bit
à bit « désactivée = PPO d'avant », hashs, reprise exacte, θ₀, parts de gradient et de déplacement, gardes, mémoire,
parité float16, configs, lanceur et outils de banc. CPU, Adam manuel."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import io
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from _ppo_reference import ReferenceLearner  # noqa: E402
from test_checkpoint import _Adam  # noqa: E402
from test_obs import prev_of, random_state  # noqa: E402
from test_ppo import Corridor, _small_cfg  # noqa: E402
from test_run import BASE, CheckpointManager, cfg_of, events, go, make_factory, rows  # noqa: E402
from test_run_4a import r4  # noqa: E402
from test_run_4a_interventions import _diff  # noqa: E402

from models import PolicyNet, ValueNet  # noqa: E402
from obs import build_obs  # noqa: E402
from training import Collector, ConfigError, Learner, OverlapTrainer, RunConfig  # noqa: E402
from training.auxtask import (  # noqa: E402
    AuxConfig,
    AuxGuardState,
    AuxHead,
    AuxTask,
    ClassReplay,
    GoalWindow,
    aux_guard,
    aux_logits,
    expected_nbytes,
    make_aux_head,
    policy_trunk,
)
from training.checkpoint import decode, encode  # noqa: E402
from training.sil import SILConfig  # noqa: E402

CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
THETA0_STD_S0 = "7fab4c747a778f6a2e99fdd12039cd3a496bc2974672d73f67e24552adcdcb8a"
"""SHA-256 des poids de π de std s0 à l'itération 0 (`production/state.md`, `docs/reports/int_sil_s0.md` §4)."""


def _load(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


abs_ = _load("aux_bench_summary")
afp = _load("aux_f16_parity")


def acfg(**kw: Any) -> AuxConfig:
    """Petite config activée : H = 2 (épisodes du Corridor ≤ 3 pas : buts, troncatures et sorties à 0), lot 8."""
    base: dict[str, Any] = {
        "enabled": True,
        "horizon_steps": 2,
        "batch_size": 8,
        "capacity_per_class": 64,
        "store_float16": False,
    }
    return AuxConfig(**(base | kw))


def steps_arrays(t_len: int, n: int, d: int = 3) -> dict[str, np.ndarray]:
    """Lot synthétique (T, N, …) : obs[t, a] = (t, a, 0), aucune fin."""
    obs = np.zeros((t_len, n, d), np.float32)
    obs[..., 0] = np.arange(t_len)[:, None]
    obs[..., 1] = np.arange(n)[None, :]
    return {
        "obs": obs,
        "rew": np.zeros((t_len, n), np.float32),
        "term": np.zeros((t_len, n), np.bool_),
        "trunc": np.zeros((t_len, n), np.bool_),
    }


def _labels_by_step(window: GoalWindow, b: dict[str, np.ndarray], agent: int = 0) -> dict[int, Any]:
    """Étiquette finale de chaque pas (lu par obs[..., 0] = t) de `agent` : -1/0/+1, « M » si masqué ; absent si le
    pas est encore dans la fenêtre."""
    out: dict[int, Any] = {}
    for t in range(b["obs"].shape[0]):
        n_before = window.length[agent]
        ex = window.step(b["obs"][t], b["rew"][t], b["term"][t], b["trunc"][t])
        for o in ex.zero_obs:
            if int(o[1]) == agent:
                out[int(o[0])] = 0
        for o, y in zip(ex.pos_obs, ex.pos_label, strict=True):
            if int(o[1]) == agent:
                out[int(o[0])] = int(y)
        if ex.n_masked and b["trunc"][t, agent] and not b["term"][t, agent]:
            for k in range(min(n_before + 1, window.horizon)):
                out[t - k] = "M"
    return out


# ---------------------------------------------------------------------------------------------------- config


def test_defauts_de_l_adr_et_validation() -> None:
    c = AuxConfig()
    assert (c.enabled, c.coef, c.horizon_steps, c.batch_size, c.capacity_per_class) == (False, 0.1, 150, 8192, 125_000)
    assert (c.store_float16, c.share_alert, c.share_alert_patience) == (True, 0.5, 50)
    assert (c.ram_min_available_mb, c.ram_low_max_s) == (1000.0, 600.0)
    assert c.anneal_start_timesteps is None and c.anneal_end_timesteps is None and c.half == 4096
    for bad in (
        {"batch_size": 7},
        {"capacity_per_class": 3, "batch_size": 8},
        {"coef": -0.1},
        {"anneal_start_timesteps": 10},
        {"anneal_start_timesteps": 10, "anneal_end_timesteps": 5},
        {"share_alert": 0.0},
    ):
        with pytest.raises(ValueError, match="AuxConfig invalide"):
            AuxConfig(**bad)


def test_lambda_constant_par_defaut_et_calendrier_optionnel() -> None:
    assert [AuxConfig(coef=0.3).coef_at(t) for t in (0, 10**9, 10**12)] == [0.3, 0.3, 0.3]
    c = AuxConfig(coef=0.1, anneal_start_timesteps=100, anneal_end_timesteps=200)
    assert c.coef_at(0) == c.coef_at(100) == 0.1
    assert c.coef_at(150) == pytest.approx(0.05) and c.coef_at(200) == c.coef_at(10**9) == 0.0


def test_aux_refusee_en_gco_s_avec_la_sil_et_cles_inconnues() -> None:
    aux = {"enabled": True}
    gco_s = {"variant": "GCO-S", "no_touch_timeout_s": 0.5, "max_episode_s": 0.5}
    kick = {"p_kick_start": 1.0, "p_kick_end": 1.0, "anneal_timesteps": 0}
    with pytest.raises(ConfigError, match="interdite"):
        cfg_of(reward=gco_s, initial_states=kick, aux=aux)
    sil = {"enabled": True, "anneal_start_timesteps": 0, "anneal_end_timesteps": 100}
    with pytest.raises(ConfigError, match="D11"):
        cfg_of(aux=aux, sil=sil)
    with pytest.raises(ConfigError, match="inconnues"):
        cfg_of(aux={**aux, "lambda": 0.1})
    budget = BASE["budget"]["total_timesteps"]
    with pytest.raises(ConfigError, match="avant la fin"):
        cfg_of(aux={**aux, "anneal_start_timesteps": 0, "anneal_end_timesteps": budget})
    cfg_of(aux={**aux, "anneal_start_timesteps": 0, "anneal_end_timesteps": budget - 1})
    cfg_of(reward=gco_s, initial_states=kick, aux={"enabled": False})  # désactivée : acceptée partout
    cfg_of(aux={"enabled": False}, sil=sil)


def test_section_aux_absente_ne_change_aucun_hash() -> None:
    # D8-2 : mêmes valeurs que tests/unit/test_sil.py (code de main f2218a6, avant la SIL)
    expected = {
        "run_4a_std.yaml": "9791371a6e47f14b1fa38b592700d8d3a9c3670a5d1ab3ad04b383a2e28ca479",
        "run_4a_s.yaml": "e6cdbb70e38450eff11d7a751ee4ec4c005b8d88e0866625310d0bf86cecdf2d",
    }
    for name, h in expected.items():
        cfg = RunConfig.from_yaml(CONFIGS / name)
        assert cfg.aux is None and "aux" not in cfg.to_dict() and cfg.trajectory_hash() == h
    cfg = RunConfig.from_yaml(CONFIGS / "int_aux.yaml")
    assert cfg.aux is not None and cfg.aux.enabled and cfg.trajectory_hash() != expected["run_4a_std.yaml"]


def _raw(name: str) -> dict[str, Any]:
    out: dict[str, Any] = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    return out


def test_bench_et_smoke_aux_ne_different_que_des_champs_annonces() -> None:
    # bench_int_aux = int_aux hors `name` et `checkpoint.keep` (opérationnel : hash de trajectoire égal hors nom)
    assert _diff(_raw("int_aux.yaml"), _raw("bench_int_aux.yaml")) == {
        ("name",): ("int_aux", "bench_int_aux"),
        ("checkpoint", "keep"): (2, 1000),
    }
    a, b = RunConfig.from_yaml(CONFIGS / "int_aux.yaml"), RunConfig.from_yaml(CONFIGS / "bench_int_aux.yaml")
    assert dataclasses.replace(b, name=a.name).trajectory_hash() == a.trajectory_hash()
    # smoke_aux = smoke + bloc aux d'int_aux à l'identique ; seul le budget change (H = 100 pas à dépasser avant la
    # reprise, voir l'en-tête et le test suivant)
    smoke, smoke_aux = _raw("smoke.yaml"), _raw("smoke_aux.yaml")
    d = _diff(smoke, smoke_aux)
    assert d.pop(("name",)) == ("smoke", "smoke_aux")
    assert d.pop(("budget", "total_timesteps")) == (1_000_000, 2_752_512)
    assert d and all(k[0] == "aux" for k in d)
    assert smoke_aux["aux"] == _raw("int_aux.yaml")["aux"]
    cfg = RunConfig.from_yaml(CONFIGS / "smoke_aux.yaml")
    assert cfg.aux is not None and cfg.aux.enabled and cfg.smoke is not None


def test_smoke_aux_remplit_le_compartiment_zero_avant_et_apres_la_reprise() -> None:
    # Régression (smoke 690a828 : `aux/replay_zero_size` = 0 sur 18 itérations, aucun lot) : un pas ne sort avec
    # y = 0 qu'après H pas sans fin d'épisode ; la 1re partie du smoke doit donc ingérer nettement plus de H pas par
    # agent. Arrange : calendrier réel de `scripts/train.py --smoke` (mise à jour s = s lots ingérés, `Learner.update`)
    import train as train_script

    cfg = RunConfig.from_yaml(CONFIGS / "smoke_aux.yaml")
    assert cfg.aux is not None and cfg.smoke is not None
    resume, total = train_script.smoke_resume_iteration(cfg), cfg.total_iterations
    t_len = cfg.collection.steps_per_batch
    n_learn = cfg.batch_learned_steps // t_len  # agents appris par pas de collecte (sans pool : tous)
    # Act : vraie fenêtre, un agent sans aucune fin d'épisode (borne haute des sorties y = 0 par agent)
    w, zero_after = GoalWindow(1, 3, cfg.aux.horizon_steps, np.float32), []
    b, n_zero = steps_arrays(t_len, 1), 0
    for _ in range(total):
        for t in range(t_len):
            n_zero += w.step(b["obs"][t], b["rew"][t], b["term"][t], b["trunc"][t]).n_zero
        zero_after.append(n_zero * n_learn)  # compartiment y = 0 après l'ingestion de la mise à jour s (index s − 1)
    first = next((s + 1 for s, z in enumerate(zero_after) if z >= cfg.aux.half), None)
    # Assert : premier lot possible au moins 2 itérations avant l'arrêt (marge pour les fenêtres remises à zéro par un
    # but) et des mises à jour après la reprise ; seul le compartiment y = 0 est borné ici (y ≠ 0 : mesuré, smoke)
    assert first is not None and first <= resume - 1, (first, resume, resume * t_len, cfg.aux.horizon_steps)
    assert zero_after[resume - 1] > 0 and total > resume
    # H = 100 (repli D2, banc du 2026-11-22) : 1re sortie y = 0 au pas 101, dans l'itération 15 (pas 99 à 105) ;
    # budget de 48 conservé (reprise à 24 : 168 pas par agent, 1er lot possible 9 itérations avant la reprise)
    assert (first, resume, total) == (15, 24, 48)  # valeurs citées dans l'en-tête de configs/smoke_aux.yaml


def test_lanceur_variante_int_aux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : le hook d'évaluation lève dès sa construction (aucun run)
    import train as train_script

    seen: list[Any] = []

    class _Stop(Exception):
        pass

    def fake_build(cfg: Any, hook_cfg: Any, *a: Any, **k: Any) -> Any:
        seen.append((cfg, hook_cfg))
        raise _Stop

    monkeypatch.setattr(train_script, "build_eval_hook", fake_build)
    # Act
    with pytest.raises(_Stop):
        r4.launch("int_aux", 0, config_dir=CONFIGS, max_hours=None, runs_root=tmp_path / "r", checkpoints_root=tmp_path)
    # Assert
    assert r4.config_file("int_aux") == "int_aux.yaml" and r4.run_name("int_aux", 0) == "int_aux_s0"
    assert r4.default_eval_config("int_aux") == "run_s1v2_eval.yaml"
    ((cfg, hook_cfg),) = seen
    assert cfg.name == "int_aux_s0" and cfg.aux is not None and cfg.aux.enabled and cfg.sil is None
    assert hook_cfg.protocol == "s1_v2" and cfg.reward.variant == "GCO-std"


# ---------------------------------------------------------------------------------------------------- étiquettes


def test_but_marque_etiquete_les_h_derniers_pas_et_zero_avant() -> None:
    # Arrange : H = 3, agent 0 marque au pas 5 (épisode de 6 pas), agent 1 encaisse (csc : r = −1) au pas 1
    w, b = GoalWindow(2, 3, 3, np.float32), steps_arrays(8, 2)
    b["rew"][5, 0], b["term"][5, 0] = 1.0, True
    b["rew"][1, 1], b["term"][1, 1] = -1.0, True
    # Act
    lab0 = _labels_by_step(GoalWindow(2, 3, 3, np.float32), b, agent=0)
    lab1 = _labels_by_step(w, b, agent=1)
    # Assert : +1 sur [T − H + 1, T] = [3, 5], 0 avant (sortis après H pas) ; nouvel épisode encore dans la fenêtre
    assert lab0 == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    # agent 1 : épisode plus court que H, tous les pas −1 ; l'épisode suivant (2 à 7) : 2 à 4 sortis à 0
    assert lab1 == {0: -1, 1: -1, 2: 0, 3: 0, 4: 0}


def test_troncature_masque_les_h_derniers_pas_et_zero_avant() -> None:
    w, b = GoalWindow(1, 3, 3, np.float32), steps_arrays(7, 1)
    b["trunc"][5, 0] = True  # no-touch / 120 s : aucun but, futur inconnu
    lab = _labels_by_step(w, b)
    assert lab == {0: 0, 1: 0, 2: 0, 3: "M", 4: "M", 5: "M"}


def test_fin_terminale_sans_recompense_etiquetee_zero_et_compteurs() -> None:
    w, b = GoalWindow(1, 3, 4, np.float32), steps_arrays(3, 1)
    b["term"][2, 0] = True  # r = 0 au terminal : aucun but pour cet agent
    exits = [w.step(b["obs"][t], b["rew"][t], b["term"][t], b["trunc"][t]) for t in range(3)]
    last = exits[-1]
    assert (last.n_zero, last.n_pos, last.n_neg, last.n_masked) == (3, 0, 0, 0)
    assert last.zero_obs[:, 0].tolist() == [0.0, 1.0, 2.0]  # du plus ancien au plus récent
    assert int(w.length[0]) == 0


def test_recompense_hors_terminal_refusee() -> None:
    w, b = GoalWindow(1, 3, 3, np.float32), steps_arrays(1, 1)
    b["rew"][0, 0] = 0.5
    with pytest.raises(ValueError, match="hors terminal"):
        w.step(b["obs"][0], b["rew"][0], b["term"][0], b["trunc"][0])


def _task(cfg: AuxConfig | None = None, obs_dim: int = 3) -> AuxTask:
    torch.manual_seed(0)
    policy = PolicyNet(obs_dim, 5, (8,))
    return AuxTask(cfg or acfg(horizon_steps=3), policy, make_aux_head(policy, 0), seed=0)


def test_camp_gele_exclu_et_fenetre_a_cheval_sur_deux_lots() -> None:
    # Arrange : agents 0 et 2 appris, 1 gelé ; agent 0 marque au pas 1 du 2e lot (pas global 4, épisode de 5 pas, H = 3)
    task, mask = _task(), np.array([True, False, True])
    b1, b2 = steps_arrays(3, 3), steps_arrays(3, 3)
    b2["obs"][..., 0] += 3  # numérotation globale des pas
    b2["rew"][1, 0], b2["term"][1, 0] = 1.0, True
    b1["rew"][1, 1], b1["term"][1, 1] = -1.0, True  # but de l'agent gelé : ignoré
    # Act
    c1 = task.ingest(b1["obs"], b1["rew"], b1["term"], b1["trunc"], mask, 0)
    c2 = task.ingest(b2["obs"], b2["rew"], b2["term"], b2["trunc"], mask, 1)
    # Assert : aucun pas de l'agent 1 ; agent 0 : pas 0 sorti à 0, pas 1 à 4 étiquetés +1 (dont 1 et 2 du 1er lot)
    assert task.pos is not None and task.zero is not None
    agents = {int(a) for a in task.pos.obs[: task.pos.size, 1]} | {int(a) for a in task.zero.obs[: task.zero.size, 1]}
    assert agents <= {0, 2}
    pos_steps = sorted(int(o[0]) for o in task.pos.obs[: task.pos.size] if int(o[1]) == 0)
    assert pos_steps == [2, 3, 4] and set(task.pos.label[: task.pos.size].tolist()) == {1}
    assert c1 == {"pos": 0, "neg": 0, "zero": 0, "masked": 0}
    assert c2["pos"] == 3 and c2["zero"] == 5 and c2["masked"] == 0  # pas 0, 1 (agent 0) ; 0, 1, 2 (agent 2)
    with pytest.raises(ValueError, match="learn_mask"):
        task.ingest(b1["obs"], b1["rew"], b1["term"], b1["trunc"], np.ones(3, np.bool_), 2)


def test_etiquetage_vectorise_egal_a_une_reference_naive() -> None:
    """Propriété : sur un lot aléatoire (buts, csc, troncatures, fins simultanées, H = 4), les multiensembles (pas,
    agent, étiquette) et le nombre de masqués coïncident avec une référence naïve pas à pas par agent."""
    rng = np.random.default_rng(0)
    t_len, n, h = 60, 6, 4
    b = steps_arrays(t_len, n)
    ends = rng.random((t_len, n))
    b["term"] = ends < 0.08
    b["rew"] = np.where(b["term"], rng.choice([-1.0, 0.0, 1.0], size=(t_len, n)), 0.0).astype(np.float32)
    b["trunc"] = (ends > 0.95) | (b["term"] & (rng.random((t_len, n)) < 0.3))
    got: list[tuple[int, int, int]] = []
    masked = 0
    w = GoalWindow(n, 3, h, np.float32)
    for t in range(t_len):
        ex = w.step(b["obs"][t], b["rew"][t], b["term"][t], b["trunc"][t])
        got += [(int(o[0]), int(o[1]), 0) for o in ex.zero_obs]
        got += [(int(o[0]), int(o[1]), int(y)) for o, y in zip(ex.pos_obs, ex.pos_label, strict=True)]
        masked += ex.n_masked
    ref: list[tuple[int, int, int]] = []
    ref_masked = 0
    for a in range(n):
        pending: list[int] = []
        for t in range(t_len):
            if len(pending) == h:
                ref.append((pending.pop(0), a, 0))
            pending.append(t)
            if b["term"][t, a]:
                ref += [(s, a, int(np.sign(b["rew"][t, a]))) for s in pending]
                pending = []
            elif b["trunc"][t, a]:
                ref_masked += len(pending)
                pending = []
    assert sorted(got) == sorted(ref) and masked == ref_masked and len(ref) > 50 and ref_masked > 0


# ---------------------------------------------------------------------------------------------------- rejeu


def test_rejeu_fifo_par_compartiment() -> None:
    r = ClassReplay(4, 2, np.float32)
    r.add(np.arange(6, dtype=np.float32).reshape(3, 2), np.array([1, -1, 1], np.int8), 0)
    r.add(np.arange(6, 10, dtype=np.float32).reshape(2, 2), np.array([-1, 1], np.int8), 1)
    assert r.size == 4 and r.pos == 1
    assert sorted(r.obs[:, 0].tolist()) == [2.0, 4.0, 6.0, 8.0]  # la plus ancienne (0) a été écrasée
    assert r.nbytes == 4 * (2 * 4 + 1 + 4)


def test_tirage_4096_plus_4096_reequilibre_et_pas_de_lot_sous_4096() -> None:
    """D8-6 aux tailles de l'ADR : lot de 8 192 = 4 096 du compartiment ± puis 4 096 du compartiment 0 ; aucun lot
    tant que le compartiment ± a moins de 4 096 transitions."""
    task = _task(AuxConfig(enabled=True, store_float16=True))
    assert task.pos is not None and task.zero is not None
    rng = np.random.default_rng(0)
    task.zero.add(rng.normal(size=(10_000, 3)).astype(np.float32), np.zeros(10_000, np.int8), 0)
    task.pos.add(rng.normal(size=(4_095, 3)).astype(np.float32), rng.choice([-1, 1], 4_095).astype(np.int8), 0)
    empty = steps_arrays(1, 2)
    kw: dict[str, Any] = {"iteration": 1, "learned_per_iteration": 2, "device": CPU, "n_ppo_steps": 4}
    opt = _Adam(list(task.trunk_params) + list(task.head.parameters()), 1e-3)
    it = task.begin(
        empty["obs"], empty["rew"], empty["term"], empty["trunc"], np.ones(2, np.bool_), optimizer=opt, **kw
    )
    assert not it.has_batch and it.stats().batch_present == 0.0
    task.pos.add(rng.normal(size=(1, 3)).astype(np.float32), np.array([1], np.int8), 0)
    it = task.begin(
        empty["obs"], empty["rew"], empty["term"], empty["trunc"], np.ones(2, np.bool_), optimizer=opt, **kw
    )
    assert it.has_batch and it._y is not None and it._obs is not None
    y = it._y.numpy()
    assert y.shape == (4, 8192) and it._obs.dtype == torch.float16  # copie float16, convertie sur l'appareil
    assert np.isin(y[:, :4096], [0, 2]).all() and (y[:, 4096:] == 1).all()
    assert 0.4 < (y[:, :4096] == 2).mean() < 0.6  # tirage uniforme dans le compartiment ± (étiquettes ±1 50/50)


# ---------------------------------------------------------------------------------------------------- learner


def _learner(aux: AuxConfig | None, lr: float = 1e-2, cls: type[Learner] = Learner) -> Learner:
    torch.manual_seed(0)
    cfg = _small_cfg(minibatch_size=64, target_batch_size=128, lr_start=lr, lr_end=lr)
    policy, value = PolicyNet(7, 3, (16,)), ValueNet(7, (16,))
    head = make_aux_head(policy, 0) if aux is not None and aux.enabled else None
    params = [*policy.parameters(), *value.parameters(), *([] if head is None else head.parameters())]
    if cls is ReferenceLearner:
        return cls(policy, value, cfg, CPU, _Adam(params, lr))
    return cls(policy, value, cfg, CPU, _Adam(params, lr), aux=aux, aux_head=head)


def _trainer(learner: Learner, n_agents: int = 16, horizon: int = 3) -> OverlapTrainer:
    env = Corridor(n_agents, horizon=horizon)
    actor = PolicyNet(7, 3, (16,))
    return OverlapTrainer(learner, Collector(env, actor, learner.cfg.rollout_steps(n_agents), CPU, seed=1))


def _params(learner: Learner, n: int | None = None) -> list[Any]:
    return [p.detach().clone() for p in learner.params[:n]]


def _same(a: list[Any], b: list[Any]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _one_batch(seed: int = 0) -> Any:
    torch.manual_seed(seed)
    return Collector(Corridor(16, horizon=3), PolicyNet(7, 3, (16,)), 8, CPU, seed=1).collect(0)


def _filled(learner: Learner, n: int = 32) -> None:
    aux = learner.aux
    assert aux is not None and aux.pos is not None and aux.zero is not None
    rng = np.random.default_rng(1)
    aux.pos.add(rng.normal(size=(n, 7)).astype(np.float32), rng.choice([-1, 1], n).astype(np.int8), 0)
    aux.zero.add(rng.normal(size=(n, 7)).astype(np.float32), np.zeros(n, np.int8), 0)


def test_aux_desactivee_bit_a_bit_identique_au_ppo_d_avant() -> None:
    """D8-1 et D8-8 : poids et statistiques identiques bit à bit, après 6 itérations en recouvrement, entre le code
    d'avant la SIL (`_ppo_reference`), le code actuel sans section, avec `enabled: false`, et avec la tâche activée
    mais λ = 0 (étiquetage actif, tête dans l'optimiseur, aucun calcul de gradient)."""
    variants = {
        "reference": _learner(None, cls=ReferenceLearner),
        "absente": _learner(None),
        "enabled_false": _learner(acfg(enabled=False)),
        "lambda_nul": _learner(acfg(coef=0.0)),
    }
    stats: dict[str, list[dict[str, Any]]] = {}
    aux_logs: list[dict[str, float]] = []
    for name, learner in variants.items():
        with _trainer(learner) as tr:
            res = [tr.iterate().stats for _ in range(6)]
        stats[name] = [s.as_dict() for s in res]
        if name == "lambda_nul":
            aux_logs = [dict(s.aux) for s in res]
    n_pv = len(_params(variants["reference"]))
    ref = _params(variants["reference"])
    for name in ("absente", "enabled_false", "lambda_nul"):
        assert _same(_params(variants[name], n_pv), ref), name
        assert stats[name] == stats["reference"], name
    on = variants["lambda_nul"].aux
    assert on is not None and on.ghost is None and on.pos is not None and on.pos.size > 0  # l'étiquetage tournait
    assert all(r["batch_present"] == 0.0 and r["disp_share"] == 0.0 and r["grad_share"] == 0.0 for r in aux_logs)
    assert variants["enabled_false"].aux is None and "aux" not in variants["enabled_false"].state_dict()
    assert len(variants["enabled_false"].params) == n_pv


def test_gradient_auxiliaire_nul_sur_la_sortie_de_pi_et_sur_v() -> None:
    """D8-3 (C2) : le gradient de L_aux est EXACTEMENT nul sur la couche de sortie de π et sur tous les paramètres de
    V, non nul sur le tronc et la tête ; dans le learner, après `joint_step` sur des gradients PPO mis à zéro."""
    # Arrange
    learner = _learner(acfg(coef=1.0))
    _filled(learner)
    aux = learner.aux
    assert aux is not None
    batch = _one_batch()
    it = aux.begin(
        batch.obs,
        batch.rew,
        batch.term,
        batch.trunc,
        batch.learn_mask,
        iteration=0,
        learned_per_iteration=128,
        device=CPU,
        n_ppo_steps=4,
        optimizer=learner.optimizer,
    )
    for p in learner.params:
        p.grad = torch.zeros_like(p)
    # Act
    it.joint_step()
    # Assert
    out_layer = list(learner.policy.net[-1].parameters())
    assert all(torch.count_nonzero(p.grad) == 0 for p in out_layer)
    assert all(torch.count_nonzero(p.grad) == 0 for p in learner.value.parameters())
    assert all(torch.count_nonzero(p.grad) > 0 for p in aux.trunk_params[:1])
    assert all(torch.count_nonzero(p.grad) > 0 for p in aux.head.parameters())
    # le tronc partage les modules de π (tranche de nn.Sequential, torch 2.15.0) et exclut la sortie
    ids = {id(p) for p in aux.trunk_params}
    assert ids == {id(p) for p in learner.policy.parameters()} - {id(p) for p in out_layer}


def test_state_dict_de_pi_inchange_et_tete_hors_des_poids_exportes() -> None:
    on, off = _learner(acfg()), _learner(None)
    assert list(on.policy.state_dict()) == list(off.policy.state_dict())
    assert all(id(p) not in {id(q) for q in on.policy.parameters()} for p in on.aux.head.parameters())  # type: ignore[union-attr]
    assert set(on.state_dict()) - set(off.state_dict()) == {"aux"}
    with pytest.raises(ValueError, match="auxiliaire"):
        off.load_state_dict(on.state_dict())


def test_optimiseur_sans_la_tete_et_combinaison_avec_la_sil_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    import training.ppo as ppo_mod

    torch.manual_seed(0)
    cfg = _small_cfg(minibatch_size=64, target_batch_size=128)
    policy, value = PolicyNet(7, 3, (16,)), ValueNet(7, (16,))
    opt = _Adam([*policy.parameters(), *value.parameters()], 1e-3)
    with pytest.raises(ValueError, match="tête"):
        Learner(policy, value, cfg, CPU, opt, aux=acfg())
    sil = SILConfig(enabled=True, anneal_start_timesteps=0, anneal_end_timesteps=1)
    with pytest.raises(ValueError, match="D11"):
        Learner(policy, value, cfg, CPU, opt, sil=sil, aux=acfg())
    # sans optimiseur fourni : l'Adam de PPO est créé sur π, V et la tête (fabrique substituée : Adam manuel)
    seen: list[list[Any]] = []

    def fake_make(params: Any, ppo: Any, device: Any) -> Any:
        seen.append(list(params))
        return _Adam(seen[-1], 1e-3)

    monkeypatch.setattr(ppo_mod, "make_optimizer", fake_make)
    learner = Learner(policy, value, cfg, CPU, aux=acfg())
    n_pv = len(list(policy.parameters())) + len(list(value.parameters()))
    assert learner.aux is not None and len(learner.params) == n_pv + 2
    assert [id(p) for p in seen[0]] == [id(p) for p in learner.params]
    assert {id(p) for p in learner.aux.head.parameters()} <= {id(p) for p in seen[0]}


def test_pas_de_lot_sous_le_seuil_trajectoire_du_ppo_seul() -> None:
    # 128 transitions par lot : le compartiment ± ne peut atteindre 1 000 (moitié du lot de 2 000)
    on, off, batch = _learner(acfg(batch_size=2000, capacity_per_class=1000)), _learner(None), _one_batch()
    s = on.update(batch).aux
    off.update(batch)
    assert s["batch_present"] == 0.0 and s["replay_pos_size"] > 0 and s["grad_share"] == 0.0
    n = len(off.params)
    assert _same(_params(on, n), _params(off))


def test_journal_c4_complet_fini_et_coherent() -> None:
    # Arrange
    on = _learner(acfg())
    _filled(on)
    # Act
    s = on.update(_one_batch())
    a = s.aux
    # Assert : toutes les clés de D9 (hors RAM, ajoutée par la boucle), finies ; parts dans ]0, 1[
    expected = {
        "coef", "loss", "batch_present", "recall_pos", "recall_neg", "recall_zero", "precision_pos",
        "precision_neg", "precision_zero", "frac_pos", "frac_neg", "frac_zero", "frac_masked",
        "replay_pos_size", "replay_zero_size", "replay_pos_age_mean", "grad_norm_trunk_aux",
        "grad_norm_trunk_ppo", "grad_share", "disp_share",
    }  # fmt: skip
    assert expected <= set(a) and all(math.isfinite(v) for v in a.values())
    assert a["batch_present"] == on.cfg.epochs * s.n_minibatches
    assert a["loss"] == pytest.approx(math.log(3), rel=0.05)  # tête gain 0,01 : prédiction quasi uniforme
    assert 0.0 < a["grad_share"] < 1.0 and a["grad_share"] == pytest.approx(
        a["grad_norm_trunk_aux"] / (a["grad_norm_trunk_aux"] + a["grad_norm_trunk_ppo"])
    )
    assert 0.0 < a["disp_share"] < 1.0
    assert a["frac_pos"] + a["frac_neg"] + a["frac_zero"] + a["frac_masked"] == pytest.approx(1.0)
    assert a["exits"] > 0 and a["frac_masked"] > 0 and a["frac_pos"] + a["frac_neg"] > 0
    assert a["replay_pos_new"] > 0 and a["replay_zero_new"] > 0
    assert "aux" not in s.as_dict() and s.sil == {}


def test_rappel_et_precision_sur_un_cas_manuel() -> None:
    # Arrange : tête qui prédit toujours « aucun but » (poids nuls, biais vers la classe 1)
    on = _learner(acfg(coef=1e-12))
    _filled(on)
    assert on.aux is not None
    with torch.no_grad():
        on.aux.head.weight.zero_()
        on.aux.head.bias.copy_(torch.tensor([0.0, 10.0, 0.0]))
    # Act
    a = on.update(_one_batch()).aux
    # Assert : lot rééquilibré moitié ±, moitié 0 ; tout est prédit 0
    assert (a["recall_zero"], a["recall_pos"], a["recall_neg"]) == (1.0, 0.0, 0.0)
    assert a["precision_zero"] == pytest.approx(0.5) and (a["precision_pos"], a["precision_neg"]) == (0.0, 0.0)


def test_part_de_gradient_sur_un_cas_manuel() -> None:
    """D8-8 : gradients PPO posés à la main sur le tronc ; ‖g_aux‖ recalculé indépendamment (autograd.grad de
    λ·CE sur le même lot) ; grad_share = ‖g_aux‖ / (‖g_aux‖ + ‖g_PPO‖)."""
    # Arrange
    lam = 0.5
    learner = _learner(acfg(coef=lam))
    _filled(learner)
    aux = learner.aux
    assert aux is not None
    batch = _one_batch()
    it = aux.begin(
        batch.obs,
        batch.rew,
        batch.term,
        batch.trunc,
        batch.learn_mask,
        iteration=0,
        learned_per_iteration=128,
        device=CPU,
        n_ppo_steps=1,
        optimizer=learner.optimizer,
    )
    assert it._obs is not None and it._y is not None
    g_ppo = [torch.full_like(p, 0.01 * (i + 1)) for i, p in enumerate(aux.trunk_params)]
    for p in learner.params:
        p.grad = None
    for p, g in zip(aux.trunk_params, g_ppo, strict=True):
        p.grad = g.clone()
    x, y = it._obs[0].float(), it._y[0]
    ce = torch.nn.functional.cross_entropy(aux_logits(aux.trunk, aux.head, x), y)
    g_aux = torch.autograd.grad(lam * ce, aux.trunk_params)
    # Act
    it.joint_step()
    torch.nn.utils.clip_grad_norm_(learner.params, 0.5)
    learner.optimizer.step()
    total = torch.tensor(1.0)  # facteur de clip non utilisé par grad_share
    it.after_step(total, 0.5)
    s = it.stats()
    # Assert
    n_aux = math.sqrt(sum(float(g.pow(2).sum()) for g in g_aux))
    n_ppo = math.sqrt(sum(float(g.pow(2).sum()) for g in g_ppo))
    assert s.grad_norm_trunk_aux == pytest.approx(n_aux, rel=1e-5)
    assert s.grad_norm_trunk_ppo == pytest.approx(n_ppo, rel=1e-6)
    assert s.grad_share == pytest.approx(n_aux / (n_aux + n_ppo), rel=1e-5) and s.loss == pytest.approx(
        float(ce.detach())
    )


def _delta(before: list[Any], params: list[Any]) -> list[Any]:
    return [p.detach() - b for p, b in zip(params, before, strict=True)]


def test_decomposition_du_deplacement_exacte_adam_manuel() -> None:
    """D8-8 : Σ_k Δθ^ppo_k + Σ_k Δθ^aux_k = Δθ réel du tronc, à 1e-5 relatif près, sur 3 mises à jour de suite (moment
    fantôme continu d'une itération à l'autre), Adam manuel de `test_ppo` (même formule que torch)."""
    on = _learner(acfg(coef=0.3))
    _filled(on)
    assert on.aux is not None
    torch.manual_seed(0)
    col = Collector(Corridor(16, horizon=3), PolicyNet(7, 3, (16,)), 8, CPU, seed=1)
    for _ in range(3):
        before = [p.detach().clone() for p in on.aux.trunk_params]
        full = on.update(dataclasses.replace(copy.deepcopy(col.collect(0)), policy_version=on.version))
        assert on.aux.last_disp is not None
        d_ppo, d_aux = on.aux.last_disp
        real = _delta(before, on.aux.trunk_params)
        err = math.sqrt(sum(float((a + b - r).pow(2).sum()) for a, b, r in zip(d_ppo, d_aux, real, strict=True)))
        norm = math.sqrt(sum(float(r.pow(2).sum()) for r in real))
        assert norm > 0 and err / norm <= 1e-5
        assert full.aux["batch_present"] == on.cfg.epochs * full.n_minibatches and 0.0 < full.aux["disp_share"] < 1.0


def test_reprise_exacte_de_l_etat_auxiliaire_par_le_checkpoint() -> None:
    # Arrange : 5 lots figés ; A : 3 mises à jour, état passé par encode/decode vers B neuf ; puis les 2 mêmes lots
    torch.manual_seed(0)
    col = Collector(Corridor(16, horizon=3), PolicyNet(7, 3, (16,)), 8, CPU, seed=1)
    batches = [copy.deepcopy(col.collect(0)) for _ in range(5)]
    a = _learner(acfg())
    _filled(a)
    for bt in batches[:3]:
        a.update(dataclasses.replace(bt, policy_version=a.version))
    saved = decode(encode(copy.deepcopy(a.state_dict())))
    b = _learner(acfg())
    b.load_state_dict(saved)
    # Act
    tail_a = [a.update(dataclasses.replace(bt, policy_version=a.version)).aux for bt in batches[3:]]
    tail_b = [b.update(dataclasses.replace(bt, policy_version=b.version)).aux for bt in batches[3:]]
    # Assert : poids (tête comprise), journaux, rejeu, fenêtre, moment fantôme, générateur identiques
    assert all(x["batch_present"] > 0 for x in tail_a) and tail_a == tail_b
    assert _same(_params(a), _params(b))
    xa, xb = a.aux, b.aux
    assert xa is not None and xb is not None and xa.pos is not None and xb.pos is not None
    assert xa.pos.size == xb.pos.size and np.array_equal(xa.pos.obs, xb.pos.obs)
    assert xa.window is not None and xb.window is not None and np.array_equal(xa.window.obs, xb.window.obs)
    assert xa.window.cursor == xb.window.cursor and np.array_equal(xa.window.length, xb.window.length)
    assert xa.ghost is not None and xb.ghost is not None and _same(xa.ghost, xb.ghost)
    assert torch.equal(xa.generator.get_state(), xb.generator.get_state())


def test_calendrier_optionnel_libere_fenetre_et_rejeu() -> None:
    # 128 appris par mise à jour : λ plein à 0, rampe à 128, 0 et libération à 256
    on = _learner(acfg(anneal_start_timesteps=64, anneal_end_timesteps=256))
    _filled(on)
    torch.manual_seed(0)
    col = Collector(Corridor(16, horizon=3), PolicyNet(7, 3, (16,)), 8, CPU, seed=1)
    logs = [on.update(dataclasses.replace(col.collect(0), policy_version=on.version)).aux for _ in range(3)]
    assert [r["coef"] for r in logs] == pytest.approx([0.1, 0.1 * (256 - 128) / 192, 0.0])
    assert logs[-1]["released"] == 1.0 and logs[-1]["batch_present"] == 0.0
    assert on.aux is not None and on.aux.window is None and on.aux.pos is None and on.aux.ghost is None
    other = _learner(acfg(anneal_start_timesteps=64, anneal_end_timesteps=256))
    other.load_state_dict(decode(encode(on.state_dict())))
    assert other.aux is not None and other.aux.released and other.aux.pos is None


# ---------------------------------------------------------------------------------------------------- θ₀


def _sha(weights: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(weights):
        h.update(name.encode("utf-8"))
        h.update(weights[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_theta0_de_pi_identique_a_std_s0_et_generateur_global_intact() -> None:
    """D8-7 : aux dimensions du run (obs 109, 90 actions, 3×256, graine 0), la tête vient d'un générateur propre :
    θ₀ de π = SHA-256 de std s0 (méthode de `test_run_4a_interventions.py:414`), état du générateur global inchangé."""
    torch.manual_seed(0)
    policy = PolicyNet(109, 90)
    state = torch.get_rng_state()
    head = make_aux_head(policy, 0)
    assert torch.equal(torch.get_rng_state(), state)
    assert _sha(policy.state_dict()) == THETA0_STD_S0
    assert head.weight.shape == (3, 256) and torch.count_nonzero(head.bias) == 0
    w = head.weight.detach()
    assert torch.allclose(w @ w.T, 1e-4 * torch.eye(3), atol=1e-9)  # orthogonale, gain 0,01


AUX_RUN = {"enabled": True, "horizon_steps": 2, "batch_size": 8, "capacity_per_class": 64, "store_float16": False}
GOALS = {(s, a): ("blue", 1) for s in (3, 11, 19) for a in range(4)}


def _ckpt_learner(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = CheckpointManager.load(path)["trainer"]["learner"]
    return out


def test_run_aux_desactivee_identique_et_theta0_identique(tmp_path: Path) -> None:
    calls: list[int] = []

    def probe() -> tuple[float, float]:
        calls.append(1)
        return 1.0, 1.0

    keep = {"every_iterations": 2, "keep": 100}  # section opérationnelle : le checkpoint 0 (θ₀) reste sur disque
    go(cfg_of(checkpoint=keep), tmp_path / "a", env_factory=make_factory(goals=GOALS))
    off = cfg_of(checkpoint=keep, aux={**AUX_RUN, "enabled": False})
    go(off, tmp_path / "b", env_factory=make_factory(goals=GOALS), memory_probe=probe)
    on = cfg_of(checkpoint=keep, aux=AUX_RUN)
    go(on, tmp_path / "c", env_factory=make_factory(goals=GOALS), memory_probe=lambda: (4096.0, 1.0))
    la, lb = (CheckpointManager(tmp_path / x / "checkpoints").load_latest() for x in ("a", "b"))
    assert la is not None and lb is not None
    for net in ("policy", "value"):
        assert all(
            torch.equal(v, lb[1]["trainer"]["learner"][net][k]) for k, v in la[1]["trainer"]["learner"][net].items()
        )
    strip = ("t", "perf/agent_steps_per_s")
    ra = [{k: v for k, v in r.items() if k not in strip} for r in rows(tmp_path / "a")]
    rb = [{k: v for k, v in r.items() if k not in strip} for r in rows(tmp_path / "b")]
    assert ra == rb and not calls and not any(k.startswith("aux/") for k in rb[-1])
    # θ₀ (checkpoint 0) : mêmes poids de π avec et sans tâche auxiliaire
    c0 = {x: next((tmp_path / x / "checkpoints").glob("ckpt_000000_it00000000.pt"), None) for x in ("a", "c")}
    assert c0["a"] is not None and c0["c"] is not None
    assert _sha(_ckpt_learner(c0["a"])["policy"]) == _sha(_ckpt_learner(c0["c"])["policy"])
    # l'évaluation et l'export chargent π sans la tête
    from eval.match import load_checkpoint_policy

    final = CheckpointManager(tmp_path / "c" / "checkpoints").load_latest()
    assert final is not None and "aux" in final[1]["trainer"]["learner"]
    _, info = load_checkpoint_policy(final[0], mode="argmax", seed=0, obs_dim=7, tick_skip=8)
    assert info.iteration == final[1]["iteration"]


def test_run_aux_journal_et_reprise_exacte(tmp_path: Path) -> None:
    # Arrange : 6 itérations de 32 appris, buts aux pas 3, 11, 19
    cfg = cfg_of(aux=AUX_RUN)
    probe = lambda: (4096.0, 512.0)  # noqa: E731
    a, b = tmp_path / "a", tmp_path / "b"
    go(cfg, a, env_factory=make_factory(goals=GOALS), memory_probe=probe)
    go(cfg, b, env_factory=make_factory(goals=GOALS), memory_probe=probe, max_iterations=3)
    go(cfg, b, env_factory=make_factory(goals=GOALS), memory_probe=probe, resume=True)
    # Assert : journal C4 à chaque itération, RAM journalisée, lots appliqués ; reprise = run continu
    ra = rows(a)
    assert all("aux/grad_share" in r and r["aux/ram_available_mb"] == 4096.0 for r in ra)
    assert any(r["aux/batch_present"] > 0 for r in ra) and any(r["aux/batch_present"] > 0 for r in ra[3:])
    strip = ("t", "perf/agent_steps_per_s")
    assert [{k: v for k, v in r.items() if k not in strip} for r in ra] == [
        {k: v for k, v in r.items() if k not in strip} for r in rows(b)
    ]
    wa, wb = (
        _ckpt_learner(sorted((a / "checkpoints").glob("*.pt"))[-1]),
        _ckpt_learner(sorted((b / "checkpoints").glob("*.pt"))[-1]),
    )
    for part in ("policy", "value"):
        assert all(torch.equal(v, wb[part][k]) for k, v in wa[part].items())
    assert all(torch.equal(v, wb["aux"]["head"][k]) for k, v in wa["aux"]["head"].items())


def test_garde_aux_alerte_apres_la_patience_et_ks1() -> None:
    cfg = AuxConfig(enabled=True, share_alert_patience=3, ram_low_max_s=10.0)
    st = AuxGuardState()
    for i in range(3):
        st, alerts, stop = aux_guard(cfg, st, share=0.6, ram_available_mb=2000.0, iteration_s=1.0)
        assert stop is None and bool(alerts) == (i == 2)
    st, alerts, _ = aux_guard(cfg, st, share=0.4, ram_available_mb=None, iteration_s=1.0)
    assert st.share_alert_streak == 0 and not alerts
    for _ in range(2):
        st, _, stop = aux_guard(cfg, st, share=0.0, ram_available_mb=500.0, iteration_s=6.0)
    assert stop is not None and stop[0] == "aux_ram" and st.ram_low_s == 12.0
    st, _, stop = aux_guard(cfg, st, share=0.0, ram_available_mb=1500.0, iteration_s=6.0)
    assert stop is None and st.ram_low_s == 0.0


def test_run_stop_ks1_et_alerte_c4_journalisees(tmp_path: Path) -> None:
    cfg = cfg_of(aux={**AUX_RUN, "ram_min_available_mb": 1000.0, "ram_low_max_s": 0.0})
    s = go(cfg, tmp_path / "r", memory_probe=lambda: (500.0, 100.0))
    assert s.status == "stopped" and s.cause is not None and s.cause.startswith("aux_ram") and s.end_iteration == 1
    cfg = cfg_of(aux={**AUX_RUN, "share_alert": 1e-9, "share_alert_patience": 1})
    go(cfg, tmp_path / "a", env_factory=make_factory(goals=GOALS), memory_probe=lambda: (4096.0, 1.0))
    ev = events(tmp_path / "a")
    assert any(e.get("event") == "alert" and e.get("rule") == "aux_share" for e in ev)
    final = CheckpointManager(tmp_path / "a" / "checkpoints").load_latest()
    assert final is not None and final[1]["extra"]["aux_guard"]["share_alert_streak"] >= 1


# ---------------------------------------------------------------------------------------------------- mémoire, float16


def test_memoire_aux_dimensions_du_run_int_aux_mesuree_en_octets() -> None:
    """Octets des tableaux aux dimensions de `configs/int_aux.yaml` : 7 373 agents appris, obs 109, H = 100 (repli de
    D2, banc du 2026-11-22), float32 (amendement D3 du 2026-11-23, parité D8-5 en échec), 2 × 125 000 transitions
    (obs 436, étiquette 1, itération 4 octets) : 321,5 + 110,3 = 431,8 Mo ; 216,5 Mo en float16 (banc du 2026-11-22)."""
    cfg = RunConfig.from_yaml(CONFIGS / "int_aux.yaml")
    assert cfg.aux is not None and not cfg.aux.store_float16
    n_learn = cfg.batch_learned_steps // cfg.collection.steps_per_batch
    window = GoalWindow(n_learn, 109, cfg.aux.horizon_steps, cfg.aux.obs_dtype)
    replay = [ClassReplay(cfg.aux.capacity_per_class, 109, cfg.aux.obs_dtype) for _ in range(2)]
    assert n_learn == 7_373
    assert window.nbytes == 7_373 * 100 * 436 + 7_373 * 8 == 321_521_784
    assert sum(r.nbytes for r in replay) == 250_000 * 441 == 110_250_000
    assert window.nbytes + sum(r.nbytes for r in replay) == expected_nbytes(n_learn, 109, cfg.aux) == 431_771_784
    # référence float16 à H = 100 (banc D5 λ = 0,1 du 2026-11-22) ; l'option float32 à H = 100 dépasse l'état SIL du
    # banc 2 (243 367 984 + 58 250 000) : la règle D2 repose sur le banc RAM float32 dédié, plus sur cette comparaison
    assert expected_nbytes(n_learn, 109, dataclasses.replace(cfg.aux, store_float16=True)) == 216_540_384
    assert expected_nbytes(n_learn, 109, cfg.aux) > 301_617_984
    del window, replay


def test_taille_serialisee_de_l_etat_auxiliaire_proche_des_tableaux() -> None:
    task = _task(acfg(capacity_per_class=1000, horizon_steps=50, store_float16=True), obs_dim=109)
    b = steps_arrays(60, 40, 109)
    b["rew"][-1], b["term"][-1] = 1.0, True
    task.ingest(b["obs"], b["rew"], b["term"], b["trunc"], np.ones(40, np.bool_), 0)
    f = io.BytesIO()
    torch.save(encode(task.state_dict()), f)
    assert task.window is not None and task.pos is not None and task.zero is not None
    filled = task.window.nbytes + (task.pos.size + task.zero.size) * (109 * 2 + 1 + 4)  # rejeu : cases remplies
    assert task.pos.size > 0 and task.zero.size > 0
    assert filled <= f.tell() <= filled * 1.1 + 50_000


def test_parite_float16_de_la_tete_sur_un_lot_synthetique() -> None:
    """D8-5 (lot synthétique) : |Δ log softmax(tête)| max ≤ 1e-2 entre obs float32 et float16, obs du vrai builder,
    réseaux v0 ; tête initiale (gain 0,01) et tête plus piquée (gain 1)."""
    rng = np.random.default_rng(0)
    obs = build_obs(random_state(rng, 2048, 2), prev_of(rng, 4096))
    torch.manual_seed(0)
    policy = PolicyNet(109)
    trunk = policy_trunk(policy)
    for gain in (0.01, 1.0):
        head = AuxHead(256)
        g = torch.Generator()
        g.manual_seed(3)
        head.init_from(g, gain=gain)
        res = afp.aux_f16_parity(trunk, head, obs)
        assert res["passed"] and res["dlogp_head_max"] <= 1e-2, res
    huge = obs.copy()
    huge[0, 0] = 1e6  # dépasse float16 : échec signalé
    assert not afp.aux_f16_parity(trunk, head, huge)["passed"]


# ---------------------------------------------------------------------------------------------------- outils de banc


@pytest.mark.parametrize(
    ("benches", "decision", "coef"),
    [
        ([], "BANC", 0.1),
        ([(0.1, 0.2)], "RETENIR", 0.1),
        ([(0.1, 0.7)], "BANC", 0.1 / math.sqrt(10)),
        ([(0.1, 0.7), (0.1 / math.sqrt(10), 0.4)], "RETENIR", 0.1 / math.sqrt(10)),
        ([(0.1, 0.7), (0.03, 0.6)], "BANC", 0.01),  # « 0,03 » de l'ADR = 0,1/√10 (comparaison à 10 % près)
        ([(0.1, 0.01), (0.3, 0.2), (1.0, 0.7)], "RETENIR", 0.3),  # λ retenu = valeur réellement mesurée
        ([(0.1, 0.7), (0.0316, 0.6), (0.01, 0.55)], "PAS DE GO", None),
        ([(0.1, 0.01)], "BANC", 0.1 * math.sqrt(10)),
        ([(0.1, 0.01), (0.316, 0.6)], "RETENIR", 0.1),
        ([(0.1, 0.01), (0.316, 0.2)], "BANC", 1.0),
        ([(0.1, 0.01), (0.316, 0.2), (1.0, 0.7)], "RETENIR", 0.316),
        ([(0.1, 0.01), (0.316, 0.02), (1.0, 0.03)], "RETENIR", 1.0),
        ([(0.1, 0.9), (0.1, 0.3)], "RETENIR", 0.1),  # λ rejoué : le dernier banc fait foi
    ],
)
def test_regle_d5_de_choix_de_lambda(benches: list[tuple[float, float]], decision: str, coef: float | None) -> None:
    out = abs_.choose_lambda(benches)
    assert out["decision"] == decision
    got = out["next_coef"] if decision == "BANC" else out["retained_coef"]
    if coef is None:
        assert got is None
    else:
        assert got == pytest.approx(coef, rel=1e-3)


def test_regle_d5_note_sous_5_pourcent_et_regle_ram_d2() -> None:
    assert abs_.choose_lambda([(0.1, 0.01), (0.316, 0.02), (1.0, 0.03)])["note"] is not None
    assert abs_.choose_lambda([(0.1, 0.2)])["note"] is None
    assert abs_.ram_rule(150, 2000.0)["decision"] == "OK"
    assert abs_.ram_rule(150, 1400.0)["decision"] == "H = 100"
    assert abs_.ram_rule(100, 1400.0)["decision"] == "PAS DE GO"
    assert abs_.ram_rule(None, 1400.0)["decision"] == "NON CONCLUANT"


def _aux_rows(n: int, *, dt: float = 2.0, size: float = 64.0, present: float = 4.0, coef: float = 0.1) -> list[Any]:
    return [
        {
            "step": i + 1,
            "t": 1000.0 + i * dt,
            "aux/grad_share": 0.1 + 0.001 * i,
            "aux/disp_share": 0.2,
            "aux/replay_pos_size": size,
            "aux/replay_zero_size": size,
            "aux/batch_present": present,
            "aux/coef": coef,
        }
        for i in range(n)
    ]


def test_fenetre_concluante_seulement_si_rejeu_plein_et_lots_presents() -> None:
    ok = abs_.aux_window(_aux_rows(400), capacity=64)
    assert ok["conclusive"] and ok["rows"] == 301 and ok["grad_share"]["median"] == pytest.approx(0.1 + 0.001 * 249)
    assert not abs_.aux_window(_aux_rows(400, size=63.0), capacity=64)["conclusive"]
    assert not abs_.aux_window(_aux_rows(400, present=0.0), capacity=64)["conclusive"]
    assert not abs_.aux_window(_aux_rows(100), capacity=64)["conclusive"]  # 198 s < 10 min
    assert not abs_.aux_window(_aux_rows(400), capacity=None)["conclusive"]


def test_occupation_du_fil_learner() -> None:
    ticks = [{"t_bench": 10.0, "learn_s": 5.0}] + [{"t_bench": 10.0 + i, "learn_s": 0.25} for i in range(1, 801)]
    occ = abs_.learner_occupancy(ticks)
    assert occ["global"] == pytest.approx(0.25) and occ["window"] == pytest.approx(0.25) and occ["loop_s"] == 800.0
    assert abs_.learner_occupancy([])["global"] is None


def test_controle_smoke_rejeu_conserve_et_reprise() -> None:
    rows_ = []
    pos = zero = 0.0
    for s in range(1, 7):
        new_p, new_z = 10.0, 30.0
        pos, zero = min(64.0, pos + new_p), min(64.0, zero + new_z)
        rows_.append(
            {
                "step": s,
                "aux/coef": 0.1,
                "aux/batch_present": 4.0 if s >= 2 else 0.0,
                "aux/loss": 1.0,
                "aux/replay_pos_size": pos,
                "aux/replay_zero_size": zero,
                "aux/replay_pos_new": new_p,
                "aux/replay_zero_new": new_z,
                "aux/released": 0.0,
                "aux/grad_share": 0.1,
                "aux/ram_available_mb": 4000.0,
            }
        )
    out = abs_.smoke_checks(rows_, capacity=64, resume_starts=[3], smoke_verdict="PASS")
    assert out["verdict"] == "PASS", out["checks"]
    rows_[4] = {**rows_[4], "aux/replay_pos_size": rows_[4]["aux/replay_pos_size"] - 1.0}
    bad = abs_.smoke_checks(rows_, capacity=64, resume_starts=[3], smoke_verdict="PASS")
    assert bad["verdict"] == "FAIL" and bad["replay_breaks"] == [5, 6]


def test_resume_des_bancs_sur_dossier_synthetique(tmp_path: Path) -> None:
    # Arrange : dossier minimal au format de bench_run (méta, ticks, journal, échantillons, run_meta)
    import json

    d = tmp_path / "bench_int_aux_x"
    d.mkdir()
    (d / "bench_meta.json").write_text(
        json.dumps({"learned_per_iter": 100, "collected_per_iter": 120, "provenance": {"git_commit": "abc"}}),
        encoding="utf-8",
    )
    (d / "bench_end.json").write_text(json.dumps({"run": {"start_iteration": 0}}), encoding="utf-8")
    (d / "run_meta.json").write_text(
        json.dumps({"config": {"aux": {"enabled": True, "capacity_per_class": 64, "horizon_steps": 150}}}),
        encoding="utf-8",
    )
    metrics = _aux_rows(400)
    for r in metrics:
        r["aux/ram_available_mb"] = 3000.0
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metrics), encoding="utf-8")
    ticks = [{"n": i, "t": 2.0 * i, "t_bench": 5.0 + 2.0 * i, "gap_s": 2.0, "learn_s": 0.5} for i in range(401)]
    (d / "bench_ticks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ticks), encoding="utf-8")
    samples = [{"unix": 1000.0 + i, "avail": 3000.0 * 1024 * 1024} for i in range(800)]
    (d / "bench_samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")
    (d / "events.jsonl").write_text("", encoding="utf-8")
    # Act
    code = abs_.main([str(d)])
    # Assert : λ = 0,1 retenu (médiane ≈ 0,35 ∈ [0,05 ; 0,5]), occupation 25 %, RAM tenue, JSON écrit
    out = json.loads((d / "aux_summary.json").read_text(encoding="utf-8"))
    assert code == 0 and out["lambda"]["decision"] == "RETENIR" and out["lambda"]["retained_coef"] == 0.1
    b = out["benches"][0]
    assert b["learner_occupancy"]["global"] == pytest.approx(0.25) and b["ram_rule"]["decision"] == "OK"
    assert b["aux"]["conclusive"] and b["throughput"]["global"]["learned_wall"] is not None
