"""Outillage des préalables D7 de l'ADR-009 (sans run, sans simulateur, CPU) : configs du banc et du smoke SIL,
parité float16 sur lot (`scripts/sil_f16_parity.py`), résumé SIL du banc et vérification du smoke
(`scripts/sil_bench_summary.py`)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from test_obs import prev_of, random_state
from test_run_4a_interventions import _diff

from training import RunConfig

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def _load(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


par = _load("sil_f16_parity")
sbs = _load("sil_bench_summary")


def _raw(name: str) -> dict[str, Any]:
    out: dict[str, Any] = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    return out


def _cfg(name: str) -> dict[str, Any]:
    return RunConfig.from_yaml(CONFIGS / name).to_dict()


# ------------------------------------------------------------------------------------------------ configs


def test_bench_int_sil_ne_differe_de_int_sil_que_par_le_nom_et_la_rotation() -> None:
    # Arrange / Act
    raw_diff = _diff(_raw("int_sil.yaml"), _raw("bench_int_sil.yaml"))
    cfg_diff = _diff(_cfg("int_sil.yaml"), _cfg("bench_int_sil.yaml"))
    # Assert : `name` et `checkpoint.keep` seulement, en YAML brut comme en config construite (défauts compris)
    expected = {("name",): ("int_sil", "bench_int_sil"), ("checkpoint", "keep"): (2, 1000)}
    assert raw_diff == expected
    assert cfg_diff == expected
    a = RunConfig.from_yaml(CONFIGS / "int_sil.yaml").trajectory_dict()
    b = RunConfig.from_yaml(CONFIGS / "bench_int_sil.yaml").trajectory_dict()
    assert a.pop("name") != b.pop("name") and a == b  # même trajectoire hors nom (checkpoint : section opérationnelle)


SMOKE_SIL_SCALED = {"anneal_start_timesteps": 500_000, "anneal_end_timesteps": 900_000, "batch_size": 256}


def test_smoke_sil_est_smoke_plus_le_bloc_sil_de_int_sil_mis_a_l_echelle() -> None:
    # Arrange
    sil_run = _raw("int_sil.yaml")["sil"]
    expected_sil = {**sil_run, **SMOKE_SIL_SCALED}
    # Act
    raw_diff = _diff(_raw("smoke.yaml"), _raw("smoke_sil.yaml"))
    cfg_diff = _diff(_cfg("smoke.yaml"), _cfg("smoke_sil.yaml"))
    # Assert : hors `name`, uniquement le bloc `sil` (absent de smoke.yaml) ...
    expected = {("name",): ("smoke", "smoke_sil"), **{("sil", k): (None, v) for k, v in expected_sil.items()}}
    assert raw_diff == expected
    assert {k: v for k, v in cfg_diff.items() if k != ("name",)} == {
        k: (None, v) for k, v in _flat_sil(_cfg("smoke_sil.yaml")).items()
    }
    # ... qui ne diffère de celui de int_sil que par les trois clés mises à l'échelle
    sil_cfg_run, sil_cfg_smoke = _cfg("int_sil.yaml")["sil"], _cfg("smoke_sil.yaml")["sil"]
    assert {k for k in sil_cfg_run if sil_cfg_run[k] != sil_cfg_smoke[k]} == set(SMOKE_SIL_SCALED)
    assert sil_cfg_smoke["enabled"] is True and sil_cfg_smoke["mode"] == "joint"


def _flat_sil(cfg: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    # mode joint : `updates_per_iteration` et `max_grad_norm` valent None des deux côtés, donc hors du diff
    return {("sil", k): v for k, v in cfg["sil"].items() if v is not None}


def test_smoke_sil_exerce_plateau_reprise_annealing_et_liberation() -> None:
    # Arrange : mêmes dérivés que `scripts/train.py --smoke` (reprise à ceil(0,5 · n)) et `Learner.update`
    # (c_sil de la mise à jour v = coef_at(v · appris par lot), `SelfImitation.begin`)
    cfg = RunConfig.from_yaml(CONFIGS / "smoke_sil.yaml")
    assert cfg.sil is not None and cfg.smoke is not None
    n, per = cfg.total_iterations, cfg.batch_learned_steps
    half = math.ceil(cfg.smoke.resume_at_fraction * n)
    # Act
    coefs = [cfg.sil.coef_at(v * per) for v in range(n)]
    # Assert
    assert (n, per, half) == (18, 57_344, 9)
    assert coefs[:9] == [0.1] * 9  # 1re partie : plateau
    assert 0.0 < coefs[half] < 0.1  # la 2e partie repart avec SIL active (état SIL rechargé et utilisé)
    assert all(c > 0.0 for c in coefs[:16]) and coefs[16:] == [0.0, 0.0]  # libération à la mise à jour 16
    assert cfg.sil.anneal_end_timesteps < cfg.budget.total_timesteps  # C2 (sinon RunConfig refuse)
    assert cfg.sil.capacity >= cfg.sil.batch_size and cfg.sil.horizon_steps == 300


# ------------------------------------------------------------------------------------------------ parité float16


def _nets(seed: int = 0) -> tuple[Any, Any]:
    from models import PolicyNet, ValueNet

    torch.manual_seed(seed)
    return PolicyNet(109), ValueNet(109)


def _real_like_obs(n_envs: int = 512) -> np.ndarray:
    from obs import build_obs

    rng = np.random.default_rng(0)
    out: np.ndarray = build_obs(random_state(rng, n_envs, 2), prev_of(rng, 2 * n_envs))
    return out


def test_parite_float16_passe_sur_un_mini_lot_d_obs_du_builder() -> None:
    # Arrange : obs du vrai builder (états physiques tirés), réseaux v0 initiaux
    policy, value = _nets()
    obs = _real_like_obs()
    act = np.arange(obs.shape[0], dtype=np.int64) % 90
    # Act
    res = par.f16_parity(policy, value, obs, act, chunk=300)  # plusieurs tranches, dont une partielle
    # Assert
    assert res["passed"] and all(res["checks"].values())
    assert res["n_obs"] == 1024 and res["obs_dim"] == 109 and res["obs_f16_nonfinite_rows"] == 0
    assert 0.0 <= res["dlogp_taken_max"] <= res["dlogp_all_max"] <= par.MAX_DLOGP
    assert 0.0 < res["dv_max"] <= par.MAX_DV
    assert res["obs_round_abs_max"] > 0.0  # l'arrondi a bien eu lieu


def test_parite_float16_echoue_au_dela_des_seuils_et_sur_depassement() -> None:
    # Arrange : critique 1e4 fois plus raide (|ΔV| amplifié d'autant) ; obs hors de la plage float16
    policy, value = _nets()
    obs = _real_like_obs(64)
    with torch.no_grad():
        value.net[-1].weight.mul_(1e4)
    big = obs.copy()
    big[0, 0] = 1e5
    # Act
    steep = par.f16_parity(policy, value, obs)
    overflow = par.f16_parity(*_nets(), big)
    # Assert
    assert not steep["passed"] and not steep["checks"]["dv_max_le_1e-3"] and steep["dv_max"] > par.MAX_DV
    assert steep["dlogp_taken_max"] is None
    assert not overflow["passed"] and overflow["obs_f16_nonfinite_rows"] == 1
    with pytest.raises(ValueError):
        par.f16_parity(policy, value, obs.astype(np.float64))


def test_obs_des_agents_appris_et_largeurs_des_reseaux() -> None:
    # Arrange : lot (T=2, N=3) dont l'agent 1 est gelé
    obs = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    act = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    batch = SimpleNamespace(obs=obs, act=act, learn_mask=np.array([True, False, True]))
    policy, value = _nets()
    # Act
    o, a = par.learned_obs(batch)
    # Assert
    assert o.shape == (4, 4) and a.tolist() == [1, 3, 4, 6]
    np.testing.assert_array_equal(o[1], obs[0, 2])
    assert par.mlp_hidden(policy.state_dict()) == ((256, 256, 256), 109, 90)
    assert par.mlp_hidden(value.state_dict()) == ((512, 512, 512), 109, 1)
    assert par.same_weights(policy.state_dict(), policy.state_dict())
    assert not par.same_weights(policy.state_dict(), _nets(1)[0].state_dict())


def _write_ckpt(directory: Path, *, with_value: bool, with_pending: bool, policy: Any, value: Any) -> Path:
    from training.checkpoint import CHECKPOINT_FORMAT_VERSION, CheckpointIdentity, CheckpointManager
    from training.rollout import RolloutBatch

    obs = _real_like_obs(32).reshape(2, 32, 109)  # T = 2, N = 32 agents
    t, n = obs.shape[:2]
    mask = np.ones(n, dtype=bool)
    mask[::4] = False  # un agent sur quatre gelé
    batch = RolloutBatch(
        obs=obs,
        act=np.zeros((t, n), dtype=np.int64),
        logp=np.zeros((t, n), dtype=np.float32),
        rew=np.zeros((t, n), dtype=np.float32),
        term=np.zeros((t, n), dtype=bool),
        trunc=np.zeros((t, n), dtype=bool),
        final_obs=np.zeros((0, 109), dtype=np.float32),
        final_t=np.zeros(0, dtype=np.int64),
        final_agent=np.zeros(0, dtype=np.int64),
        last_obs=obs[-1].copy(),
        learn_mask=mask,
        policy_version=3,
    )
    learner: dict[str, Any] = {"policy": policy.state_dict()}
    if with_value:
        learner["value"] = value.state_dict()
    ident = CheckpointIdentity.current(reward_config_hash="r" * 64, obs_dim=109, tick_skip=8)
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "identity": dataclasses.asdict(ident),
        "iteration": 7,
        "train_config_hash": "h",
        "versions": {"git_commit": "abc"},
        "trainer": {"learner": learner, **({"pending": batch} if with_pending else {})},
    }
    return CheckpointManager(directory, keep=10).save(state, healthy=True)


def test_script_de_parite_sur_checkpoints_synthetiques(tmp_path: Path) -> None:
    # Arrange : un checkpoint de reprise (π, V, lot en attente) et un instantané (π seule, mêmes poids)
    policy, value = _nets()
    full = _write_ckpt(tmp_path / "ck", with_value=True, with_pending=True, policy=policy, value=value)
    snap = _write_ckpt(tmp_path / "snap", with_value=False, with_pending=False, policy=policy, value=value)
    out1, out2, out3 = tmp_path / "o" / "full.json", tmp_path / "o" / "snap.json", tmp_path / "o" / "bad.json"
    # Act
    code_full = par.main(["--policy", str(full), "--out", str(out1)])
    code_snap = par.main(["--policy", str(snap), "--value", str(full), "--out", str(out2)])
    code_again = par.main(["--policy", str(full), "--out", str(out1)])  # sortie existante
    code_noval = par.main(["--policy", str(snap), "--out", str(out3)])  # instantané sans critique
    # Assert
    assert (code_full, code_snap, code_again, code_noval) == (0, 0, 2, 2)
    assert not out3.exists()
    r1, r2 = (json.loads(p.read_text(encoding="utf-8")) for p in (out1, out2))
    assert r1["verdict"] == r2["verdict"] == "PASS"
    assert r1["n_obs"] == 2 * 24 and r1["obs_source"]["learned_agents"] == 24  # agents appris seulement
    assert r2["policy_equals_obs_file_policy"] is True and r2["files"]["policy"]["path"] == str(snap)
    for key in ("command", "git_commit", "git_dirty", "date_utc", "platform", "processor", "torch"):
        assert key in r1["provenance"]
    assert r1["thresholds"]["dlogp_all_max"] == 1e-2 and r1["thresholds"]["dv_max"] == 1e-3


# ------------------------------------------------------------------------------------------------ résumé SIL


@pytest.mark.parametrize(
    ("coef", "median", "decision", "next_coef"),
    [
        (0.1, 0.6, "RÉDUIRE", 0.1 / math.sqrt(10)),
        (0.1 / math.sqrt(10), 0.51, "RÉDUIRE", 0.01),
        (0.01, 0.7, "PAS DE GO", None),
        (0.1, 0.5, "RETENIR", None),
        (0.0316, 0.2, "RETENIR", None),
    ],
)
def test_regle_d5_mecanique(coef: float, median: float, decision: str, next_coef: float | None) -> None:
    out = sbs.calibration(coef, median, True)
    assert out["decision"] == decision
    assert out["next_coef"] == (pytest.approx(next_coef) if next_coef is not None else None)
    if next_coef == 0.01:
        assert out["next_coef"] == 0.01  # 0,1 -> 0,0316 -> 0,01 exactement (échelle de D5)


def test_regle_d5_note_sous_5_pourcent_et_non_concluant() -> None:
    low = sbs.calibration(0.1, 0.02, True)
    assert low["decision"] == "RETENIR" and low["note"] is not None
    assert sbs.calibration(0.0316, 0.02, True)["note"] is None  # la note ne vaut qu'à c_sil = 0,1
    assert sbs.calibration(0.1, 0.9, False)["decision"] == "NON CONCLUANT"
    assert sbs.calibration(None, None, True)["decision"] == "NON CONCLUANT"


def _sil_rows(
    n_min: float = 15.0, dt: float = 1.0, full_after_s: float = 250.0, cap: int = 1000
) -> list[dict[str, Any]]:
    rows = []
    t0 = 1_800_000_000.0
    for i in range(int(n_min * 60 / dt)):
        t = i * dt
        rows.append(
            {
                "step": i + 1,
                "t": t0 + t,
                "sil/coef": 0.1,
                "sil/disp_share": 0.9 if t < 250 else 0.3 + 0.1 * (i % 3),  # médiane finale 0,4
                "sil/buffer_size": float(cap if t >= full_after_s else int(cap * t / full_after_s)),
                "sil/steps_done": 6.0,
                "sil/grad_share": 0.1,
                "sil/frac_pos": 0.5,
                "sil/ram_available_mb": 3000.0,
            }
        )
    return rows


def test_fenetre_des_10_dernieres_minutes_et_buffer_plein() -> None:
    # Arrange : 15 min, disp_share élevé et buffer non plein avant 250 s, hors de la fenêtre finale (t >= 299 s)
    rows = _sil_rows()
    # Act
    win = sbs.sil_window(rows, capacity=1000)
    half_full = sbs.sil_window(_sil_rows(full_after_s=700.0), capacity=1000)
    short = sbs.sil_window(_sil_rows(n_min=8.0), capacity=1000)
    # Assert
    assert win["conclusive"] and win["coef"] == 0.1
    assert win["disp_share"]["median"] == pytest.approx(0.4)  # le début du banc est hors fenêtre
    assert win["rows"] == 601 and win["window_covered_s"] == pytest.approx(600.0)
    assert win["buffer_full_after_s"] == pytest.approx(250.0)
    assert not half_full["conclusive"] and "buffer non plein" in half_full["not_conclusive_because"][0]
    assert not short["conclusive"]
    assert not sbs.sil_window(rows, capacity=None)["conclusive"]


def test_ram_autour_des_checkpoints() -> None:
    # Arrange : RAM disponible 3 000 Mo, creux à 1 200 Mo pendant l'écriture (t = 100 à 103 s), échantillons à 1 s
    mb = sbs.MB
    t0 = 1_800_000_000.0
    samples = [
        {"unix": t0 + t, "avail": (1200 if 100 <= t <= 103 else 3000) * mb, "private": 6000 * mb} for t in range(200)
    ]
    events = [{"event": "checkpoint", "t": t0 + 103.5, "capture_s": 1.0, "write_s": 2.5, "iteration": 250, "bytes": 8}]
    coarse = samples[::10]
    # Act
    ram = sbs.ram_summary(samples, [{"sil/ram_available_mb": 2500.0}], events)
    ram_coarse = sbs.ram_summary(coarse, [], events)
    # Assert
    ck = ram["checkpoints"]["rows"][0]
    assert ck["avail_min_mb"] == 1200 and ck["avail_baseline_mb"] == 3000 and ck["drop_mb"] == 1800
    assert ram["avail_min_mb"] == 1200 and ram["below_1_5gb"] and not ram["ks1_triggered"]
    assert ram["private_max_mb"] == 6000 and ram["avail_min_mb_metrics"] == 2500
    assert ram["checkpoints"]["sample_period_median_s"] == 1.0 and not ram["warnings"]
    assert ram_coarse["checkpoints"]["undersampled"] == [250] and "--sample-s 1" in ram_coarse["warnings"][0]
    assert sbs.longest_below([(0, 900), (300, 900), (700, 900), (701, 2000)], 1000) == 700  # KS1 : > 600 s
    assert sbs.ram_summary([{"unix": t0 + 700 * i, "avail": 900 * mb} for i in range(2)], [], [])["ks1_triggered"]


def _smoke_rows(cap: int = 250_000, break_at: int | None = None) -> list[dict[str, Any]]:
    rows, size = [], 0.0
    for s in range(1, 19):
        released = s >= 17
        new = 0.0 if released else 300.0
        size = 0.0 if released else min(cap, size + new)
        if break_at == s:
            size = new  # buffer perdu à la reprise
        rows.append(
            {
                "step": s,
                "sil/coef": 0.0 if released else 0.1,
                "sil/released": float(released),
                "sil/buffer_size": size,
                "sil/buffer_new": new,
                "sil/steps_done": 0.0 if released or s < 3 else 6.0,
                "sil/loss_policy": 0.0 if released or s < 3 else 0.02,
                "sil/disp_share": 0.0 if released or s < 3 else 0.2,
                "sil/ram_available_mb": 3000.0,
            }
        )
    return rows


def test_verification_du_smoke_sil() -> None:
    # Arrange / Act : 18 itérations, reprise à 9, libération à 17 ; puis buffer perdu à la reprise ; puis smoke FAIL
    ok = sbs.smoke_checks(_smoke_rows(), capacity=250_000, resume_starts=[9], smoke_verdict="PASS")
    lost = sbs.smoke_checks(_smoke_rows(break_at=10), capacity=250_000, resume_starts=[9], smoke_verdict="PASS")
    failed = sbs.smoke_checks(_smoke_rows(), capacity=250_000, resume_starts=[9], smoke_verdict="FAIL")
    nan_rows = _smoke_rows()
    nan_rows[4]["sil/disp_share"] = "nan"
    # Assert
    assert ok["verdict"] == "PASS" and all(ok["checks"].values())
    assert lost["verdict"] == "FAIL" and lost["buffer_breaks"] == [10]
    assert not lost["checks"]["buffer_conserved_each_iteration"]
    assert failed["verdict"] == "FAIL" and not failed["checks"]["smoke_pass"]
    bad = sbs.smoke_checks(nan_rows, capacity=250_000, resume_starts=[9], smoke_verdict="PASS")
    assert bad["nonfinite_keys"] == ["sil/disp_share"] and bad["verdict"] == "FAIL"
    assert (
        sbs.smoke_checks(_smoke_rows(), capacity=250_000, resume_starts=[], smoke_verdict="PASS")["verdict"] == "FAIL"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_resume_complet_d_un_dossier_de_banc_synthetique(tmp_path: Path) -> None:
    # Arrange : 15 min de banc à 1 itération/s, buffer plein après 250 s, médiane finale 0,4 à c_sil = 0,1
    d = tmp_path / "bench_int_sil_x"
    d.mkdir()
    rows = _sil_rows()
    t0 = rows[0]["t"]
    meta = {"name": d.name, "learned_per_iter": 51_611, "collected_per_iter": 57_344, "provenance": {"git_commit": "c"}}
    (d / "bench_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "bench_end.json").write_text(json.dumps({"run": {"start_iteration": 0}}), encoding="utf-8")
    (d / "run_meta.json").write_text(json.dumps({"config": {"sil": {"enabled": True, "capacity": 1000}}}), "utf-8")
    ticks = [
        {"n": i, "t": float(i), "gap_s": 1.0 if i else 0.0, "t_bench": 10.0 + i, "unix": t0 + i} for i in range(901)
    ]
    _write_jsonl(d / "bench_ticks.jsonl", ticks)
    _write_jsonl(d / "metrics.jsonl", [{**r, "perf/agent_steps_per_s": 57_344.0} for r in rows])
    _write_jsonl(
        d / "events.jsonl", [{"event": "checkpoint", "t": t0 + 250, "capture_s": 1, "write_s": 2, "iteration": 250}]
    )
    _write_jsonl(d / "bench_samples.jsonl", [{"unix": t0 + t, "avail": 3000 * sbs.MB} for t in range(900)])
    # Act
    code = sbs.main([str(d)])
    # Assert
    assert code == 0
    s = json.loads((d / "sil_summary.json").read_text(encoding="utf-8"))
    assert s["calibration"]["decision"] == "RETENIR" and s["sil"]["disp_share"]["median"] == pytest.approx(0.4)
    assert s["throughput"]["global"]["learned_wall"] == pytest.approx(51_611.0)
    assert not s["throughput"]["ks4"]["conclusive"]  # 15 min < 30 min
    assert s["ram"]["avail_min_mb"] == 3000 and s["ram"]["checkpoints"]["count"] == 1
    assert sbs.main([str(d)]) == 0 and (d / "sil_summary_001.json").exists()  # jamais écrasé
    assert sbs.main([str(tmp_path / "absent")]) == 2
