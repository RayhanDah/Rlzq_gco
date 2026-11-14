"""Partie pure du banc soutenu (`scripts/bench_run.py`, A5) : agrégation par tranche, K6, extrapolations, résumé.
Sans simulateur ni run."""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("bench_run", ROOT / "scripts" / "bench_run.py")
assert _spec is not None and _spec.loader is not None
br = importlib.util.module_from_spec(_spec)
sys.modules["bench_run"] = br
_spec.loader.exec_module(br)

LEARNED, COLLECTED = 51_611, 57_344


def _rec(it: int, t_end: float, gap: float, rate: float | None, pool: float | None = None) -> Any:
    return br.IterRecord(it, t_end, gap, rate, LEARNED, COLLECTED, pool)


def test_percentile_matches_numpy_linear() -> None:
    xs = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0]
    for q in (0, 10, 50, 90, 100, 37.5):
        assert br.percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))
    assert br.percentile([], 50) is None
    assert br.percentile([4.0], 90) == 4.0


def test_dist_ignores_non_finite() -> None:
    d = br.dist([1.0, float("nan"), 3.0, float("inf")])
    assert d["n"] == 2 and d["median"] == 2.0 and d["min"] == 1.0 and d["max"] == 3.0


def test_build_iter_records_joins_ticks_and_metrics_and_drops_first_iteration_rate() -> None:
    ticks = [{"t": 0.0, "gap_s": 0.0}, {"t": 2.0, "gap_s": 2.0}, {"t": 3.5, "gap_s": 1.5}]
    metrics = {
        1: {"perf/agent_steps_per_s": 90_000.0, "pool/size": 1.0},
        2: {"perf/agent_steps_per_s": 50_000.0, "pool/size": 1.0},
    }
    recs = br.build_iter_records(
        ticks, metrics, start_iteration=0, learned_per_iter=LEARNED, collected_per_iter=COLLECTED
    )
    assert [r.iteration for r in recs] == [1, 2]
    assert recs[0].loop_rate_collected is None and recs[0].collected == 2 * COLLECTED  # deux lots
    assert recs[1].loop_rate_collected == 50_000.0 and recs[1].collected == COLLECTED
    assert recs[1].gap_s == 1.5 and recs[1].t_end == 3.5 and recs[1].pool_size == 1.0
    # reprise : la 1re itération du segment n'est pas la 1re du run
    resumed = br.build_iter_records(
        ticks, metrics, start_iteration=1, learned_per_iter=LEARNED, collected_per_iter=COLLECTED
    )
    assert resumed[0].iteration == 2 and resumed[0].loop_rate_collected == 50_000.0


def test_slice_stats_groups_by_wall_time_and_computes_rates() -> None:
    recs = [_rec(i + 1, t_end=float(i + 1), gap=1.0, rate=40_000.0 + 1_000 * i, pool=float(i // 5)) for i in range(15)]
    sl = br.slice_stats(recs, slice_s=10.0, threshold=50_000.0)
    assert [s["iterations"] for s in sl] == [9, 6]  # t_end 1..9 puis 10..15
    first = sl[0]
    loop = [40_000.0 + 1_000 * i for i in range(9)]
    assert first["collected_loop"]["median"] == pytest.approx(float(np.median(loop)))
    assert first["collected_loop"]["p10"] == pytest.approx(float(np.percentile(loop, 10)))
    assert first["learned_loop"]["p90"] == pytest.approx(float(np.percentile(loop, 90)) * LEARNED / COLLECTED)
    assert first["learned_wall"] == pytest.approx(LEARNED)  # 9 itérations en 9 s
    assert first["below_threshold"] is False  # 51,6k >= 50k
    assert first["partial"] is False and sl[1]["partial"] is True  # 6 s sur 10
    assert first["pool_size_max"] == 1.0
    with pytest.raises(ValueError):
        br.slice_stats(recs, slice_s=0)


def test_global_rates_wall_and_loop() -> None:
    recs = [_rec(1, 2.0, 2.0, None), _rec(2, 4.0, 2.0, COLLECTED / 1.0), _rec(3, 6.0, 2.0, COLLECTED / 1.0)]
    g = br.global_rates(recs)
    assert g["learned_wall"] == pytest.approx(3 * LEARNED / 6.0)
    assert g["learned_loop"] == pytest.approx(LEARNED / 1.0)  # 1 s d'iterate par itération connue
    assert g["collected_loop"] == pytest.approx(COLLECTED)


def _slices(durations_below: list[tuple[float, bool]]) -> list[dict[str, Any]]:
    return [{"slice": i, "duration_s": d, "below_threshold": b} for i, (d, b) in enumerate(durations_below)]


def test_k6_triggers_only_on_contiguous_streak_longer_than_window() -> None:
    sl = _slices([(300, True)] * 7)  # 35 min d'affilée
    k = br.k6_check(sl, 20_000.0)
    assert k["triggered"] and k["longest_below_s"] == 2100 and k["global_below"] and k["conclusive"]
    broken = _slices([(300, True)] * 5 + [(300, False)] + [(300, True)] * 5)
    k2 = br.k6_check(broken, 30_000.0)
    assert (
        not k2["triggered"] and k2["longest_below_s"] == 1500 and k2["slices_below"] == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    )
    exactly = br.k6_check(_slices([(300, True)] * 6), 20_000.0)
    assert not exactly["triggered"]  # « pendant > 30 min » : 30 min pile ne déclenche pas
    short = br.k6_check(_slices([(300, False)] * 2), 60_000.0)
    assert not short["conclusive"]


def test_throttle_ratio_uses_full_slices() -> None:
    def s(med: float, partial: bool = False) -> dict[str, Any]:
        return {"partial": partial, "learned_loop": {"median": med}}

    assert br.throttle_ratio([s(80.0), s(60.0), s(40.0), s(10.0, partial=True)])["ratio"] == pytest.approx(0.5)
    assert br.throttle_ratio([s(80.0)])["ratio"] is None


def test_extrapolate_and_fit() -> None:
    assert br.extrapolate_linear(10.0, 2.0, 1000.0) == pytest.approx(5000.0)
    assert br.extrapolate_linear(10.0, 0.0, 1000.0) is None
    a, b = br.fit_slope([1.0, 2.0, 3.0], [12.0, 14.0, 16.0]) or (0, 0)
    assert a == pytest.approx(2.0) and b == pytest.approx(10.0)
    assert br.fit_slope([2.0, 2.0], [1.0, 3.0]) is None


def test_checkpoint_projection_to_100_snapshots_and_cadences() -> None:
    ck = [
        {"iteration": 0, "bytes": 100, "capture_s": 0.1, "write_s": 0.2, "nets": 1},  # sans lot en attente
        {"iteration": 50, "bytes": 1_000, "capture_s": 0.5, "write_s": 1.5, "nets": 2},
        {"iteration": 100, "bytes": 1_010, "capture_s": 0.5, "write_s": 1.5, "nets": 3},
    ]
    p = br.checkpoint_projection(ck, unit_bytes=10, target_nets=100, iteration_s=2.0, cadences=(250,))
    assert p["available"] and p["label"] == "extrapolé"
    assert p["bytes"] == pytest.approx(1_010 + 97 * 10)
    scale = (1_010 + 970) / 1_010
    assert p["write_s"] == pytest.approx(1.5 * scale) and p["total_s"] == pytest.approx(2.0 * scale)
    assert p["slope_bytes_per_net_fit"] == pytest.approx(10.0)  # itération 0 exclue de l'ajustement
    (c,) = p["cadences"]
    assert c["minutes_between"] == pytest.approx(250 * 2.0 / 60) and c["cost_fraction"] == pytest.approx(
        2.0 * scale / 500.0
    )
    # sans unité mesurée : pente ajustée
    p2 = br.checkpoint_projection(ck, unit_bytes=None, target_nets=4)
    assert p2["bytes"] == pytest.approx(1_020)
    assert br.checkpoint_projection([], unit_bytes=10) == {"available": False}


def test_nets_at() -> None:
    adds = [{"iteration": 0, "nets": 1}, {"iteration": 50, "nets": 2}, {"iteration": 100, "nets": 3}]
    assert br.nets_at(adds, 0) == 1 and br.nets_at(adds, 99) == 2 and br.nets_at(adds, 100) == 3
    assert br.nets_at(adds[1:], 10) is None


def _eval(it: int, total: float, minutes: float = 40.0) -> dict[str, Any]:
    return {
        "event": "eval",
        "k": it // 10,
        "iteration": it,
        "passed": False,
        "timing": {"total_s": total, "snapshot_s": 0.1, "random_s": total / 2, "frozen_s": total / 2},
        "random": {"episodes": 512, "minutes": minutes / 2, "decisive": 3},
        "frozen": {"episodes": 512, "minutes": minutes / 2},
    }


def test_eval_analysis_flags_slow_eval_and_tells_slow_machine_from_longer_episodes() -> None:
    evals = [_eval(10, 16.0), _eval(20, 17.0), _eval(30, 18.0), _eval(40, 179.0), _eval(50, 60.0, 400.0)]
    ticks = {
        10: {"eval_arena_steps": 1_000_000, "eval_sim_s": 5.0, "eval_env_step_s": 8.0},
        20: {"eval_arena_steps": 1_000_000, "eval_sim_s": 5.2, "eval_env_step_s": 8.5},
        30: {"eval_arena_steps": 1_000_000, "eval_sim_s": 5.4, "eval_env_step_s": 9.0},
        40: {"eval_arena_steps": 1_000_000, "eval_sim_s": 100.0, "eval_env_step_s": 150.0},  # machine ralentie
        50: {"eval_arena_steps": 10_000_000, "eval_sim_s": 50.0, "eval_env_step_s": 52.0},  # épisodes plus longs
    }
    res = br.eval_analysis(evals, ticks)
    rows = {r["iteration"]: r for r in res["rows"]}
    assert res["count"] == 5 and res["slow"] == [40, 50]
    assert rows[10]["sim_arena_steps_per_s"] == pytest.approx(200_000.0)
    assert rows[10]["non_sim_s"] == pytest.approx(8.0) and rows[10]["episodes"] == 1024
    assert rows[10]["game_minutes"] == pytest.approx(40.0)
    assert "machine ralentie" in rows[40]["diagnosis"] and "épisodes plus longs" not in rows[40]["diagnosis"]
    assert "épisodes plus longs" in rows[50]["diagnosis"] and "machine ralentie" not in rows[50]["diagnosis"]
    assert rows[20]["slow"] is False and rows[20]["diagnosis"] is None
    # événements autres qu'« eval » ignorés ; une seule évaluation n'est jamais « lente »
    one = br.eval_analysis([{"event": "rollback", "iteration": 1}, _eval(10, 500.0)], {})
    assert one["count"] == 1 and one["slow"] == [] and one["rows"][0]["arena_steps"] is None


def test_ram_summary_slope_curve_and_battery() -> None:
    samples = [
        {"t": float(t), "rss": 1e9 + 1e5 * t, "avail": 8e9 - 1e5 * t, "private": 2e9, "ac_line": 1}
        for t in range(0, 1200, 10)
    ]
    samples[-1]["ac_line"] = 0
    r = br.ram_summary(samples, bin_s=60.0, horizon_h=8.0, warmup_s=300.0)
    assert r["rss_max"] == pytest.approx(1e9 + 1e5 * 1190)
    assert r["rss_slope_bytes_per_h"] == pytest.approx(1e5 * 3600)
    assert r["rss_at_horizon_extrapolated"] == pytest.approx(1e9 + 1e5 * 8 * 3600)
    assert r["avail_min"] == pytest.approx(8e9 - 1e5 * 1190) and r["private_max"] == 2e9
    assert len(r["curve"]) == 20 and r["curve"][1]["rss_max"] == pytest.approx(1e9 + 1e5 * 110)
    assert r["on_battery_seen"] is True and r["ac_line_values"] == [0, 1]
    assert br.ram_summary([])["n"] == 0


def test_log_projection_follows_timesteps_sessions_and_evaluation_count() -> None:
    sizes = {
        "metrics.jsonl": 1_000,
        "evals.jsonl": 300,
        "evalhook_state.jsonl": 100,
        "run_meta.json": 50,
        "segments.jsonl": 20,
    }
    p = br.log_projection(sizes, trained=100_000_000, evals_done=2, evals_target=20, target=1_000_000_000)
    f = p["files"]
    assert f["metrics.jsonl"]["extrapolated"] == pytest.approx(10_000)  # linéaire en timesteps (× 10)
    assert f["evals.jsonl"]["extrapolated"] == pytest.approx(3_000)  # par évaluation (× 20 / 2)
    # θ₀ + 2 évaluations : 1 + 2 + 3 = 6 unités ; θ₀ + 20 : 21·22/2 = 231 unités
    assert f["evalhook_state.jsonl"]["extrapolated"] == pytest.approx(100 * 231 / 6)
    assert "très imprécis" in f["evals.jsonl"]["label"] and "très imprécis" in f["evalhook_state.jsonl"]["label"]
    assert "label" not in f["metrics.jsonl"]
    assert f["run_meta.json"]["extrapolated"] == 50 and f["segments.jsonl"]["extrapolated"] == 20
    assert p["total_bytes"] == 1_470 and p["label"] == "extrapolé"
    assert p["total_extrapolated"] == pytest.approx(10_000 + 3_000 + 100 * 231 / 6 + 70)
    precise = br.log_projection(sizes, trained=100_000_000, evals_done=3, evals_target=20)
    assert "label" not in precise["files"]["evals.jsonl"]
    none = br.log_projection(sizes, trained=0)
    assert (
        none["files"]["metrics.jsonl"]["extrapolated"] is None and none["files"]["evals.jsonl"]["extrapolated"] is None
    )
    assert none["total_extrapolated"] is None


def test_k6_on_collected_rate() -> None:
    sl = [{"slice": i, "duration_s": 300.0, "below_threshold": False, "collected_wall": 20_000.0} for i in range(7)]
    k = br.k6_check(sl, 20_000.0, rate_key="collected_wall")
    assert k["triggered"] and k["rate"] == "collected_wall" and k["global_below"]
    assert not br.k6_check(sl, 30_000.0)["triggered"]  # le drapeau appris, lui, est faux


def test_ram_summary_xpu_and_pessimistic_label() -> None:
    samples = [{"t": float(t), "rss": 1e9, "xpu_reserved": 1e8 * (1 + t // 60)} for t in range(0, 180, 30)]
    r = br.ram_summary(samples, bin_s=60.0)
    assert r["xpu_reserved_max"] == pytest.approx(3e8) and [c["xpu_reserved_max"] for c in r["curve"]] == [
        1e8,
        2e8,
        3e8,
    ]
    assert "pessimiste" in r["rss_at_horizon_label"]


def test_is_under_protected_roots(tmp_path: Path) -> None:
    roots = [tmp_path / "runs", tmp_path / "checkpoints"]
    assert br.is_under(tmp_path / "runs" / "x" / "probe", roots)
    assert br.is_under(tmp_path / "checkpoints", roots)
    assert not br.is_under(tmp_path / "runs_other", roots) and not br.is_under(tmp_path / "scratch", roots)


def test_write_unique_never_overwrites(tmp_path: Path) -> None:
    a = br.write_unique(tmp_path / "s.json", "1")
    b = br.write_unique(tmp_path / "s.json", "2")
    assert a.name == "s.json" and b.name == "s_001.json"
    assert a.read_text() == "1" and b.read_text() == "2"


def test_unique_name_never_reuses() -> None:
    now = datetime.datetime(2026, 11, 14, 9, 8, 7, tzinfo=datetime.UTC)
    taken = {"bench_4a_20261114T090807Z", "bench_4a_20261114T090807Z_001"}
    assert br.unique_name("bench_4a", now, lambda n: n in taken) == "bench_4a_20261114T090807Z_002"
    assert br.unique_name("x", now, lambda n: False) == "x_20261114T090807Z"


def _synthetic_inputs(n_iter: int = 40, gap: float = 1.5) -> dict[str, Any]:
    t0 = 1_000_000.0  # unix de l'origine
    ticks: list[dict[str, Any]] = [{"n": 0, "t": 0.0, "t_bench": 10.0, "unix": t0, "gap_s": 0.0}]
    for i in range(1, n_iter + 1):
        g = gap + (20.0 if i == 20 else 0.0)  # évaluation à l'itération 20
        t = ticks[-1]["t"] + g
        rec = {
            "n": i,
            "t": t,
            "t_bench": 10.0 + t,
            "unix": t0 + t,
            "gap_s": g,
            "collect_s": 1.0,
            "learn_s": 0.8,
            "train_env_step_s": 0.6,
            "train_sim_step_s": 0.4,
            "eval_arena_steps": 2_000_000 if i == 20 else 0,
            "eval_sim_s": 10.0 if i == 20 else 0.0,
            "eval_env_step_s": 15.0 if i == 20 else 0.0,
        }
        ticks.append(rec)
    metrics = [{"step": i, "perf/agent_steps_per_s": COLLECTED / 1.2, "pool/size": 1.0 + i // 20} for i in range(1, 41)]
    events = [
        {"event": "checkpoint", "t": t0 - 1, "iteration": 0, "bytes": 100_000, "capture_s": 0.1, "write_s": 0.2},
        {"event": "checkpoint", "t": t0 + 30, "iteration": 20, "bytes": 1_000_000, "capture_s": 0.3, "write_s": 0.7},
        {
            "event": "checkpoint",
            "t": t0 + ticks[-1]["t"] + 0.5,
            "iteration": 40,
            "bytes": 1_010_000,
            "capture_s": 0.3,
            "write_s": 0.7,
        },
    ]
    return {
        "name": "bench_test",
        "provenance": {"git_commit": "abc", "git_dirty": False, "seed": 0, "command": ["python", "x"]},
        "run_summary": {"status": "interrupted", "start_iteration": 0, "end_iteration": 40, "loop_s": 48.0}
        | {"trained_timesteps": 40 * LEARNED, "restores": 0},
        "ticks": ticks,
        "metrics_rows": metrics,
        "events": events,
        "evals": [_eval(20, 20.0)],
        "pool_adds": [
            {"iteration": 0, "s": 0.01, "size": 1, "nets": 1},
            {"iteration": 20, "s": 0.02, "size": 2, "nets": 2},
        ],
        "samples": [{"t": float(t), "rss": 5e9, "avail": 3e9, "ac_line": 1} for t in range(0, 80, 10)],
        "log_sizes": {"metrics.jsonl": 40_000, "events.jsonl": 1_000},
        "eval_snapshots": {"count": 2, "bytes": 1_400_000},
        "learned_per_iter": LEARNED,
        "collected_per_iter": COLLECTED,
        "unit_bytes": 700_000,
        "eval_every": 50_000_000,
        "t_run_start": 2.0,
        "t_run_end": 10.0 + ticks[-1]["t"] + 1.5,
        "slice_s": 30.0,
    }


def test_summarize_bench_end_to_end_and_table() -> None:
    inp = _synthetic_inputs()
    s = br.summarize_bench(**inp)
    json.dumps(s, default=str)  # sérialisable
    b = s["breakdown"]
    loop_wall = inp["ticks"][-1]["t"]
    assert b["loop_wall_s"] == pytest.approx(loop_wall)
    assert b["setup_s"] == pytest.approx(8.0) and b["teardown_s"] == pytest.approx(1.5)
    assert b["evals_s"] == pytest.approx(20.0)
    assert b["checkpoints_s"] == pytest.approx(1.0)  # seul le checkpoint 20 est dans la boucle
    assert b["pool_snapshots_s"] == pytest.approx(0.02)  # θ₀ (itération 0) appartient à l'assemblage
    assert b["other_s"] == pytest.approx(loop_wall - 48.0 - 20.0 - 1.0 - 0.02)
    assert b["collect_sim_step_s"] == pytest.approx(16.0) and b["collect_obs_reward_s"] == pytest.approx(8.0)
    assert b["collect_other_s"] == pytest.approx(16.0) and b["learn_s"] == pytest.approx(32.0)
    phases = [r["phase"] for r in s["checkpoints"]["rows"]]
    assert phases == ["setup", "loop", "final"] and s["checkpoints"]["rows"][-1]["final"]
    assert [r["nets"] for r in s["checkpoints"]["rows"]] == [1, 2, 2]
    proj = s["checkpoints"]["projection"]
    assert proj["bytes"] == pytest.approx(1_010_000 + 98 * 700_000)
    (ev,) = s["evals"]["rows"]
    assert ev["arena_steps"] == 2_000_000 and ev["sim_arena_steps_per_s"] == pytest.approx(200_000.0)
    assert s["global"]["iterations"] == 40
    assert s["global"]["learned_wall"] == pytest.approx(40 * LEARNED / loop_wall)
    assert s["extrapolation_1b"]["hours"] == pytest.approx(1e9 / s["global"]["learned_wall"] / 3600)
    assert s["eval_snapshots"]["extrapolated_bytes"] == pytest.approx(700_000 * 21)
    assert s["k6"]["conclusive"] is False and any("K6 non concluant" in w for w in s["warnings"])
    assert s["pool"]["final_size"] == 2 and s["ram"]["rss_max"] == 5e9
    table = br.render_table(s)
    for needle in ("Débit par tranche", "K6", "Évaluations", "Checkpoints", "RAM", "Journaux", "extrapolé"):
        assert needle in table


def test_summarize_bench_flags_dirty_tree_restores_and_battery() -> None:
    inp = _synthetic_inputs(n_iter=10)
    inp["provenance"] = {**inp["provenance"], "git_dirty": True}
    inp["events"] = [*inp["events"], {"event": "restore", "t": 1_000_005.0}]
    inp["samples"] = [{"t": 0.0, "rss": 1e9, "avail": 1e9, "ac_line": 0}]
    s = br.summarize_bench(**inp)
    text = " ".join(s["warnings"])
    assert "arbre git sale" in text and "restauration" in text and "batterie" in text


# ---------------------------------------------------------------------------------------------- revue A5 (M1)


def test_summarize_bench_without_checkpoint() -> None:
    inp = _synthetic_inputs()
    inp["events"] = []
    s = br.summarize_bench(**inp)
    assert s["checkpoints"]["rows"] == [] and s["checkpoints"]["projection"] == {"available": False}
    assert s["breakdown"]["checkpoints_s"] == 0.0
    br.render_table(s)


def test_summarize_bench_without_evaluation_and_without_ticks() -> None:
    inp = _synthetic_inputs()
    inp["evals"] = []
    s = br.summarize_bench(**inp)
    assert s["evals"]["count"] == 0 and s["evals"]["median_s"] is None
    br.render_table(s)
    inp["ticks"], inp["metrics_rows"], inp["samples"], inp["pool_adds"] = [], [], [], []
    empty = br.summarize_bench(**inp)
    assert empty["global"]["iterations"] == 0 and empty["breakdown"]["loop_wall_s"] is None
    br.render_table(empty)


def test_summarize_bench_with_incomplete_events() -> None:
    inp = _synthetic_inputs()
    inp["events"] = [
        *inp["events"],
        {"event": "checkpoint", "t": 1_000_030.0},  # ni itération ni durées : écriture interrompue
        {"event": "checkpoint", "t": 1_000_031.0, "iteration": 25, "bytes": None, "capture_s": "nan"},
    ]
    inp["evals"] = [*inp["evals"], {"event": "eval", "timing": {}}, {"event": "eval", "iteration": "x"}]
    inp["pool_adds"] = [*inp["pool_adds"], {"iteration": 30}]
    s = br.summarize_bench(**inp)
    assert [r["iteration"] for r in s["checkpoints"]["rows"]] == [0, 20, 25, 40]
    assert s["evals"]["count"] == 1
    (warn,) = [w for w in s["warnings"] if "incomplètes" in w]
    assert "checkpoints 1" in warn and "évaluations 2" in warn and "instantanés du pool 1" in warn
    assert "champs de checkpoint 1" in warn
    br.render_table(s)


def _write_bench_dir(d: Path, inp: dict[str, Any], *, with_end: bool) -> None:
    d.mkdir()

    def jl(name: str, rows: list[dict[str, Any]]) -> None:
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"tronquée', encoding="utf-8")

    meta = {
        "name": inp["name"],
        "provenance": inp["provenance"],
        "checkpoint_dir": str(d / "ckpt"),
        "learned_per_iter": LEARNED,
        "collected_per_iter": COLLECTED,
        "eval_every": inp["eval_every"],
        "slice_s": inp["slice_s"],
        "unit_bytes": inp["unit_bytes"],
        "t_run_start": inp["t_run_start"],
    }
    (d / br.BENCH_META).write_text(json.dumps(meta), encoding="utf-8")
    if with_end:
        end = {"run": inp["run_summary"], "t_run_end": inp["t_run_end"], "error": None, "provenance_end": {}}
        (d / br.BENCH_END).write_text(json.dumps(end), encoding="utf-8")
    jl("bench_ticks.jsonl", inp["ticks"])
    jl(br.BENCH_POOL, inp["pool_adds"])
    jl("bench_samples.jsonl", inp["samples"])
    jl("metrics.jsonl", inp["metrics_rows"])
    jl("events.jsonl", inp["events"])
    jl("evals.jsonl", inp["evals"])


def test_finalize_offline_matches_in_memory_summary_and_never_overwrites(tmp_path: Path) -> None:
    inp = _synthetic_inputs()
    d = tmp_path / "bench"
    _write_bench_dir(d, inp, with_end=True)
    code, path = br.finalize(d)
    assert code == 0 and path.name == "bench_summary.json" and (d / "bench_summary.txt").exists()
    s = json.loads(path.read_text(encoding="utf-8"))
    ref = br.summarize_bench(**inp)
    assert s["global"]["learned_wall"] == pytest.approx(ref["global"]["learned_wall"])
    assert s["breakdown"]["other_s"] == pytest.approx(ref["breakdown"]["other_s"])
    assert s["pool"]["adds"] == 2 and s["evals"]["count"] == 1
    code2, path2 = br.finalize(d)  # --summarize relancé : nouveau fichier, l'ancien intact
    assert code2 == 0 and path2.name == "bench_summary_001.json" and path.exists()


def test_finalize_without_end_file_gives_partial_summary(tmp_path: Path) -> None:
    d = tmp_path / "bench"
    _write_bench_dir(d, _synthetic_inputs(), with_end=False)
    (d / "segments.jsonl").write_text(json.dumps({"status": "error", "start_iteration": 0}) + "\n", encoding="utf-8")
    code, path = br.finalize(d)
    s = json.loads(path.read_text(encoding="utf-8"))
    assert code == 0 and any("absent" in w for w in s["warnings"]) and s["run"]["status"] == "error"


def test_finalize_failure_writes_error_file_and_returns_1(tmp_path: Path) -> None:
    d = tmp_path / "pas_un_banc"
    d.mkdir()
    (d / "metrics.jsonl").write_text("", encoding="utf-8")
    code, path = br.finalize(d)
    err = json.loads(path.read_text(encoding="utf-8"))
    assert code == 1 and path.name == "bench_summary_error.json"
    assert "bench_meta.json" in err["error"] and "--summarize" in err["retry"] and "metrics.jsonl" in err["raw_files"]
    bad = tmp_path / "meta_cassee"
    _write_bench_dir(bad, _synthetic_inputs(), with_end=True)
    meta = json.loads((bad / br.BENCH_META).read_text(encoding="utf-8"))
    del meta["learned_per_iter"]
    (bad / br.BENCH_META).write_text(json.dumps(meta), encoding="utf-8")
    code_bad, p_bad = br.finalize(bad)
    assert code_bad == 1 and p_bad.name == "bench_summary_error.json"
    assert (bad / "bench_ticks.jsonl").exists()  # fichiers bruts intacts


class _FakePool:
    def __init__(self, max_size: int) -> None:
        self.max_size, self.n = max_size, 0

    def __len__(self) -> int:
        return self.n


class _FakeOpponents:
    def __init__(self, max_size: int) -> None:
        self.pool, self.nets, self.calls = _FakePool(max_size), {}, 0

    def add_snapshot(self, policy: Any, iteration: int) -> int:
        self.calls += 1
        self.pool.n = min(self.pool.n + 1, self.pool.max_size)
        self.nets[self.calls] = policy
        return self.calls


def test_prefill_complete_le_pool_a_l_iteration_0_seulement(tmp_path: Path) -> None:
    # Arrange
    inst = br.Instrument(64, 16, tmp_path / "pool.jsonl", prefill=5)
    wrapped = inst.pool_add(_FakeOpponents.add_snapshot)
    opp = _FakeOpponents(max_size=100)
    # Act
    first = wrapped(opp, "theta0", 0)
    wrapped(opp, "theta50", 50)
    # Assert
    assert first == 1  # uid du premier ajout, celui de θ₀
    assert [a["size"] for a in inst.pool_adds] == [5, 6]
    assert [a["prefilled"] for a in inst.pool_adds] == [4, 0]
    rows = br.read_jsonl(tmp_path / "pool.jsonl")
    assert [r["prefilled"] for r in rows] == [4, 0]


def test_prefill_borne_par_la_taille_max_et_inactif_par_defaut() -> None:
    capped = br.Instrument(64, 16, prefill=500)
    opp = _FakeOpponents(max_size=3)
    capped.pool_add(_FakeOpponents.add_snapshot)(opp, "theta0", 0)
    assert opp.calls == 3 and capped.pool_adds[0]["size"] == 3

    off = br.Instrument(64, 16)
    opp2 = _FakeOpponents(max_size=3)
    off.pool_add(_FakeOpponents.add_snapshot)(opp2, "theta0", 0)
    assert opp2.calls == 1 and off.pool_adds[0]["prefilled"] == 0

    with pytest.raises(ValueError):
        br.Instrument(64, 16, prefill=-1)


def test_summarize_bench_signale_le_pre_remplissage_et_tolere_son_absence() -> None:
    inp = _synthetic_inputs()
    old = br.summarize_bench(**inp)  # ancien banc : pas de champ `prefilled`
    assert old["pool"]["prefilled"] == 0 and not any("pré-rempli" in w for w in old["warnings"])
    assert "PRÉ-REMPLI" not in br.render_table(old)

    inp["pool_adds"] = [{**inp["pool_adds"][0], "prefilled": 99}, *inp["pool_adds"][1:]]
    new = br.summarize_bench(**inp)
    assert new["pool"]["prefilled"] == 99 and any("pré-rempli (99" in w for w in new["warnings"])
    assert "PRÉ-REMPLI : 99" in br.render_table(new)
