"""Banc soutenu de la VRAIE boucle d'entraînement (tâche A5) : `training.run` par les mêmes chemins que
`scripts/train.py --eval` (pool d'adversaires et évaluation en cours de run actifs), pendant une durée murale donnée,
puis arrêt propre par le crochet `should_stop` de `run` (l'itération en cours finit, checkpoint final ; pas de kill).

    vendor/venv-xpu/Scripts/python scripts/bench_run.py                      # 40 min, configs/bench_4a.yaml, XPU
    vendor/venv-xpu/Scripts/python scripts/bench_run.py --minutes 45 --checkpoint-probe
    vendor/venv-xpu/Scripts/python scripts/bench_run.py --probe-only <checkpoint.pt>   # sonde seule, sans run

Sorties (jamais écrasées ni supprimées) : `runs/<nom>/` (journaux du run, `bench_ticks.jsonl` : une ligne par
itération, `bench_samples.jsonl` : RAM et alimentation, `bench_summary.json`, `bench_summary.txt`) et
`checkpoints/<nom>/`, avec `<nom>` = `bench_4a_<horodatage UTC>` unique. Le banc refuse un arbre git sale (sauf
`--allow-dirty`, alors signalé dans le résumé).

Instrumentation (dans ce script seulement, `src/` n'est pas modifié) : des enveloppes de chronométrage sont posées sur
`Learner.update`, `Collector.collect`, `BatchEnv.step`, `RocketGPUSimulator.step` et `PastOpponents.add_snapshot` le
temps du run (mêmes objets, mêmes appels, un `perf_counter` de plus par appel). Le simulateur d'entraînement et celui
de l'évaluation se distinguent par leur nombre d'arènes. Le reste vient des journaux de `run` (`metrics.jsonl`,
`events.jsonl`, `evals.jsonl`).

Définitions :
- débit « boucle » : `perf/agent_steps_per_s` de `metrics.jsonl` (collecte + apprentissage en recouvrement, sans
  évaluation, instantané ni checkpoint) ; appris = collecté × (appris par lot / collecté par lot) ;
- débit « mural » : timesteps appris / durée murale complète des itérations (évaluations, instantanés, checkpoints et
  journal compris). K6 (ADR-005 D6) est jugé sur le débit appris mural par tranche, le plus prudent ;
- la 1re itération d'un run neuf collecte deux lots : exclue des statistiques par itération.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import functools
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

K6_THRESHOLD = 25_000.0
"""Débit appris minimal (timesteps/s) : ADR-005 D6 K6, critère de révision de l'ADR-001 (moitié du plateau de M4)."""
K6_WINDOW_S = 30 * 60.0
TARGET_LEARNED = 1_000_000_000
"""Budget d'un run 4a en timesteps appris (ADR-005 D3) : horizon des extrapolations."""
TARGET_POOL = 100
"""Plafond du pool (ADR-005 D5) : horizon de l'extrapolation de taille de checkpoint."""
BENCH_FILES = ("bench_ticks.jsonl", "bench_samples.jsonl", "bench_summary.json", "bench_summary.txt")
CADENCES = (100, 250, 500, 1000)
SLOW_EVAL_FACTOR = 2.0
"""Une évaluation est signalée lente si elle dure plus de ce facteur fois la médiane des évaluations du banc."""


# ====================================================================================================== partie pure


def percentile(xs: Sequence[float], q: float) -> float | None:
    """Percentile `q` (0–100) par interpolation linéaire entre rangs (méthode « linear » de numpy) ; None si vide."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def dist(xs: Sequence[float]) -> dict[str, float | int | None]:
    """Médiane, p10, p90, min, max et effectif des valeurs finies."""
    ok = [float(x) for x in xs if x is not None and math.isfinite(x)]
    return {
        "n": len(ok),
        "median": percentile(ok, 50),
        "p10": percentile(ok, 10),
        "p90": percentile(ok, 90),
        "min": min(ok) if ok else None,
        "max": max(ok) if ok else None,
    }


@dataclasses.dataclass(frozen=True)
class IterRecord:
    """Une itération vue par le banc."""

    iteration: int
    t_end: float
    """Secondes depuis l'origine de la boucle (premier appel de `should_stop`, après le checkpoint de l'itération 0)."""
    gap_s: float
    """Durée murale complète de l'itération (iterate, évaluation, instantané, checkpoint, journal)."""
    loop_rate_collected: float | None
    """Débit collecté pendant `iterate` seul ; None si inconnu ou non représentatif (1re itération : deux lots)."""
    learned: int
    collected: int
    pool_size: float | None = None


def build_iter_records(
    ticks: Sequence[Mapping[str, Any]],
    metrics: Mapping[int, Mapping[str, Any]],
    *,
    start_iteration: int,
    learned_per_iter: int,
    collected_per_iter: int,
) -> list[IterRecord]:
    """Joint les ticks du banc (un par appel de `should_stop`, le premier sert d'origine) et `metrics.jsonl` (par
    step). Le tick n >= 1 clôt l'itération `start_iteration + n` (jointure exacte sans restauration)."""
    out: list[IterRecord] = []
    for n, tk in enumerate(ticks):
        if n == 0:
            continue
        it = start_iteration + n
        row = metrics.get(it, {})
        rate: Any = row.get("perf/agent_steps_per_s")
        first = it == 1 and start_iteration == 0
        ok_rate = isinstance(rate, int | float) and math.isfinite(rate) and rate > 0 and not first
        pool = row.get("pool/size")
        out.append(
            IterRecord(
                iteration=it,
                t_end=float(tk["t"]),
                gap_s=float(tk["gap_s"]),
                loop_rate_collected=float(rate) if ok_rate else None,
                learned=learned_per_iter,
                collected=collected_per_iter * (2 if first else 1),
                pool_size=float(pool) if isinstance(pool, int | float) else None,
            )
        )
    return out


def slice_stats(
    records: Sequence[IterRecord], *, slice_s: float, threshold: float = K6_THRESHOLD
) -> list[dict[str, Any]]:
    """Statistiques par tranche murale de `slice_s` (selon `t_end`) : débits boucle par itération (médiane, p10, p90),
    débits muraux de la tranche, taille max du pool, comparaison au seuil K6 (débit appris mural)."""
    if slice_s <= 0:
        raise ValueError("slice_s > 0")
    groups: dict[int, list[IterRecord]] = {}
    for r in records:
        groups.setdefault(int(r.t_end // slice_s), []).append(r)
    out: list[dict[str, Any]] = []
    for idx in sorted(groups):
        rs = groups[idx]
        dur = sum(r.gap_s for r in rs)
        coll = [r.loop_rate_collected for r in rs if r.loop_rate_collected is not None]
        learned_loop = [
            r.loop_rate_collected * r.learned / r.collected
            for r in rs
            if r.loop_rate_collected is not None and r.collected > 0
        ]
        learned = sum(r.learned for r in rs)
        collected = sum(r.collected for r in rs)
        pools = [r.pool_size for r in rs if r.pool_size is not None]
        wall_learned = learned / dur if dur > 0 else None
        out.append(
            {
                "slice": idx,
                "start_min": idx * slice_s / 60.0,
                "end_min": (idx + 1) * slice_s / 60.0,
                "duration_s": dur,
                "partial": dur < 0.9 * slice_s,
                "iterations": len(rs),
                "collected_loop": dist(coll),
                "learned_loop": dist(learned_loop),
                "learned_wall": wall_learned,
                "collected_wall": collected / dur if dur > 0 else None,
                "pool_size_max": max(pools) if pools else None,
                "below_threshold": wall_learned is not None and wall_learned < threshold,
            }
        )
    return out


def global_rates(records: Sequence[IterRecord]) -> dict[str, float | int | None]:
    """Débits globaux : muraux (toutes les itérations) et boucle (itérations au débit connu)."""
    wall = sum(r.gap_s for r in records)
    learned = sum(r.learned for r in records)
    collected = sum(r.collected for r in records)
    known = [r for r in records if r.loop_rate_collected is not None and r.loop_rate_collected > 0]
    loop_s = sum(r.collected / r.loop_rate_collected for r in known if r.loop_rate_collected)
    return {
        "iterations": len(records),
        "wall_s": wall,
        "learned": learned,
        "collected": collected,
        "learned_wall": learned / wall if wall > 0 else None,
        "collected_wall": collected / wall if wall > 0 else None,
        "learned_loop": sum(r.learned for r in known) / loop_s if loop_s > 0 else None,
        "collected_loop": sum(r.collected for r in known) / loop_s if loop_s > 0 else None,
    }


def k6_check(
    slices: Sequence[Mapping[str, Any]],
    global_learned_wall: float | None,
    *,
    threshold: float = K6_THRESHOLD,
    window_s: float = K6_WINDOW_S,
) -> dict[str, Any]:
    """K6 : débit appris (mural, par tranche) sous `threshold` pendant plus de `window_s` d'affilée. Les tranches sont
    supposées contiguës (elles le sont : elles pavent le temps de la boucle)."""
    longest = cur = 0.0
    for s in slices:
        cur = cur + float(s["duration_s"]) if s["below_threshold"] else 0.0
        longest = max(longest, cur)
    observed = sum(float(s["duration_s"]) for s in slices)
    return {
        "threshold_learned_per_s": threshold,
        "window_s": window_s,
        "global_learned_wall": global_learned_wall,
        "global_below": global_learned_wall is not None and global_learned_wall < threshold,
        "longest_below_s": longest,
        "slices_below": [int(s["slice"]) for s in slices if s["below_threshold"]],
        "triggered": longest > window_s,
        "observed_s": observed,
        "conclusive": observed > window_s,
    }


def throttle_ratio(slices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Médiane du débit appris boucle de la dernière tranche complète sur celle de la première (< 1 : ralenti)."""
    full = [s for s in slices if not s["partial"] and s["learned_loop"]["median"] is not None]
    if len(full) < 2:
        return {"first": None, "last": None, "ratio": None}
    a, b = full[0]["learned_loop"]["median"], full[-1]["learned_loop"]["median"]
    return {"first": a, "last": b, "ratio": b / a if a else None}


def extrapolate_linear(value: float, done: float, target: float) -> float | None:
    """`value` mesuré après `done` unités, ramené à `target` unités (proportionnel) ; None si `done` <= 0."""
    if done <= 0:
        return None
    return value * target / done


def fit_slope(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """Moindres carrés y = a·x + b ; None s'il y a moins de deux x distincts."""
    if len(xs) != len(ys) or len(set(xs)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return a, my - a * mx


def checkpoint_projection(
    ckpts: Sequence[Mapping[str, Any]],
    *,
    unit_bytes: int | None,
    target_nets: int = TARGET_POOL,
    iteration_s: float | None = None,
    cadences: Iterable[int] = CADENCES,
) -> dict[str, Any]:
    """Taille et durée d'un checkpoint extrapolées à `target_nets` instantanés de pool, depuis le dernier checkpoint
    mesuré : + (target − nets) × `unit_bytes` (taille sérialisée d'une politique) ; durées proportionnelles à la
    taille (hypothèse, étiquetée « extrapolé »). Pente mesurée (octets par instantané) si au moins deux tailles de pool
    distinctes. Coût relatif par cadence : durée projetée / (cadence × durée d'une itération)."""
    valid = [c for c in ckpts if c.get("nets") is not None and c.get("bytes")]
    if not valid:
        return {"available": False}
    last = valid[-1]
    nets, size = int(last["nets"]), float(last["bytes"])
    # le checkpoint de l'itération 0 n'a pas encore de lot en attente (plus petit) : hors de l'ajustement
    later = [c for c in valid if int(c.get("iteration") or 0) > 0]
    fit = fit_slope([float(c["nets"]) for c in later], [float(c["bytes"]) for c in later])
    per_net = unit_bytes if unit_bytes is not None else (fit[0] if fit else None)
    out: dict[str, Any] = {
        "available": True,
        "label": "extrapolé",
        "from_iteration": last.get("iteration"),
        "from_nets": nets,
        "from_bytes": size,
        "unit_bytes_measured": unit_bytes,
        "slope_bytes_per_net_fit": fit[0] if fit else None,
        "target_nets": target_nets,
    }
    if per_net is None:
        out["bytes"] = None
        return out
    proj = size + max(target_nets - nets, 0) * per_net
    scale = proj / size if size > 0 else None
    cap = float(last.get("capture_s") or 0.0)
    wr = float(last.get("write_s") or 0.0)
    out["bytes"] = proj
    out["capture_s"] = cap * scale if scale else None
    out["write_s"] = wr * scale if scale else None
    total = (cap + wr) * scale if scale else None
    out["total_s"] = total
    if iteration_s and total is not None:
        out["cadences"] = [
            {
                "every_iterations": n,
                "minutes_between": n * iteration_s / 60.0,
                "cost_fraction": total / (n * iteration_s),
                "replay_max_min": n * iteration_s / 60.0,
            }
            for n in cadences
        ]
    return out


def nets_at(pool_adds: Sequence[Mapping[str, Any]], iteration: int) -> int | None:
    """Poids d'instantanés en mémoire après le dernier ajout au pool d'itération <= `iteration` (None avant le 1er)."""
    best: Mapping[str, Any] | None = None
    for a in pool_adds:
        if int(a["iteration"]) <= iteration:
            best = a
    return None if best is None else int(best["nets"])


def eval_analysis(
    evals: Sequence[Mapping[str, Any]],
    tick_by_iteration: Mapping[int, Mapping[str, Any]],
    *,
    slow_factor: float = SLOW_EVAL_FACTOR,
) -> dict[str, Any]:
    """Une ligne par évaluation (durées, épisodes, minutes de jeu, steps d'arène simulés, débits de simulation) et
    diagnostic des évaluations lentes : machine ralentie (débit de simulation bas), épisodes plus longs (plus de steps
    simulés) ou temps hors simulation (politique, observations)."""
    rows: list[dict[str, Any]] = []
    for e in evals:
        if e.get("event") != "eval":
            continue
        it = int(e["iteration"])
        timing = e.get("timing") or {}
        rnd, frz = e.get("random") or {}, e.get("frozen") or {}
        tk = tick_by_iteration.get(it, {})
        steps = tk.get("eval_arena_steps")
        sim_s = tk.get("eval_sim_s")
        env_s = tk.get("eval_env_step_s")
        total = timing.get("total_s")
        minutes = float(rnd.get("minutes", 0.0)) + float(frz.get("minutes", 0.0))
        episodes = int(rnd.get("episodes", 0)) + int(frz.get("episodes", 0))
        rows.append(
            {
                "k": e.get("k"),
                "iteration": it,
                "trained_timesteps": e.get("trained_timesteps"),
                "passed": e.get("passed"),
                "reason": e.get("reason"),
                "total_s": total,
                "snapshot_s": timing.get("snapshot_s"),
                "random_s": timing.get("random_s"),
                "frozen_s": timing.get("frozen_s"),
                "episodes": episodes,
                "game_minutes": minutes,
                "decisive_random": rnd.get("decisive"),
                "arena_steps": steps,
                "sim_s": sim_s,
                "env_step_s": env_s,
                "sim_arena_steps_per_s": (steps / sim_s) if steps and sim_s else None,
                "arena_steps_per_wall_s": (steps / total) if steps and total else None,
                "game_minutes_per_wall_s": (minutes / total) if total else None,
                "non_sim_s": (total - env_s) if total is not None and env_s is not None else None,
            }
        )
    totals = [r["total_s"] for r in rows if r["total_s"] is not None]
    med = percentile(totals, 50)
    med_rate = percentile([r["sim_arena_steps_per_s"] for r in rows if r["sim_arena_steps_per_s"]], 50)
    med_steps = percentile([float(r["arena_steps"]) for r in rows if r["arena_steps"]], 50)
    med_non_sim = percentile([r["non_sim_s"] for r in rows if r["non_sim_s"] is not None], 50)
    for r in rows:
        slow = med is not None and r["total_s"] is not None and len(totals) >= 2 and r["total_s"] > slow_factor * med
        r["slow"] = slow
        if not slow:
            r["diagnosis"] = None
            continue
        causes = []
        if med_rate and r["sim_arena_steps_per_s"] and r["sim_arena_steps_per_s"] < med_rate / 1.5:
            causes.append("machine ralentie (débit de simulation bas)")
        if med_steps and r["arena_steps"] and r["arena_steps"] > 1.5 * med_steps:
            causes.append("épisodes plus longs (plus de steps simulés)")
        if med_non_sim is not None and r["non_sim_s"] is not None and r["non_sim_s"] > 2 * max(med_non_sim, 1e-9):
            causes.append("temps hors simulation (politique, observations)")
        r["diagnosis"] = " ; ".join(causes) if causes else "non expliqué par les compteurs"
    return {
        "rows": rows,
        "count": len(rows),
        "total_s": sum(totals),
        "median_s": med,
        "slow": [r["iteration"] for r in rows if r["slow"]],
    }


def ram_summary(
    samples: Sequence[Mapping[str, Any]], *, bin_s: float = 60.0, horizon_h: float = 8.0, warmup_s: float = 300.0
) -> dict[str, Any]:
    """RSS (working set) et RAM disponible : max, min, courbe par `bin_s` (RSS max et disponible min de chaque case),
    pente de la RSS après `warmup_s` (moindres carrés) et RSS extrapolée à `horizon_h` heures (étiquetée)."""
    rss = [(float(s["t"]), float(s["rss"])) for s in samples if s.get("rss") is not None]
    avail = [(float(s["t"]), float(s["avail"])) for s in samples if s.get("avail") is not None]
    out: dict[str, Any] = {"n": len(samples)}
    if rss:
        out["rss_max"] = max(v for _, v in rss)
        out["rss_first"], out["rss_last"] = rss[0][1], rss[-1][1]
        late = [(t, v) for t, v in rss if t >= warmup_s]
        fit = fit_slope([t for t, _ in late], [v for _, v in late]) if len(late) >= 3 else None
        out["rss_slope_bytes_per_h"] = fit[0] * 3600.0 if fit else None
        out["rss_at_horizon_extrapolated"] = (fit[0] * horizon_h * 3600.0 + fit[1]) if fit else None
        out["horizon_h"] = horizon_h
    if avail:
        out["avail_min"] = min(v for _, v in avail)
        out["avail_first"] = avail[0][1]
    private = [float(s["private"]) for s in samples if s.get("private") is not None]
    if private:
        out["private_max"] = max(private)
    curve: dict[int, dict[str, float]] = {}
    for s in samples:
        b = int(float(s["t"]) // bin_s)
        c = curve.setdefault(b, {"t_min": b * bin_s / 60.0})
        if s.get("rss") is not None:
            c["rss_max"] = max(c.get("rss_max", 0.0), float(s["rss"]))
        if s.get("avail") is not None:
            c["avail_min"] = min(c.get("avail_min", math.inf), float(s["avail"]))
    out["curve"] = [curve[b] for b in sorted(curve)]
    ac = {s.get("ac_line") for s in samples if s.get("ac_line") is not None}
    out["ac_line_values"] = sorted(ac)
    out["on_battery_seen"] = 0 in ac
    return out


def log_projection(sizes: Mapping[str, int], trained: int, target: int = TARGET_LEARNED) -> dict[str, Any]:
    """Tailles de journaux ramenées à `target` timesteps appris (linéaire, étiqueté « extrapolé »).
    `evalhook_state.jsonl` réécrit tout l'historique à chaque ligne : croissance quadratique, d'où une borne haute
    (rapport au carré) en plus."""
    r = target / trained if trained > 0 else None
    files: dict[str, Any] = {}
    for name, size in sorted(sizes.items()):
        per_segment = name.startswith("run_meta") or name == "segments.jsonl"  # une entrée par session, pas par step
        rec: dict[str, Any] = {"bytes": size, "extrapolated_linear": size if per_segment else (size * r if r else None)}
        if per_segment:
            rec["per_segment"] = True
        if name == "evalhook_state.jsonl":
            rec["extrapolated_quadratic_bound"] = size * r * r if r else None
        files[name] = rec
    total = sum(sizes.values())
    return {
        "label": "extrapolé",
        "trained": trained,
        "target": target,
        "files": files,
        "total_bytes": total,
        "total_extrapolated_linear": total * r if r else None,
    }


def unique_name(prefix: str, now: datetime.datetime, taken: Callable[[str], bool]) -> str:
    """`<prefix>_<AAAAMMJJTHHMMSSZ>`, suffixé `_001`... si déjà pris."""
    base = f"{prefix}_{now.strftime('%Y%m%dT%H%M%SZ')}"
    name, n = base, 0
    while taken(name):
        n += 1
        name = f"{base}_{n:03d}"
    return name


def _mb(x: float | None) -> str:
    return "—" if x is None else f"{x / 1e6:,.1f} Mo".replace(",", " ")


def _k(x: float | None) -> str:
    return "—" if x is None else f"{x / 1e3:,.1f}k".replace(",", " ")


def _s(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f} s"


def render_table(summary: Mapping[str, Any]) -> str:
    """Tableau lisible du résumé (texte brut, français)."""
    L: list[str] = []
    p = summary.get("provenance", {})
    L.append(f"Banc A5 : {summary.get('name')}  ({summary.get('label')})")
    L.append(
        f"commit {p.get('git_commit')} (sale : {p.get('git_dirty')}), seed {p.get('seed')}, "
        f"device {p.get('device_resolved')}, {p.get('date_utc_start')} -> {p.get('date_utc_end')}"
    )
    L.append(f"matériel : {p.get('cpu_name')} ; {p.get('xpu_name')} ; RAM {_mb(p.get('ram_total'))}")
    L.append(f"commande : {' '.join(map(str, p.get('command', [])))}")
    run_s = summary.get("run", {})
    L.append(
        f"run : statut {run_s.get('status')} ({run_s.get('cause')}), itérations {run_s.get('start_iteration')} -> "
        f"{run_s.get('end_iteration')}, appris {run_s.get('trained_timesteps')}, restaurations {run_s.get('restores')}"
    )
    L.append("")
    L.append("Débit par tranche (boucle = iterate seul ; mural = tout compris ; k = milliers de timesteps/s)")
    L.append(
        f"{'tranche':>11} {'durée':>7} {'it.':>5} {'coll. méd [p10;p90]':>24} {'appris méd [p10;p90]':>24} "
        f"{'appris mural':>12} {'pool':>5} {'<25k':>5}"
    )
    for s in summary.get("slices", []):
        c, a = s["collected_loop"], s["learned_loop"]
        L.append(
            f"{s['start_min']:>5.1f}-{s['end_min']:<5.1f} {s['duration_s']:>6.0f}s {s['iterations']:>5} "
            f"{_k(c['median']):>8} [{_k(c['p10'])};{_k(c['p90'])}] "
            f"{_k(a['median']):>8} [{_k(a['p10'])};{_k(a['p90'])}] "
            f"{_k(s['learned_wall']):>12} {s['pool_size_max'] if s['pool_size_max'] is not None else '—':>5} "
            f"{'OUI' if s['below_threshold'] else 'non':>5}{' (partielle)' if s['partial'] else ''}"
        )
    g = summary.get("global", {})
    L.append(
        f"global : appris mural {_k(g.get('learned_wall'))}/s, appris boucle {_k(g.get('learned_loop'))}/s, "
        f"collecté mural {_k(g.get('collected_wall'))}/s, collecté boucle {_k(g.get('collected_loop'))}/s"
    )
    k6 = summary.get("k6", {})
    L.append(
        f"K6 (appris mural < {_k(k6.get('threshold_learned_per_s'))}/s > 30 min) : "
        f"{'DÉCLENCHÉ' if k6.get('triggered') else 'non déclenché'} ; plus longue série sous le seuil "
        f"{(k6.get('longest_below_s') or 0) / 60:.1f} min ; global sous le seuil : {k6.get('global_below')} ; "
        f"concluant (> 30 min observées) : {k6.get('conclusive')}"
    )
    th = summary.get("throttle", {})
    if th.get("ratio") is not None:
        L.append(f"throttling : appris boucle dernière/première tranche complète = {th['ratio']:.2f}")
    ext = summary.get("extrapolation_1b", {})
    if ext.get("hours") is not None:
        L.append(f"durée d'un run de 1B au débit mural global : {ext['hours']:.2f} h (extrapolé)")
    L.append("")
    b = summary.get("breakdown", {})
    L.append("Temps mural")
    for key, lab in (
        ("setup_s", "assemblage + θ₀ + checkpoint 0"),
        ("loop_wall_s", "boucle (origine -> dernier tick)"),
        ("iterate_s", "  iterate (collecte || apprentissage)"),
        ("evals_s", "  évaluations"),
        ("checkpoints_s", "  checkpoints (capture + écriture)"),
        ("pool_snapshots_s", "  instantanés du pool"),
        ("other_s", "  reste (journal, gardes, crochet)"),
        ("teardown_s", "fin (checkpoint final, fermeture)"),
        ("collect_s", "collecte (fil principal)"),
        ("collect_env_step_s", "  env.step entraînement"),
        ("collect_sim_step_s", "    dont sim.step"),
        ("collect_obs_reward_s", "    dont obs + récompense + resets"),
        ("collect_other_s", "  inférence acteur + pool + tampons"),
        ("learn_s", "apprentissage (fil learner)"),
    ):
        v = b.get(key)
        frac = b.get("fractions", {}).get(key)
        L.append(f"  {lab:<42} {_s(v):>12}" + (f"  {100 * frac:5.1f} %" if frac is not None else ""))
    L.append("")
    ev = summary.get("evals", {})
    L.append(f"Évaluations : {ev.get('count', 0)} (total {_s(ev.get('total_s'))}, médiane {_s(ev.get('median_s'))})")
    L.append(
        f"{'it.':>6} {'total':>8} {'aléa':>7} {'gelé':>7} {'épis.':>6} {'min jeu':>8} {'steps arène':>12} "
        f"{'sim/s':>9} {'hors sim':>8} lente"
    )
    for r in ev.get("rows", []):
        L.append(
            f"{r['iteration']:>6} {_s(r['total_s']):>8} {_s(r['random_s']):>7} {_s(r['frozen_s']):>7} "
            f"{r['episodes']:>6} {r['game_minutes']:>8.1f} {r['arena_steps'] or 0:>12} "
            f"{_k(r['sim_arena_steps_per_s']):>9} {_s(r['non_sim_s']):>8} "
            f"{'OUI : ' + str(r['diagnosis']) if r['slow'] else 'non'}"
        )
    L.append("")
    ck = summary.get("checkpoints", {})
    L.append(f"Checkpoints : {len(ck.get('rows', []))}")
    for r in ck.get("rows", []):
        L.append(
            f"  it {r['iteration']:>6} {_mb(r['bytes']):>10} capture {_s(r['capture_s'])} écriture {_s(r['write_s'])} "
            f"instantanés {r.get('nets')}{' (final)' if r.get('final') else ''}"
        )
    pr = ck.get("projection", {})
    if pr.get("available") and pr.get("bytes") is not None:
        L.append(
            f"  à {pr['target_nets']} instantanés (extrapolé) : {_mb(pr['bytes'])}, capture {_s(pr.get('capture_s'))}, "
            f"écriture {_s(pr.get('write_s'))} ; unité mesurée {_mb(pr.get('unit_bytes_measured'))}, pente ajustée "
            f"{_mb(pr.get('slope_bytes_per_net_fit'))}"
        )
        for c in pr.get("cadences", []):
            L.append(
                f"    tous les {c['every_iterations']:>5} it. : ≈ {c['minutes_between']:.1f} min, coût "
                f"{100 * c['cost_fraction']:.2f} %, rejeu max à la restauration {c['replay_max_min']:.1f} min"
            )
    probe = summary.get("checkpoint_probe")
    if probe:
        L.append(
            f"  sonde directe ({probe.get('n_snapshots')} instantanés, mesuré) : {_mb(probe.get('bytes'))}, "
            f"écriture {_s(probe.get('save_s'))}, relecture {_s(probe.get('reload_s'))}"
        )
    L.append("")
    ps = summary.get("pool", {})
    L.append(
        f"Pool : {ps.get('adds', 0)} instantanés ajoutés, durée {_s(ps.get('total_s'))} "
        f"(méd. {_s(ps.get('median_s'))}), taille finale {ps.get('final_size')}"
    )
    es = summary.get("eval_snapshots", {})
    L.append(
        f"Instantanés d'évaluation : {es.get('count')} fichiers, {_mb(es.get('bytes'))} ; à 1B "
        f"{_mb(es.get('extrapolated_bytes'))} (extrapolé)"
    )
    L.append("")
    ram = summary.get("ram", {})
    L.append(
        f"RAM : RSS max {_mb(ram.get('rss_max'))} (début {_mb(ram.get('rss_first'))}, fin {_mb(ram.get('rss_last'))}), "
        f"privée max {_mb(ram.get('private_max'))}, disponible min {_mb(ram.get('avail_min'))} ; pente RSS "
        f"{_mb(ram.get('rss_slope_bytes_per_h'))}/h, RSS à {ram.get('horizon_h')} h "
        f"{_mb(ram.get('rss_at_horizon_extrapolated'))} (extrapolé) ; batterie vue : {ram.get('on_battery_seen')}"
    )
    lg = summary.get("logs", {})
    L.append(f"Journaux : {_mb(lg.get('total_bytes'))}, à 1B {_mb(lg.get('total_extrapolated_linear'))} (extrapolé)")
    for name, rec in lg.get("files", {}).items():
        extra = rec.get("extrapolated_quadratic_bound")
        L.append(
            f"  {name:<26} {_mb(rec['bytes']):>10} -> {_mb(rec['extrapolated_linear'])}"
            + (f" (borne quadratique {_mb(extra)})" if extra is not None else "")
        )
    warns = summary.get("warnings", [])
    if warns:
        L.append("")
        L.append("Avertissements :")
        L.extend(f"  - {w}" for w in warns)
    return "\n".join(L) + "\n"


# ====================================================================================================== mesures OS


def _win_memory() -> dict[str, int] | None:
    """RSS (working set) et mémoire privée du processus, RAM physique totale et disponible (Windows, ctypes)."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class MSX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    msx = MSX()
    msx.dwLength = ctypes.sizeof(MSX)
    ok1 = k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    ok2 = k32.GlobalMemoryStatusEx(ctypes.byref(msx))
    out: dict[str, int] = {}
    if ok1:
        out.update(rss=int(pmc.WorkingSetSize), rss_peak=int(pmc.PeakWorkingSetSize), private=int(pmc.PrivateUsage))
    if ok2:
        out.update(avail=int(msx.ullAvailPhys), total=int(msx.ullTotalPhys), load=int(msx.dwMemoryLoad))
    return out


def _win_power() -> dict[str, int | None] | None:
    """Secteur (1), batterie (0) ou inconnu (255) ; pourcentage de batterie (255 = inconnu)."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class SPS(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", wintypes.BYTE),
            ("BatteryFlag", wintypes.BYTE),
            ("BatteryLifePercent", wintypes.BYTE),
            ("SystemStatusFlag", wintypes.BYTE),
            ("BatteryLifeTime", wintypes.DWORD),
            ("BatteryFullLifeTime", wintypes.DWORD),
        ]

    sps = SPS()
    if not ctypes.WinDLL("kernel32").GetSystemPowerStatus(ctypes.byref(sps)):
        return None
    ac = int(sps.ACLineStatus) & 0xFF
    pct = int(sps.BatteryLifePercent) & 0xFF
    return {
        "ac_line": None if ac == 255 else ac,
        "battery_pct": None if pct == 255 else pct,
        "saver": int(sps.SystemStatusFlag) & 0xFF,
    }


def _cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "inconnu"


class Sampler(threading.Thread):
    """Échantillonne RAM et alimentation toutes les `interval_s` secondes ; écrit chaque échantillon dans `path`."""

    def __init__(self, path: Path, interval_s: float, t0: float, xpu: bool) -> None:
        super().__init__(name="bench-sampler", daemon=True)
        self.path, self.interval_s, self.t0, self.xpu = path, interval_s, t0, xpu
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()

    def sample(self) -> dict[str, Any]:
        rec: dict[str, Any] = {"t": time.perf_counter() - self.t0, "unix": time.time()}
        rec.update(_win_memory() or {})
        rec.update(_win_power() or {})
        if self.xpu:
            with contextlib.suppress(Exception):
                import torch

                rec["xpu_reserved"] = int(torch.xpu.memory_reserved(0))
                rec["xpu_allocated"] = int(torch.xpu.memory_allocated(0))
        return rec

    def run(self) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            while True:
                rec = self.sample()
                self.samples.append(rec)
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                if self._stop_event.wait(self.interval_s):
                    break

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=30)


# ====================================================================================================== instrumentation


class Instrument:
    """Enveloppes de chronométrage posées le temps du run (retirées dans `uninstall`), compteurs par rôle
    (« train » : simulateur et env d'entraînement ; « eval » : ceux de l'évaluation, reconnus au nombre d'arènes)."""

    def __init__(self, train_envs: int, eval_envs: int) -> None:
        if train_envs == eval_envs:
            raise ValueError("arènes d'entraînement et d'évaluation identiques : rôles indiscernables")
        self.train_envs, self.eval_envs = train_envs, eval_envs
        self.c: dict[str, float] = {}
        self.pool_adds: list[dict[str, Any]] = []
        self._orig: list[tuple[Any, str, Any]] = []
        self._lock = threading.Lock()

    def role(self, n_envs: int) -> str:
        return "train" if n_envs == self.train_envs else "eval" if n_envs == self.eval_envs else f"n{n_envs}"

    def add(self, key: str, value: float) -> None:
        with self._lock:
            self.c[key] = self.c.get(key, 0.0) + value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self.c)

    def _patch(self, owner: Any, name: str, make: Callable[[Any], Any]) -> None:
        orig = getattr(owner, name)
        self._orig.append((owner, name, orig))
        setattr(owner, name, functools.wraps(orig)(make(orig)))

    def install(self) -> None:
        from env.batch_env import BatchEnv
        from env.sim_adapter import RocketGPUSimulator
        from training.ppo import Learner
        from training.rollout import Collector
        from training.selfplay import PastOpponents

        inst = self

        def timed(key: str) -> Callable[[Any], Any]:
            def make(orig: Any) -> Any:
                def wrapper(self: Any, *a: Any, **k: Any) -> Any:
                    t0 = time.perf_counter()
                    try:
                        return orig(self, *a, **k)
                    finally:
                        inst.add(key + "_s", time.perf_counter() - t0)
                        inst.add(key + "_n", 1)

                return wrapper

            return make

        def by_role(prefix: str, arena_steps: bool) -> Callable[[Any], Any]:
            def make(orig: Any) -> Any:
                def wrapper(self: Any, *a: Any, **k: Any) -> Any:
                    t0 = time.perf_counter()
                    try:
                        return orig(self, *a, **k)
                    finally:
                        r = inst.role(int(self.n_envs))
                        inst.add(f"{r}_{prefix}_s", time.perf_counter() - t0)
                        inst.add(f"{r}_{prefix}_n", 1)
                        if arena_steps:
                            inst.add(f"{r}_arena_steps", int(self.n_envs))

                return wrapper

            return make

        def pool_add(orig: Any) -> Any:
            def wrapper(self: Any, policy: Any, iteration: int) -> Any:
                t0 = time.perf_counter()
                uid = orig(self, policy, iteration)
                dt = time.perf_counter() - t0
                inst.add("pool_add_s", dt)
                inst.pool_adds.append(
                    {"iteration": int(iteration), "s": dt, "size": len(self.pool), "nets": len(self.nets)}
                )
                return uid

            return wrapper

        self._patch(Learner, "update", timed("learn"))
        self._patch(Collector, "collect", timed("collect"))
        self._patch(BatchEnv, "step", by_role("env_step", False))
        self._patch(RocketGPUSimulator, "step", by_role("sim_step", True))
        self._patch(PastOpponents, "add_snapshot", pool_add)

    def uninstall(self) -> None:
        while self._orig:
            owner, name, orig = self._orig.pop()
            setattr(owner, name, orig)


class Ticker:
    """Appelé à chaque `should_stop` (en tête de chaque itération de `run`) : enregistre l'instant et les compteurs
    écoulés depuis l'appel précédent, puis dit s'il faut arrêter (durée murale atteinte)."""

    def __init__(self, inst: Instrument, path: Path, t_bench0: float, deadline_s: float) -> None:
        self.inst, self.t_bench0, self.deadline_s = inst, t_bench0, deadline_s
        self.ticks: list[dict[str, Any]] = []
        self.origin: float | None = None
        self._prev_t = 0.0
        self._prev_c: dict[str, float] = {}
        self._f = open(path, "a", encoding="utf-8")

    def __call__(self) -> bool:
        now = time.perf_counter()
        c = self.inst.snapshot()
        if self.origin is None:
            self.origin = now
        delta = {k: v - self._prev_c.get(k, 0.0) for k, v in c.items()}
        rec = {
            "n": len(self.ticks),
            "t": now - self.origin,
            "t_bench": now - self.t_bench0,
            "unix": time.time(),
            "gap_s": (now - self.origin) - self._prev_t if self.ticks else 0.0,
            "eval_arena_steps": int(delta.get("eval_arena_steps", 0)),
            "eval_sim_s": delta.get("eval_sim_step_s", 0.0),
            "eval_env_step_s": delta.get("eval_env_step_s", 0.0),
            **{k: v for k, v in delta.items() if not k.startswith("eval_")},
        }
        self.ticks.append(rec)
        self._f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._f.flush()
        self._prev_t, self._prev_c = rec["t"], c
        return now - self.t_bench0 >= self.deadline_s

    def close(self) -> None:
        self._f.close()


# ====================================================================================================== sonde


def checkpoint_probe(src: Path, probe_dir: Path, n_snapshots: int, device: str, keep: bool) -> dict[str, Any]:
    """Recharge `src`, porte le pool à `n_snapshots` instantanés (copies distinctes des poids présents, sur `device`
    comme dans le run), puis l'écrit avec `CheckpointManager.save` dans `probe_dir` (hors run) : taille et durée
    réelles (sérialisation, somme de contrôle, écriture atomique avec fsync, vérification de rotation), puis relecture.
    Le fichier sondé est supprimé ensuite (dossier temporaire du banc) sauf `keep`."""
    import numpy as np
    import torch

    from training.checkpoint import CheckpointManager

    t0 = time.perf_counter()
    state = CheckpointManager.load(src)
    load_s = time.perf_counter() - t0
    opp = state["trainer"]["collector"].get("opponents")
    if not opp or not opp["nets"]:
        raise RuntimeError(f"{src.name} : pas d'état de pool à étendre")
    base = opp["nets"]
    dev = torch.device(device)
    opp["nets"] = [{k: v.clone().to(dev) for k, v in base[i % len(base)].items()} for i in range(n_snapshots)]
    opp["net_uids"] = list(range(n_snapshots))
    pool = opp["pool"]
    pool["uids"] = list(range(n_snapshots))
    pool["q"] = [0.0] * n_snapshots
    pool["iteration"] = [50 * i for i in range(n_snapshots)]
    pool["next_uid"] = n_snapshots
    opp["assignment"] = np.asarray(opp["assignment"], dtype=np.int64) % n_snapshots
    mgr = CheckpointManager(probe_dir, keep=10**9)
    t1 = time.perf_counter()
    path = mgr.save(state, healthy=True)
    save_s = time.perf_counter() - t1
    size = path.stat().st_size
    t2 = time.perf_counter()
    CheckpointManager.load(path)
    reload_s = time.perf_counter() - t2
    out = {
        "label": "mesuré (hors run)",
        "source": str(src),
        "source_bytes": src.stat().st_size,
        "source_load_s": load_s,
        "n_snapshots": n_snapshots,
        "source_nets": len(base),
        "device": str(dev),
        "bytes": size,
        "save_s": save_s,
        "reload_s": reload_s,
        "path": str(path),
        "kept": keep,
    }
    if not keep:
        path.unlink()
        with contextlib.suppress(OSError):
            probe_dir.rmdir()
    return out


# ====================================================================================================== journaux


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lignes JSON lisibles de `path` (lignes tronquées ignorées) ; [] si absent."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_train_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_rocketgpu_defaults() -> dict[str, str | None]:
    """Variables RocketGPU par défaut (build PGO et meshes de `vendor/`, comme le smoke test) si absentes."""
    defaults = {
        "ROCKETGPU_PY_DIR": ROOT / "vendor" / "build" / "rocketgpu-cpu-pgo" / "python",
        "ROCKETGPU_COLLISION_MESH_DIR": ROOT / "vendor" / "collision_meshes",
    }
    for k, v in defaults.items():
        if not os.environ.get(k) and v.is_dir():
            os.environ[k] = str(v)
    return {k: os.environ.get(k) for k in (*defaults, "ROCKETGPU_DLL_DIR")}


# ====================================================================================================== résumé


def summarize_bench(
    *,
    name: str,
    provenance: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    ticks: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    evals: Sequence[Mapping[str, Any]],
    pool_adds: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    log_sizes: Mapping[str, int],
    eval_snapshots: Mapping[str, Any],
    learned_per_iter: int,
    collected_per_iter: int,
    unit_bytes: int | None,
    eval_every: int,
    t_run_start: float,
    t_run_end: float,
    slice_s: float = 300.0,
    threshold: float = K6_THRESHOLD,
    probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble le résumé JSON du banc à partir des mesures brutes (fonction pure : testée sans run).
    `t_run_start`, `t_run_end` : appel et retour de `run`, en secondes depuis le début du banc (même horloge que
    `t_bench` des ticks)."""
    warnings: list[str] = []
    if provenance.get("git_dirty"):
        warnings.append("arbre git sale : mesure non attribuable à un commit")
    start_it = int(run_summary.get("start_iteration", 0))
    metrics = {int(r["step"]): r for r in metrics_rows if "step" in r}
    records = build_iter_records(
        ticks,
        metrics,
        start_iteration=start_it,
        learned_per_iter=learned_per_iter,
        collected_per_iter=collected_per_iter,
    )
    restores = [e for e in events if e.get("event") == "restore"]
    if restores:
        warnings.append(f"{len(restores)} restauration(s) pendant le banc : jointure ticks/metrics approximative")
    slices = slice_stats(records, slice_s=slice_s, threshold=threshold)
    glob = global_rates(records)
    k6 = k6_check(slices, glob["learned_wall"], threshold=threshold)
    if not k6["conclusive"]:
        warnings.append("moins de 30 min de boucle observées : K6 non concluant")
    gaps = [r.gap_s for r in records]
    iter_s = percentile(gaps, 50)

    # --- temps
    ck_events = [e for e in events if e.get("event") == "checkpoint"]
    origin_bench = float(ticks[0]["t_bench"]) if ticks else None
    last_bench = float(ticks[-1]["t_bench"]) if ticks else None
    loop_wall = (last_bench - origin_bench) if origin_bench is not None and last_bench is not None else None
    # checkpoint de la boucle : après le 1er tick et avant le dernier (horodatage unix) ; le checkpoint 0 précède
    # l'origine (assemblage), le checkpoint final suit le dernier tick (fin)
    first_unix = float(ticks[0].get("unix", -math.inf)) if ticks else -math.inf
    last_unix = float(ticks[-1].get("unix", math.inf)) if ticks else math.inf

    def phase(e: Mapping[str, Any]) -> str:
        t = float(e.get("t", 0.0))
        return "setup" if t < first_unix else "final" if t > last_unix else "loop"

    in_loop_ck = [e for e in ck_events if phase(e) == "loop"]

    def sum_t(key: str) -> float:
        return sum(float(t.get(key, 0.0)) for t in ticks)

    ev = eval_analysis(evals, {r.iteration: ticks[i + 1] for i, r in enumerate(records)})
    pool_in_loop = [a for a in pool_adds if int(a["iteration"]) > 0]
    iterate_s = run_summary.get("loop_s")
    ck_loop_s = sum(float(e["capture_s"]) + float(e["write_s"]) for e in in_loop_ck)
    pool_s = sum(float(a["s"]) for a in pool_in_loop)
    other = None
    if loop_wall is not None and iterate_s is not None:
        other = loop_wall - float(iterate_s) - ev["total_s"] - ck_loop_s - pool_s
    collect_s, env_s, sim_s = sum_t("collect_s"), sum_t("train_env_step_s"), sum_t("train_sim_step_s")
    breakdown: dict[str, Any] = {
        "setup_s": (origin_bench - t_run_start) if origin_bench is not None else None,
        "loop_wall_s": loop_wall,
        "iterate_s": iterate_s,
        "evals_s": ev["total_s"],
        "checkpoints_s": ck_loop_s,
        "pool_snapshots_s": pool_s,
        "other_s": other,
        "teardown_s": (t_run_end - last_bench) if last_bench is not None else None,
        "collect_s": collect_s,
        "collect_env_step_s": env_s,
        "collect_sim_step_s": sim_s,
        "collect_obs_reward_s": env_s - sim_s,
        "collect_other_s": collect_s - env_s,
        "learn_s": sum_t("learn_s"),
    }
    if loop_wall:
        breakdown["fractions"] = {
            k: float(v) / loop_wall
            for k, v in breakdown.items()
            if k not in ("setup_s", "loop_wall_s", "teardown_s") and isinstance(v, int | float)
        }
    breakdown["note"] = (
        "collecte (fil principal) et apprentissage (fil learner) se recouvrent : leurs fractions ne s'additionnent "
        "pas ; l'inférence de l'acteur et des instantanés du pool est dans « inférence acteur + pool + tampons »"
    )
    if ev["slow"]:
        warnings.append(f"évaluation(s) anormalement lente(s) aux itérations {ev['slow']}")

    # --- checkpoints
    ck_rows = []
    for e in ck_events:
        it = int(e["iteration"])
        ck_rows.append(
            {
                "iteration": it,
                "file": e.get("file"),
                "bytes": e.get("bytes"),
                "capture_s": e.get("capture_s"),
                "write_s": e.get("write_s"),
                "healthy": e.get("healthy"),
                "nets": nets_at(pool_adds, it),
                "phase": phase(e),
                "final": phase(e) == "final",
            }
        )
    projection = checkpoint_projection(ck_rows, unit_bytes=unit_bytes, iteration_s=iter_s)

    trained = int(run_summary.get("trained_timesteps", 0)) - start_it * learned_per_iter
    n_evals_1b = TARGET_LEARNED // eval_every if eval_every > 0 else 0
    snap_count = int(eval_snapshots.get("count", 0))
    snap_bytes = int(eval_snapshots.get("bytes", 0))
    pool_sizes = [{"iteration": a["iteration"], "size": a["size"], "nets": a["nets"]} for a in pool_adds]
    lw = glob["learned_wall"]
    out: dict[str, Any] = {
        "name": name,
        "label": "mesure du banc A5",
        "provenance": dict(provenance),
        "run": dict(run_summary),
        "iteration_s": dist(gaps),
        "slices": slices,
        "slice_s": slice_s,
        "global": glob,
        "k6": k6,
        "throttle": throttle_ratio(slices),
        "extrapolation_1b": {
            "label": "extrapolé (débit mural global du banc maintenu)",
            "hours": (TARGET_LEARNED / lw / 3600.0) if lw else None,
        },
        "breakdown": breakdown,
        "evals": ev,
        "checkpoints": {"rows": ck_rows, "projection": projection},
        "checkpoint_probe": dict(probe) if probe else None,
        "pool": {
            "adds": len(pool_adds),
            "total_s": sum(float(a["s"]) for a in pool_adds),
            "median_s": percentile([float(a["s"]) for a in pool_adds], 50),
            "max_s": max((float(a["s"]) for a in pool_adds), default=None),
            "final_size": pool_adds[-1]["size"] if pool_adds else None,
            "over_time": pool_sizes,
        },
        "eval_snapshots": {
            **dict(eval_snapshots),
            "extrapolated_bytes": (snap_bytes / snap_count * (n_evals_1b + 1)) if snap_count else None,
            "label": f"extrapolé ({n_evals_1b} évaluations + θ₀ à 1B)",
        },
        "ram": ram_summary(samples),
        "logs": log_projection(log_sizes, trained),
        "warnings": warnings,
    }
    if out["ram"].get("on_battery_seen"):
        warnings.append("le laptop est passé sur batterie pendant le banc")
    return out


# ====================================================================================================== main


def _dir_stats(d: Path) -> dict[str, Any]:
    files = [p for p in d.iterdir() if p.is_file()] if d.is_dir() else []
    return {"count": len(files), "bytes": sum(p.stat().st_size for p in files)}


def _unit_bytes(cfg: Any) -> int:
    """Taille sérialisée (torch.save après `encode`, comme dans un checkpoint) des poids d'une politique."""
    import torch

    from actions import N_ACTIONS
    from models import PolicyNet
    from obs import obs_dim
    from training.checkpoint import encode

    p_kw: dict[str, Any] = {} if cfg.model.policy_hidden is None else {"hidden": cfg.model.policy_hidden}
    with torch.random.fork_rng(devices=[]):
        sd = PolicyNet(obs_dim(cfg.env.n_cars), N_ACTIONS, **p_kw).state_dict()
    buf = io.BytesIO()
    torch.save(encode(sd), buf)
    return len(buf.getvalue())


def provenance(args: argparse.Namespace, cfg: Any, device: Any, train_mod: ModuleType) -> dict[str, Any]:
    import numpy as np
    import torch

    from training.checkpoint import git_commit
    from training.run import rocketgpu_commit

    mem = _win_memory() or {}
    xpu_name = None
    if str(device).startswith("xpu"):
        with contextlib.suppress(Exception):
            xpu_name = str(torch.xpu.get_device_name(0))
    return {
        "command": [sys.executable, *sys.argv],
        "seed": cfg.seed,
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "eval_config": str(args.eval_config),
        "eval_config_sha256": _sha256(args.eval_config),
        "config_hash": cfg.trajectory_hash(),
        "config_full_hash": cfg.full_hash(),
        "git_commit": git_commit(ROOT),
        "git_dirty": bool(train_mod.git_dirty(ROOT)),
        "date_utc_start": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "device_requested": cfg.device,
        "device_resolved": str(device),
        "xpu_name": xpu_name,
        "cpu_name": _cpu_name(),
        "cpu_logical": os.cpu_count(),
        "ram_total": mem.get("total"),
        "power_start": _win_power(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "rocketgpu_commit": rocketgpu_commit(),
        "minutes": args.minutes,
    }


def _write_new(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8") as f:  # mode exclusif : jamais d'écrasement
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "bench_4a.yaml")
    ap.add_argument("--eval-config", type=Path, default=ROOT / "configs" / "bench_4a_eval.yaml")
    ap.add_argument("--minutes", type=float, default=40.0, help="durée murale avant l'arrêt propre (défaut 40)")
    ap.add_argument("--device", help="remplace `device` de la config (défaut : celui de la config, xpu)")
    ap.add_argument("--max-iterations", type=int, help="plafond d'itérations (arrêt `paused`), pour les essais")
    ap.add_argument("--name-prefix", default="bench_4a")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--checkpoints-root", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--sample-s", type=float, default=10.0, help="période d'échantillonnage RAM et alimentation")
    ap.add_argument("--slice-min", type=float, default=5.0, help="largeur des tranches de débit (minutes)")
    ap.add_argument("--allow-dirty", action="store_true", help="accepte un arbre git sale (signalé dans le résumé)")
    ap.add_argument("--checkpoint-probe", action="store_true", help="après le run : checkpoint à 100 instantanés")
    ap.add_argument("--probe-only", type=Path, help="sonde seule sur ce checkpoint, sans run")
    ap.add_argument("--probe-snapshots", type=int, default=TARGET_POOL)
    ap.add_argument("--probe-dir", type=Path, help="dossier de la sonde (défaut : dossier temporaire neuf)")
    ap.add_argument("--keep-probe", action="store_true", help="conserve le fichier sondé")
    args = ap.parse_args(argv)

    import torch

    probe_device = args.device or "cpu"
    if args.probe_only:
        pdir = args.probe_dir or Path(tempfile.mkdtemp(prefix="bench_ckpt_probe_"))
        res = checkpoint_probe(args.probe_only, pdir, args.probe_snapshots, probe_device, args.keep_probe)
        print(json.dumps(res, indent=2))
        return 0

    rg_env = _set_rocketgpu_defaults()
    train_mod = _load_train_script()
    from training import ConfigError, RunConfig, RunError, resolve_device, run
    from training.evalhook import EvalHookConfig

    try:
        cfg = RunConfig.from_yaml(args.config)
        if args.device:
            cfg = dataclasses.replace(cfg, device=args.device)
        hook_cfg = EvalHookConfig.from_yaml(args.eval_config)
    except (ConfigError, ValueError, OSError) as exc:
        print(f"configuration invalide : {exc}", file=sys.stderr)
        return 1
    if cfg.pool is None:
        print("le banc exige une section `pool` (ADR-005 D5)", file=sys.stderr)
        return 1
    if cfg.device.startswith("xpu") and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU indisponible dans cet interpréteur : lancer avec vendor/venv-xpu/Scripts/python", file=sys.stderr)
        return 1
    device = resolve_device(cfg.device)
    prov = provenance(args, cfg, device, train_mod)
    prov["rocketgpu_env"] = rg_env
    if prov["git_dirty"] and not args.allow_dirty:
        print("arbre git sale : mesure non attribuable à un commit (committer, ou --allow-dirty)", file=sys.stderr)
        return 1

    now = datetime.datetime.now(datetime.UTC)
    name = unique_name(
        args.name_prefix, now, lambda n: (args.runs_root / n).exists() or (args.checkpoints_root / n).exists()
    )
    out_dir, ck_dir = args.runs_root / name, args.checkpoints_root / name
    out_dir.mkdir(parents=True, exist_ok=False)
    ck_dir.mkdir(parents=True, exist_ok=False)
    print(f"[bench] {name} : journaux {out_dir}, checkpoints {ck_dir}, {args.minutes} min, device {device}", flush=True)

    t_bench0 = time.perf_counter()
    inst = Instrument(cfg.env.n_envs, hook_cfg.n_envs)
    sampler = Sampler(out_dir / "bench_samples.jsonl", args.sample_s, t_bench0, str(device).startswith("xpu"))
    ticker = Ticker(inst, out_dir / "bench_ticks.jsonl", t_bench0, args.minutes * 60.0)
    status_code = 0
    summary_obj: Any = None
    error: str | None = None
    inst.install()
    sampler.start()
    t_run_start = time.perf_counter() - t_bench0
    try:
        hook = train_mod.build_eval_hook(cfg, hook_cfg, out_dir, ck_dir, resume=False)
        summary_obj = run(
            cfg,
            out_dir,
            max_iterations=args.max_iterations,
            checkpoint_dir=ck_dir,
            command=prov["command"],
            eval_hook=hook,
            should_stop=ticker,
        )
    except (RunError, Exception) as exc:
        error = f"{type(exc).__name__} : {exc}"
        print(f"[bench] run en erreur : {error}", file=sys.stderr, flush=True)
        status_code = 1
    finally:
        t_run_end = time.perf_counter() - t_bench0
        inst.uninstall()
        sampler.stop()
        ticker.close()
    prov["date_utc_end"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    prov["power_end"] = _win_power()

    run_dict: dict[str, Any] = {}
    if summary_obj is not None:
        run_dict = {k: v for k, v in summary_obj.as_dict().items() if k != "steps_per_s"}
        if summary_obj.status not in ("completed", "paused", "interrupted"):
            status_code = 1
    else:
        seg = read_jsonl(out_dir / "segments.jsonl")
        run_dict = {k: v for k, v in (seg[-1] if seg else {}).items() if k != "steps_per_s"}
        run_dict["bench_error"] = error

    probe: dict[str, Any] | None = None
    if args.checkpoint_probe:
        from training.checkpoint import CheckpointManager

        refs = CheckpointManager(ck_dir, keep=10**9).checkpoints()
        if refs:
            pdir = args.probe_dir or Path(tempfile.mkdtemp(prefix="bench_ckpt_probe_"))
            try:
                probe = checkpoint_probe(refs[-1].path, pdir, args.probe_snapshots, str(device), args.keep_probe)
            except Exception as exc:
                probe = {"error": f"{type(exc).__name__} : {exc}"}

    log_sizes = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file() and p.name not in BENCH_FILES}
    summary = summarize_bench(
        name=name,
        provenance=prov,
        run_summary=run_dict,
        ticks=ticker.ticks,
        metrics_rows=read_jsonl(out_dir / "metrics.jsonl"),
        events=read_jsonl(out_dir / "events.jsonl"),
        evals=read_jsonl(out_dir / "evals.jsonl"),
        pool_adds=inst.pool_adds,
        samples=sampler.samples,
        log_sizes=log_sizes,
        eval_snapshots=_dir_stats(ck_dir / "eval_snapshots"),
        learned_per_iter=cfg.batch_learned_steps,
        collected_per_iter=cfg.batch_agent_steps,
        unit_bytes=_unit_bytes(cfg),
        eval_every=hook_cfg.every_timesteps,
        t_run_start=t_run_start,
        t_run_end=t_run_end,
        slice_s=args.slice_min * 60.0,
        probe=probe,
    )
    summary["checkpoint_dir_bytes"] = _dir_stats(ck_dir)
    summary["disk_free_bytes"] = shutil.disk_usage(ck_dir).free
    _write_new(out_dir / "bench_summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    table = render_table(summary)
    _write_new(out_dir / "bench_summary.txt", table)
    print(table)
    print(f"[bench] résumé : {out_dir / 'bench_summary.json'}")
    return status_code


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consoles Windows en cp1252 par défaut
    raise SystemExit(main())
