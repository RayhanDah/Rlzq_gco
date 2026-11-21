"""Résumé SIL d'un banc (ADR-009 D7, calibrage de c_sil par la règle D5) et vérification SIL d'un smoke test, à partir
des SEULS fichiers bruts d'un dossier de run (lecture seule ; aucun run, aucun simulateur, stdlib + bench_run).

    python scripts/sil_bench_summary.py runs/bench_int_sil_<horodatage>      # banc : disp_share, débit, RAM, D5
    python scripts/sil_bench_summary.py --smoke runs/smoke_sil_<commit>      # smoke : le chemin SIL a bien tourné

`scripts/bench_run.py --summarize` ne rapporte ni `sil/*`, ni la RAM autour des checkpoints : ce script le complète
(il réutilise les définitions de débit de bench_run : `build_iter_records`, `global_rates`).

Banc (fichiers : `bench_meta.json`, `bench_end.json`, `bench_ticks.jsonl`, `bench_samples.jsonl`, `metrics.jsonl`,
`events.jsonl`, `run_meta.json`) :
- **disp_share** (indicateur C4, D5) : médiane (p10, p90, effectif) sur les lignes de `metrics.jsonl` des `--window-min`
  (10) dernières minutes MURALES (champ `t`, unix) ; buffer plein exigé sur toute la fenêtre (`sil/buffer_size` ≥
  `sil.capacity` de `run_meta.json`), au moins un pas SIL par ligne, fenêtre réellement couverte ; sinon NON CONCLUANT ;
- **décision D5**, mécanique : médiane > 0,5 -> c_sil / √10 et nouveau banc (0,1 -> 0,0316 -> 0,01) ; c_sil ≤ 0,01 et
  médiane > 0,5 -> pas de GO, révision de l'ADR ; médiane ≤ 0,5 -> c_sil retenu (si < 0,05 à c_sil = 0,1 : noté,
  run maintenu, critère de révision de l'ADR) ;
- **débit appris** (mural, définitions de bench_run) : global et sur la fenêtre ; KS4 (< 40k/s sur 30 min) ;
- **RAM disponible** (`bench_samples.jsonl`, `avail` en octets, rapportée en Mo de 2^20 octets comme
  `training.memory`) : minimum global, durée maximale continue sous 1 Go (KS1), et pour chaque checkpoint le minimum
  dans [début − marge, fin + marge] (début = fin − capture − écriture, `events.jsonl`) contre la médiane de la
  minute qui précède : pic transitoire mesuré. Lancer le banc avec `--sample-s 1` (le défaut de 10 s manque des
  écritures de quelques secondes : signalé). Règle D7 : minimum < 1,5 Go -> fenêtre de 150 pas.

Smoke (`metrics.jsonl`, `run_meta*.json`, `smoke_summary.json`) : verdict du smoke, pas SIL appliqués (avant et après
la reprise), perte SIL non nulle, buffer exactement conservé à la reprise (taille_k = min(capacité, taille_k−1 +
ajouts_k) à chaque itération), libération en fin d'annealing, valeurs `sil/*` finies, RAM journalisée.

Sortie : tableau sur stdout et JSON `sil_summary.json` (ou `sil_smoke_check.json`) dans le dossier, jamais écrasé
(suffixe `_001`...), sauf `--no-write`. Code de sortie : 0 si le résumé est calculé (banc) ou si le smoke SIL passe ;
1 sinon ; 2 si le dossier est illisible.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import bench_run as br  # noqa: E402

MB = 1024 * 1024
WINDOW_S = 600.0
"""ADR-009 D5 : médiane de disp_share sur les 10 dernières minutes du banc."""
SHARE_MAX = 0.5
"""ADR-009 D5 : au-delà, c_sil ← c_sil / √10."""
SHARE_LOW = 0.05
"""ADR-009 D5 et Conséquences : sous ce seuil à c_sil = 0,1, noté (critère de révision), run maintenu."""
COEF_TOP, COEF_FLOOR = 0.1, 0.01
"""ADR-009 D5 : échelle 0,1 -> 0,1/√10 -> 0,01 ; si 0,01 dépasse encore 0,5 : pas de GO."""
RAM_WINDOW150_MB = 1500.0
"""ADR-009 D7 : mémoire disponible minimale sous 1,5 Go -> fenêtre de 150 pas (D1)."""
KS1_MB, KS1_S = 1000.0, 600.0
"""ADR-009 KS1 : mémoire disponible < 1 Go pendant > 10 min."""
KS4_LEARNED, KS4_S = 40_000.0, 1800.0
"""ADR-009 KS4 : débit appris < 40k pas/s sur 30 min."""
CKPT_MARGIN_S, CKPT_BASELINE_S = 2.0, 60.0


# ====================================================================================================== partie pure


def _f(x: Any) -> float | None:
    return br.num(x)


def _median(xs: Sequence[float]) -> float | None:
    return statistics.median(xs) if xs else None


def window_rows(rows: Sequence[Mapping[str, Any]], window_s: float) -> tuple[list[Mapping[str, Any]], float, float]:
    """Lignes dont `t` (unix) est dans les `window_s` dernières secondes ; rend aussi la durée couverte par la fenêtre
    (t_max − t du premier retenu) et la durée totale du journal."""
    timed = [r for r in rows if _f(r.get("t")) is not None]
    if not timed:
        return [], 0.0, 0.0
    ts = [float(r["t"]) for r in timed]
    t_max, t_min = max(ts), min(ts)
    kept = [r for r in timed if float(r["t"]) >= t_max - window_s]
    covered = t_max - min(float(r["t"]) for r in kept) if kept else 0.0
    return kept, covered, t_max - t_min


def sil_window(
    rows: Sequence[Mapping[str, Any]], *, window_s: float = WINDOW_S, capacity: int | None
) -> dict[str, Any]:
    """Statistiques `sil/*` sur la fenêtre finale et conditions de validité du calibrage D5."""
    sil_rows = [r for r in rows if "sil/disp_share" in r]
    win, covered, total = window_rows(sil_rows, window_s)

    def col(rs: Sequence[Mapping[str, Any]], key: str) -> list[float]:
        return [v for v in (_f(r.get(key)) for r in rs) if v is not None]

    share = col(win, "sil/disp_share")
    sizes = col(win, "sil/buffer_size")
    steps = col(win, "sil/steps_done")
    coefs = sorted(set(col(win, "sil/coef")))
    nonfinite = sum(1 for r in win if _f(r.get("sil/disp_share")) is None)
    full_t = next(
        (
            float(r["t"])
            for r in sil_rows
            if capacity is not None and (_f(r.get("sil/buffer_size")) or 0.0) >= capacity and _f(r.get("t"))
        ),
        None,
    )
    t0 = min((float(r["t"]) for r in sil_rows if _f(r.get("t")) is not None), default=None)
    reasons: list[str] = []
    if not sil_rows:
        reasons.append("aucune ligne `sil/*` dans metrics.jsonl (SIL désactivée ?)")
    if total < window_s:
        reasons.append(f"journal de {total / 60:.1f} min < fenêtre de {window_s / 60:.0f} min")
    if capacity is None:
        reasons.append("capacité du buffer inconnue (run_meta.json) : buffer plein non vérifiable")
    elif not sizes or min(sizes) < capacity:
        reasons.append(f"buffer non plein sur la fenêtre (min {min(sizes) if sizes else None} < {capacity})")
    if not steps or min(steps) < 1:
        reasons.append("itération(s) sans pas SIL dans la fenêtre (disp_share = 0 par construction)")
    if len(coefs) != 1:
        reasons.append(f"c_sil non constant sur la fenêtre : {coefs}")
    if nonfinite:
        reasons.append(f"{nonfinite} disp_share non fini(s)")
    return {
        "window_s": window_s,
        "window_covered_s": covered,
        "log_span_s": total,
        "rows": len(win),
        "coef": coefs[0] if len(coefs) == 1 else coefs,
        "disp_share": br.dist(share),
        "disp_share_all_rows": br.dist(col(sil_rows, "sil/disp_share")),
        "grad_share": br.dist(col(win, "sil/grad_share")),
        "frac_pos": br.dist(col(win, "sil/frac_pos")),
        "steps_done": br.dist(steps),
        "buffer_size": br.dist(sizes),
        "buffer_capacity": capacity,
        "buffer_full_after_s": (full_t - t0) if full_t is not None and t0 is not None else None,
        "buffer_generic_frac": br.dist(col(win, "sil/buffer_generic_frac")),
        "buffer_age_mean": br.dist(col(win, "sil/buffer_age_mean")),
        "episodes_added": br.dist(col(win, "sil/episodes_added")),
        "conclusive": not reasons,
        "not_conclusive_because": reasons,
    }


def calibration(coef: float | None, median: float | None, conclusive: bool) -> dict[str, Any]:
    """Règle D5 de l'ADR-009, appliquée telle quelle (pré-enregistrée)."""
    out: dict[str, Any] = {"rule": "ADR-009 D5", "coef": coef, "median_disp_share": median, "next_coef": None}
    if not conclusive or coef is None or median is None:
        out.update(decision="NON CONCLUANT", action="banc à refaire ou à prolonger (voir not_conclusive_because)")
        return out
    if median > SHARE_MAX:
        if coef <= COEF_FLOOR * (1 + 1e-9):
            out.update(decision="PAS DE GO", action="c_sil = 0,01 dépasse encore 0,5 : révision de l'ADR-009")
            return out
        nxt = coef / math.sqrt(10.0)
        if math.isclose(nxt, COEF_FLOOR, rel_tol=1e-6):
            nxt = COEF_FLOOR
        out.update(
            decision="RÉDUIRE",
            next_coef=nxt,
            action=f"médiane {median:.3g} > {SHARE_MAX} : c_sil ← {coef:g} / √10 = {nxt:.6g}, nouveau banc",
        )
        return out
    note = None
    if median < SHARE_LOW and math.isclose(coef, COEF_TOP, rel_tol=1e-9):
        note = f"médiane {median:.3g} < {SHARE_LOW} à c_sil = 0,1 : noté, run maintenu (critère de révision de l'ADR)"
    out.update(decision="RETENIR", action=f"médiane {median:.3g} ≤ {SHARE_MAX} : c_sil = {coef:g} retenu", note=note)
    return out


def _samples_mb(samples: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    out = []
    for s in samples:
        u, a = _f(s.get("unix")), _f(s.get("avail"))
        if u is not None and a is not None:
            out.append((u, a / MB))
    return sorted(out)


def longest_below(points: Sequence[tuple[float, float]], threshold_mb: float) -> float:
    """Plus longue durée continue (entre échantillons consécutifs tous deux sous le seuil) sous `threshold_mb`."""
    best = cur = 0.0
    for (t0, v0), (t1, v1) in itertools.pairwise(points):
        cur = cur + (t1 - t0) if v0 < threshold_mb and v1 < threshold_mb else 0.0
        best = max(best, cur)
    return best


def checkpoint_ram(
    events: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    margin_s: float = CKPT_MARGIN_S,
    baseline_s: float = CKPT_BASELINE_S,
) -> dict[str, Any]:
    """RAM disponible autour de chaque écriture de checkpoint (événement `checkpoint` : `t` = fin, unix)."""
    pts = _samples_mb(samples)
    periods = [b[0] - a[0] for a, b in itertools.pairwise(pts)]
    rows = []
    for e in events:
        if e.get("event") != "checkpoint":
            continue
        end = _f(e.get("t"))
        if end is None:
            continue
        dur = (_f(e.get("capture_s")) or 0.0) + (_f(e.get("write_s")) or 0.0)
        lo, hi = end - dur - margin_s, end + margin_s
        inside = [v for t, v in pts if lo <= t <= hi]
        before = [v for t, v in pts if lo - baseline_s <= t < lo]
        base = _median(before)
        low = min(inside) if inside else None
        rows.append(
            {
                "iteration": br.integer(e.get("iteration")),
                "bytes": br.integer(e.get("bytes")),
                "capture_s": _f(e.get("capture_s")),
                "write_s": _f(e.get("write_s")),
                "samples_in_window": len(inside),
                "avail_min_mb": low,
                "avail_baseline_mb": base,
                "drop_mb": (base - low) if base is not None and low is not None else None,
            }
        )
    lows = [r["avail_min_mb"] for r in rows if r["avail_min_mb"] is not None]
    drops = [r["drop_mb"] for r in rows if r["drop_mb"] is not None]
    return {
        "rows": rows,
        "count": len(rows),
        "avail_min_mb": min(lows) if lows else None,
        "drop_max_mb": max(drops) if drops else None,
        "drop_median_mb": _median(drops),
        "sample_period_median_s": _median(periods),
        "undersampled": [r["iteration"] for r in rows if r["samples_in_window"] < 2],
    }


def ram_summary(
    samples: Sequence[Mapping[str, Any]], metrics_rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    pts = _samples_mb(samples)
    priv = [v / MB for v in (_f(s.get("private")) for s in samples) if v is not None]
    logged = [v for v in (_f(r.get("sil/ram_available_mb")) for r in metrics_rows) if v is not None]
    ck = checkpoint_ram(events, samples)
    avail_min = min((v for _, v in pts), default=None)
    overall = min(
        [x for x in (avail_min, min(logged, default=None), ck["avail_min_mb"]) if x is not None], default=None
    )
    warnings = []
    if ck["undersampled"]:
        warnings.append(
            f"checkpoints {ck['undersampled']} : moins de 2 échantillons pendant l'écriture (période médiane "
            f"{ck['sample_period_median_s']} s) : pic possiblement manqué, relancer le banc avec --sample-s 1"
        )
    if not pts:
        warnings.append("aucun échantillon de RAM (bench_samples.jsonl vide : hors Windows ?)")
    return {
        "unit": "Mo (2^20 octets)",
        "avail_min_mb_samples": avail_min,
        "avail_min_mb_metrics": min(logged, default=None),
        "avail_min_mb": overall,
        "private_max_mb": max(priv, default=None),
        "longest_below_1gb_s": longest_below(pts, KS1_MB),
        "ks1_triggered": longest_below(pts, KS1_MB) > KS1_S,
        "below_1_5gb": overall is not None and overall < RAM_WINDOW150_MB,
        "d7_rule": "minimum < 1,5 Go -> fenêtre de 150 pas (ADR-009 D1, D7)",
        "checkpoints": ck,
        "warnings": warnings,
    }


def throughput(
    meta: Mapping[str, Any],
    end: Mapping[str, Any],
    ticks: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    *,
    window_s: float = WINDOW_S,
) -> dict[str, Any]:
    """Débits appris et collectés (définitions de `bench_run`), global et sur la fenêtre finale ; KS4."""
    ticks = [t for t in ticks if all(_f(t.get(k)) is not None for k in ("t", "gap_s"))]
    metrics = {int(r["step"]): r for r in metrics_rows if br.integer(r.get("step")) is not None}
    start = br.integer((end.get("run") or {}).get("start_iteration")) or 0
    records = br.build_iter_records(
        ticks,
        metrics,
        start_iteration=start,
        learned_per_iter=int(meta["learned_per_iter"]),
        collected_per_iter=int(meta["collected_per_iter"]),
    )
    glob = br.global_rates(records)
    last = records[-1].t_end if records else 0.0
    tail = br.global_rates([r for r in records if r.t_end >= last - window_s])
    wall = float(glob["wall_s"] or 0.0)
    lw = _f(glob["learned_wall"])
    return {
        "global": glob,
        "window": tail,
        "ks4": {
            "threshold": KS4_LEARNED,
            "conclusive": wall >= 0.98 * KS4_S,
            "alert": lw is not None and wall >= 0.98 * KS4_S and lw < KS4_LEARNED,
            "rule": "ADR-009 KS4 : débit appris mural < 40k/s sur 30 min -> alerte au rapport",
        },
    }


def smoke_checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    capacity: int,
    resume_starts: Sequence[int],
    smoke_verdict: str | None,
) -> dict[str, Any]:
    """Vérifications du chemin SIL d'un smoke test (lignes de `metrics.jsonl`, dernière occurrence par itération)."""
    by_step: dict[int, Mapping[str, Any]] = {}
    for r in rows:
        s = br.integer(r.get("step"))
        if s is not None and "sil/coef" in r:
            by_step[s] = r
    steps = sorted(by_step)

    def v(s: int, key: str) -> float:
        x = _f(by_step[s].get(key))
        return math.nan if x is None else x

    applied = [s for s in steps if v(s, "sil/steps_done") > 0]
    after_resume = [s for s in applied if resume_starts and s > min(resume_starts)]
    broken: list[int] = []
    for a, b in itertools.pairwise(steps):
        if b != a + 1 or v(b, "sil/released") > 0:
            continue
        if v(b, "sil/buffer_size") != min(capacity, v(a, "sil/buffer_size") + v(b, "sil/buffer_new")):
            broken.append(b)
    nonfinite = sorted({k for s in steps for k, x in by_step[s].items() if k.startswith("sil/") and _f(x) is None})
    resume_ok = [
        s
        for s in resume_starts
        if s in by_step and s + 1 in by_step and s + 1 not in broken and v(s, "sil/buffer_size") > 0
    ]
    last = steps[-1] if steps else None
    checks = {
        "smoke_pass": smoke_verdict == "PASS",
        "sil_logged": bool(steps),
        "sil_steps_applied": bool(applied),
        "sil_loss_nonzero": any(v(s, "sil/loss_policy") != 0.0 for s in applied),
        "sil_steps_after_resume": bool(after_resume),
        "buffer_conserved_each_iteration": bool(steps) and not broken,
        "buffer_nonempty_at_resume": bool(resume_starts) and len(resume_ok) == len(resume_starts),
        "released_at_end": last is not None and v(last, "sil/released") == 1.0 and v(last, "sil/coef") == 0.0,
        "sil_values_finite": not nonfinite,
        "ram_logged": bool(steps) and all(_f(by_step[s].get("sil/ram_available_mb")) is not None for s in steps),
    }
    return {
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "iterations": len(steps),
        "iterations_with_sil_steps": applied,
        "resume_starts": list(resume_starts),
        "buffer_size_by_iteration": {s: v(s, "sil/buffer_size") for s in steps},
        "coef_by_iteration": {s: v(s, "sil/coef") for s in steps},
        "disp_share_by_iteration": {s: v(s, "sil/disp_share") for s in steps},
        "buffer_breaks": broken,
        "nonfinite_keys": nonfinite,
    }


# ====================================================================================================== lecture


def _run_metas(d: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(d.glob("run_meta*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _sil_cfg(metas: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for m in metas:
        sil = (m.get("config") or {}).get("sil")
        if isinstance(sil, Mapping):
            return sil
    return {}


def summarize_bench_dir(d: Path, *, window_s: float = WINDOW_S, capacity: int | None = None) -> dict[str, Any]:
    meta = br._read_json(d / br.BENCH_META)
    if not meta:
        raise FileNotFoundError(f"{br.BENCH_META} absent de {d} : pas un dossier de banc")
    end = br._read_json(d / br.BENCH_END)
    metas = _run_metas(d)
    sil = _sil_cfg(metas)
    cap = capacity if capacity is not None else br.integer(sil.get("capacity"))
    metrics_rows = br.read_jsonl(d / "metrics.jsonl")
    events = br.read_jsonl(d / "events.jsonl")
    win = sil_window(metrics_rows, window_s=window_s, capacity=cap)
    coef = win["coef"] if isinstance(win["coef"], float) else None
    warnings: list[str] = []
    prov = {**dict(meta.get("provenance") or {}), **dict(end.get("provenance_end") or {})}
    if prov.get("git_dirty"):
        warnings.append("arbre git sale : mesure non attribuable à un commit")
    if not end:
        warnings.append(f"{br.BENCH_END} absent : banc interrompu (résumé partiel)")
    if end.get("error"):
        warnings.append(f"run en erreur : {end['error']}")
    restores = [e for e in events if e.get("event") == "restore"]
    if restores:
        warnings.append(f"{len(restores)} restauration(s) pendant le banc")
    if not sil.get("enabled"):
        warnings.append("config du run sans SIL activée (run_meta.json)")
    ram = ram_summary(br.read_jsonl(d / "bench_samples.jsonl"), metrics_rows, events)
    warnings += ram["warnings"]
    return {
        "label": "résumé SIL du banc D7 (ADR-009), calculé hors ligne depuis les fichiers bruts",
        "dir": str(d),
        "computed_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "provenance": prov,
        "sil_config": dict(sil),
        "sil": win,
        "calibration": calibration(coef, win["disp_share"]["median"], win["conclusive"]),
        "throughput": throughput(meta, end, br.read_jsonl(d / "bench_ticks.jsonl"), metrics_rows, window_s=window_s),
        "ram": ram,
        "checkpoint_probe": br._read_json(d / br.BENCH_PROBE) or None,
        "warnings": warnings,
    }


def smoke_dir_checks(d: Path) -> dict[str, Any]:
    metas = _run_metas(d)
    if not metas:
        raise FileNotFoundError(f"run_meta.json absent de {d} : pas un dossier de run")
    sil = _sil_cfg(metas)
    cap = br.integer(sil.get("capacity"))
    if cap is None:
        raise ValueError("config sans section `sil` (run_meta.json) : pas un smoke SIL")
    summary = br._read_json(d / "smoke_summary.json")
    resume_starts = [
        int(m["start_iteration"]) for m in metas if m.get("resume") and br.integer(m.get("start_iteration"))
    ]
    res = smoke_checks(
        br.read_jsonl(d / "metrics.jsonl"),
        capacity=cap,
        resume_starts=sorted(resume_starts),
        smoke_verdict=summary.get("verdict") if summary else None,
    )
    return {
        "label": "vérification du chemin SIL d'un smoke test (ADR-009 D7)",
        "dir": str(d),
        "computed_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "git_commit": metas[0].get("git_commit"),
        "sil_config": dict(sil),
        **res,
    }


# ====================================================================================================== rendu


def _n(x: Any, fmt: str = ".3g") -> str:
    return "—" if x is None else format(x, fmt) if isinstance(x, int | float) else str(x)


def render_bench(s: Mapping[str, Any]) -> str:
    w, c, t, r = s["sil"], s["calibration"], s["throughput"], s["ram"]
    ds = w["disp_share"]
    lines = [
        f"Banc SIL {s['dir']} (commit {s['provenance'].get('git_commit')}, sale={s['provenance'].get('git_dirty')})",
        f"disp_share, {w['window_s'] / 60:.0f} dernières min ({w['rows']} itérations, c_sil {w['coef']}) : médiane "
        f"{_n(ds['median'])} (p10 {_n(ds['p10'])}, p90 {_n(ds['p90'])}) ; grad_share médiane "
        f"{_n(w['grad_share']['median'])} ; frac_pos médiane {_n(w['frac_pos']['median'])}",
        f"buffer : min {_n(w['buffer_size']['min'], '.0f')} / {w['buffer_capacity']} sur la fenêtre, plein après "
        f"{_n(w['buffer_full_after_s'], '.0f')} s ; part générique médiane {_n(w['buffer_generic_frac']['median'])}",
        f"Décision D5 : {c['decision']} — {c['action']}" + (f" ({c['note']})" if c.get("note") else ""),
    ]
    if w["not_conclusive_because"]:
        lines.append("  non concluant : " + " ; ".join(w["not_conclusive_because"]))
    g, tw, k4 = t["global"], t["window"], t["ks4"]
    lines.append(
        f"Débit appris mural : global {_n(g['learned_wall'], '.0f')}/s sur {_n(g['wall_s'], '.0f')} s ; fenêtre "
        f"{_n(tw['learned_wall'], '.0f')}/s ; KS4 {'ALERTE' if k4['alert'] else 'non'} "
        f"({'concluant' if k4['conclusive'] else 'non concluant : < 30 min'})"
    )
    ck = r["checkpoints"]
    lines.append(
        f"RAM disponible min {_n(r['avail_min_mb'], '.0f')} Mo (échantillons {_n(r['avail_min_mb_samples'], '.0f')}, "
        f"journal {_n(r['avail_min_mb_metrics'], '.0f')}) ; pendant les checkpoints "
        f"{_n(ck['avail_min_mb'], '.0f')} Mo, "
        f"chute max {_n(ck['drop_max_mb'], '.0f')} Mo ({ck['count']} checkpoints) ; < 1,5 Go : {r['below_1_5gb']} ; "
        f"KS1 : {r['ks1_triggered']} (plus long < 1 Go : {r['longest_below_1gb_s']:.0f} s)"
    )
    lines += [f"! {x}" for x in s["warnings"]]
    return "\n".join(lines) + "\n"


def render_smoke(s: Mapping[str, Any]) -> str:
    lines = [f"Smoke SIL {s['dir']} (commit {s['git_commit']}) : {s['verdict']}"]
    lines += [f"  {'ok ' if ok else 'NON'} {k}" for k, ok in s["checks"].items()]
    lines.append(f"  itérations avec pas SIL : {s['iterations_with_sil_steps']} ; reprises : {s['resume_starts']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="dossier du banc (runs/bench_int_sil_...) ou du smoke (avec --smoke)")
    ap.add_argument("--smoke", action="store_true", help="vérifie le chemin SIL d'un smoke test")
    ap.add_argument("--window-min", type=float, default=WINDOW_S / 60.0, help="fenêtre finale (minutes, défaut 10)")
    ap.add_argument("--capacity", type=int, help="capacité du buffer (défaut : run_meta.json)")
    ap.add_argument("--no-write", action="store_true", help="n'écrit pas le JSON dans le dossier")
    args = ap.parse_args(argv)
    try:
        if args.smoke:
            s = smoke_dir_checks(args.dir)
            text, name = render_smoke(s), "sil_smoke_check.json"
        else:
            s = summarize_bench_dir(args.dir, window_s=args.window_min * 60.0, capacity=args.capacity)
            text, name = render_bench(s), "sil_summary.json"
    except (OSError, ValueError, KeyError) as exc:
        print(f"lecture impossible : {type(exc).__name__} : {exc}", file=sys.stderr)
        return 2
    print(text, end="")
    if not args.no_write:
        path = br.write_unique(args.dir / name, br._dump(s))
        print(f"[sil] résumé : {path}")
    if args.smoke:
        return 0 if s["verdict"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consoles Windows en cp1252 par défaut
    raise SystemExit(main())
