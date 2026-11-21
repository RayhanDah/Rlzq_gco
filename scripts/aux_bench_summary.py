"""Résumé des bancs de la tâche auxiliaire (ADR-010 D5 : choix de λ_aux ; D2 : règle RAM ; D7 : débit, occupation du
fil learner) et vérification du chemin auxiliaire d'un smoke test, à partir des SEULS fichiers bruts des dossiers de
run (lecture seule ; aucun run, aucun simulateur, stdlib + bench_run + sil_bench_summary).

    python scripts/aux_bench_summary.py runs/bench_int_aux_<horodatage> [runs/bench_int_aux_<autre> ...]
    python scripts/aux_bench_summary.py --smoke runs/smoke_aux_<commit>

Banc (fichiers de `scripts/bench_run.py` : `bench_meta.json`, `bench_end.json`, `bench_ticks.jsonl`,
`bench_samples.jsonl`, `metrics.jsonl`, `events.jsonl`, `run_meta.json`), pour chaque dossier :
- **grad_share** (indicateur de D5, C4) : médiane (p10, p90, effectif) sur les lignes de `metrics.jsonl` des
  `--window-min` (10) dernières minutes MURALES ; concluant si les deux compartiments du rejeu sont pleins sur toute la
  fenêtre (`aux/replay_*_size` ≥ `aux.capacity_per_class` de `run_meta.json`), un lot à chaque minibatch de chaque
  ligne (`aux/batch_present` ≥ 1), λ constant, journal ≥ fenêtre, valeurs finies ; disp_share au rapport (diagnostic) ;
- **débit appris** mural (définitions de bench_run), KS4 (< 40k/s sur 30 min : alerte) ;
- **occupation du fil learner** : Σ `learn_s` des ticks (temps passé dans `Learner.update`, instrumenté par bench_run)
  / durée de boucle (dernier `t_bench` − premier), globale et sur la fenêtre ;
- **RAM** (`sil_bench_summary.checkpoint_ram`, `longest_below`) : minimum disponible (échantillons, journal
  `aux/ram_available_mb`, sonde de checkpoint), chute pendant les écritures de checkpoint, KS1.

Règle D5 sur l'ENSEMBLE des bancs concluants (un λ = un banc ; le dernier banc d'un même λ fait foi), pré-enregistrée :
λ = 0,1 d'abord ; médiane > 0,5 -> 0,1/√10 puis 0,01 ; on retient le plus grand λ de médiane ≤ 0,5 ; si 0,01 dépasse
encore 0,5 : PAS DE GO ; si la médiane est < 0,05 à λ = 0,1 : bancs à 0,1·√10 puis 1 (valeur d'UNREAL, plafond), même
règle. L'ADR écrit « 0,03 » et « 0,3 » : ce sont les valeurs arrondies de l'échelle ×/÷ √10 (règle d'ADR-009 D5, que
l'ADR-010 reprend) ; le script propose 0,0316 et 0,316 et compare les λ mesurés à 10 % près (deux échelons voisins
diffèrent d'un facteur 3,16) : un banc à 0,03 ou 0,3 compte pour son échelon, et le λ RETENU est la valeur réellement
mesurée. Le λ retenu avec une médiane < 0,05 est noté (critère de révision de l'ADR, Conséquences).
Règle D2 (RAM) : minimum disponible < 1,5 Go à H = 150 -> H = 100 et nouveau banc ; encore < 1,5 Go à H = 100 -> pas
de GO.

Smoke (`metrics.jsonl`, `run_meta*.json`, `smoke_summary.json`) : verdict du smoke, lots auxiliaires appliqués (avant et
après la reprise), perte non nulle, rejeu exactement conservé à la reprise (taille_k = min(capacité, taille_k−1 +
ajouts_k) pour chaque compartiment, à chaque itération), valeurs `aux/*` finies, RAM journalisée.

Sortie : tableau sur stdout et JSON `aux_summary.json` (ou `aux_smoke_check.json`) dans le PREMIER dossier, jamais
écrasé (suffixe `_001`...), sauf `--no-write`. Code de sortie : 0 si le résumé est calculé (banc) ou si le smoke passe ;
1 sinon ; 2 si un dossier est illisible.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import bench_run as br  # noqa: E402
import sil_bench_summary as sbs  # noqa: E402

WINDOW_S = 600.0
"""ADR-010 D5 : médiane de grad_share sur les 10 dernières minutes du banc."""
SHARE_MAX = 0.5
"""ADR-010 D5 (seuil de C4) : au-delà, λ ← λ / √10."""
SHARE_LOW = 0.05
"""ADR-010 D5 : sous ce seuil à λ = 0,1, bancs à 0,1·√10 puis 1 ; au λ retenu : critère de révision."""
LAMBDA_START, LAMBDA_FLOOR, LAMBDA_CEIL = 0.1, 0.01, 1.0
DOWN = (0.1 / math.sqrt(10.0), 0.01)
"""ADR-010 D5 : descente 0,1 -> 0,0316 (« 0,03 ») -> 0,01."""
UP = (0.1 * math.sqrt(10.0), 1.0)
"""ADR-010 D5 : montée 0,1 -> 0,316 (« 0,3 ») -> 1 (UNREAL, annexe B)."""
REL_TOL = 0.1
RAM_RULE_MB = 1500.0
"""ADR-010 D2 : minimum disponible < 1,5 Go -> H = 100 et nouveau banc ; encore < 1,5 Go à H = 100 -> pas de GO."""
H_FALLBACK = 100
KS1_MB, KS1_S = 1000.0, 600.0
"""ADR-010 KS1 : mémoire disponible < 1 Go pendant > 10 min."""
KS4_LEARNED, KS4_S = 40_000.0, 1800.0
"""ADR-010 KS4 : débit appris < 40k pas/s sur 30 min (alerte au rapport)."""


# ====================================================================================================== partie pure


def _f(x: Any) -> float | None:
    return br.num(x)


def aux_window(
    rows: Sequence[Mapping[str, Any]], *, window_s: float = WINDOW_S, capacity: int | None
) -> dict[str, Any]:
    """Statistiques `aux/*` sur la fenêtre finale et conditions de validité du calibrage D5."""
    aux_rows = [r for r in rows if "aux/grad_share" in r]
    win, covered, total = sbs.window_rows(aux_rows, window_s)

    def col(rs: Sequence[Mapping[str, Any]], key: str) -> list[float]:
        return [v for v in (_f(r.get(key)) for r in rs) if v is not None]

    share = col(win, "aux/grad_share")
    pos, zero = col(win, "aux/replay_pos_size"), col(win, "aux/replay_zero_size")
    present = col(win, "aux/batch_present")
    coefs = sorted(set(col(win, "aux/coef")))
    nonfinite = sum(1 for r in win if _f(r.get("aux/grad_share")) is None)
    reasons: list[str] = []
    if not aux_rows:
        reasons.append("aucune ligne `aux/*` dans metrics.jsonl (tâche auxiliaire désactivée ?)")
    if total < window_s:
        reasons.append(f"journal de {total / 60:.1f} min < fenêtre de {window_s / 60:.0f} min")
    if capacity is None:
        reasons.append("capacité du rejeu inconnue (run_meta.json) : rejeu plein non vérifiable")
    else:
        for name, sizes in (("±", pos), ("0", zero)):
            if not sizes or min(sizes) < capacity:
                reasons.append(
                    f"compartiment {name} non plein sur la fenêtre (min {min(sizes, default=None)} < {capacity})"
                )
    if not present or min(present) < 1:
        reasons.append("itération(s) sans lot auxiliaire dans la fenêtre (grad_share = 0 par construction)")
    if len(coefs) != 1:
        reasons.append(f"λ non constant sur la fenêtre : {coefs}")
    if nonfinite:
        reasons.append(f"{nonfinite} grad_share non fini(s)")
    return {
        "window_s": window_s,
        "window_covered_s": covered,
        "log_span_s": total,
        "rows": len(win),
        "coef": coefs[0] if len(coefs) == 1 else coefs,
        "grad_share": br.dist(share),
        "grad_share_all_rows": br.dist(col(aux_rows, "aux/grad_share")),
        "disp_share": br.dist(col(win, "aux/disp_share")),
        "loss": br.dist(col(win, "aux/loss")),
        "recall_pos": br.dist(col(win, "aux/recall_pos")),
        "recall_neg": br.dist(col(win, "aux/recall_neg")),
        "recall_zero": br.dist(col(win, "aux/recall_zero")),
        "frac_pos": br.dist(col(win, "aux/frac_pos")),
        "frac_neg": br.dist(col(win, "aux/frac_neg")),
        "frac_masked": br.dist(col(win, "aux/frac_masked")),
        "replay_pos_size": br.dist(pos),
        "replay_zero_size": br.dist(zero),
        "replay_pos_age_mean": br.dist(col(win, "aux/replay_pos_age_mean")),
        "capacity_per_class": capacity,
        "conclusive": not reasons,
        "not_conclusive_because": reasons,
    }


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL)


def _benched(results: Mapping[float, float], coef: float) -> float | None:
    """λ réellement mesuré correspondant à l'échelon `coef` (à `REL_TOL` près), ou None."""
    return next((c for c in results if _same(c, coef)), None)


def _median_at(results: Mapping[float, float], coef: float) -> float | None:
    c = _benched(results, coef)
    return None if c is None else results[c]


def choose_lambda(results: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Règle D5 appliquée telle quelle à des bancs concluants (λ, médiane de grad_share), dans l'ordre des bancs (un
    λ rejoué : le dernier fait foi). Rend la décision : `BANC` (λ suivant à mesurer), `RETENIR` (λ retenu) ou
    `PAS DE GO`, l'action et une note éventuelle."""
    res: dict[float, float] = {}
    for c, m in results:
        res = {k: v for k, v in res.items() if not _same(k, c)}
        res[c] = m
    out: dict[str, Any] = {"rule": "ADR-010 D5", "benches": [{"coef": c, "median": m} for c, m in res.items()]}

    def bench(c: float, why: str) -> dict[str, Any]:
        return {**out, "decision": "BANC", "next_coef": c, "retained_coef": None, "action": why, "note": None}

    def keep(step: float, why: str) -> dict[str, Any]:
        measured = _benched(res, step)
        c = step if measured is None else measured
        m = res.get(c)
        note = None
        if m is not None and m < SHARE_LOW:
            note = f"médiane {m:.3g} < {SHARE_LOW} au λ retenu : critère de révision de l'ADR-010 (Conséquences)"
        return {**out, "decision": "RETENIR", "next_coef": None, "retained_coef": c, "action": why, "note": note}

    m0 = _median_at(res, LAMBDA_START)
    if m0 is None:
        return bench(LAMBDA_START, "premier banc à λ = 0,1")
    if m0 > SHARE_MAX:
        for c in DOWN:
            m = _median_at(res, c)
            if m is None:
                return bench(c, f"médiane > {SHARE_MAX} au λ précédent : λ ← λ / √10 = {c:.4g}, nouveau banc")
            if m <= SHARE_MAX:
                return keep(c, f"médiane {m:.3g} ≤ {SHARE_MAX} : λ = {c:.4g} retenu (plus grand λ conforme)")
        return {
            **out,
            "decision": "PAS DE GO",
            "next_coef": None,
            "retained_coef": None,
            "action": f"λ = 0,01 dépasse encore {SHARE_MAX} : révision de l'ADR-010",
            "note": None,
        }
    if m0 >= SHARE_LOW:
        return keep(LAMBDA_START, f"médiane {m0:.3g} dans [{SHARE_LOW}, {SHARE_MAX}] : λ = 0,1 retenu")
    best = LAMBDA_START
    for c in UP:
        m = _median_at(res, c)
        if m is None:
            return bench(c, f"médiane < {SHARE_LOW} à λ = 0,1 : banc à λ = {c:.4g} (montée ×√10, plafond 1)")
        if m > SHARE_MAX:
            return keep(best, f"médiane {m:.3g} > {SHARE_MAX} à λ = {c:.4g} : λ = {best:.4g} retenu")
        best = c
    return keep(best, f"médiane ≤ {SHARE_MAX} jusqu'au plafond : λ = {best:.4g} retenu (plafond d'UNREAL)")


def ram_rule(horizon: int | None, avail_min_mb: float | None) -> dict[str, Any]:
    """Règle D2 (RAM) pour un banc : minimum < 1,5 Go à H = 150 -> H = 100 ; encore < 1,5 Go à H = 100 -> pas de GO."""
    out: dict[str, Any] = {"rule": "ADR-010 D2", "horizon_steps": horizon, "avail_min_mb": avail_min_mb}
    if avail_min_mb is None or horizon is None:
        return {**out, "decision": "NON CONCLUANT", "action": "minimum de RAM ou horizon inconnu"}
    if avail_min_mb >= RAM_RULE_MB:
        return {**out, "decision": "OK", "action": f"minimum {avail_min_mb:.0f} Mo ≥ 1,5 Go : H = {horizon} tenu"}
    if horizon > H_FALLBACK:
        return {**out, "decision": "H = 100", "action": "minimum < 1,5 Go : H = 100 et nouveau banc (repli de D2)"}
    return {**out, "decision": "PAS DE GO", "action": f"minimum < 1,5 Go à H = {horizon} : pas de GO (D2)"}


def learner_occupancy(ticks: Sequence[Mapping[str, Any]], *, window_s: float = WINDOW_S) -> dict[str, Any]:
    """Part du temps de boucle passée dans `Learner.update` (fil learner), globale et sur la fenêtre finale : Σ
    `learn_s` / (dernier `t_bench` − premier). Les ticks d'apprentissage en recouvrement chevauchent la collecte."""
    ts = [t for t in ticks if _f(t.get("t_bench")) is not None]
    if len(ts) < 2:
        return {"global": None, "window": None, "loop_s": None}

    def frac(rs: Sequence[Mapping[str, Any]]) -> float | None:
        span = float(rs[-1]["t_bench"]) - float(rs[0]["t_bench"])
        busy = sum(_f(r.get("learn_s")) or 0.0 for r in rs[1:])  # deltas : le 1er tick couvre l'assemblage
        return busy / span if span > 0 else None

    last = float(ts[-1]["t_bench"])
    tail = [t for t in ts if float(t["t_bench"]) >= last - window_s]
    return {
        "global": frac(ts),
        "window": frac(tail) if len(tail) >= 2 else None,
        "loop_s": last - float(ts[0]["t_bench"]),
        "definition": "Σ learn_s des ticks (Learner.update instrumenté par bench_run) / durée de boucle",
    }


def ram_summary(
    samples: Sequence[Mapping[str, Any]], metrics_rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    pts = sbs._samples_mb(samples)
    logged = [v for v in (_f(r.get("aux/ram_available_mb")) for r in metrics_rows) if v is not None]
    ck = sbs.checkpoint_ram(events, samples)
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
    longest = sbs.longest_below(pts, KS1_MB)
    return {
        "unit": "Mo (2^20 octets)",
        "avail_min_mb_samples": avail_min,
        "avail_min_mb_metrics": min(logged, default=None),
        "avail_min_mb": overall,
        "longest_below_1gb_s": longest,
        "ks1_triggered": longest > KS1_S,
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
    """Débits (`sil_bench_summary.throughput`, définitions de bench_run) ; règle KS4 d'ADR-010 (même seuil)."""
    out = sbs.throughput(meta, end, ticks, metrics_rows, window_s=window_s)
    out["ks4"]["rule"] = "ADR-010 KS4 : débit appris mural < 40k/s sur 30 min -> alerte au rapport (K6 à 25k inchangé)"
    return out


def smoke_checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    capacity: int,
    resume_starts: Sequence[int],
    smoke_verdict: str | None,
) -> dict[str, Any]:
    """Vérifications du chemin auxiliaire d'un smoke test (lignes de `metrics.jsonl`, dernière occurrence par
    itération)."""
    by_step: dict[int, Mapping[str, Any]] = {}
    for r in rows:
        s = br.integer(r.get("step"))
        if s is not None and "aux/coef" in r:
            by_step[s] = r
    steps = sorted(by_step)

    def v(s: int, key: str) -> float:
        x = _f(by_step[s].get(key))
        return math.nan if x is None else x

    applied = [s for s in steps if v(s, "aux/batch_present") > 0]
    after_resume = [s for s in applied if resume_starts and s > min(resume_starts)]
    broken: list[int] = []
    for a, b in itertools.pairwise(steps):
        if b != a + 1 or v(b, "aux/released") > 0:
            continue
        for comp in ("pos", "zero"):
            size = f"aux/replay_{comp}_size"
            if v(b, size) != min(capacity, v(a, size) + v(b, f"aux/replay_{comp}_new")):
                broken.append(b)
                break
    nonfinite = sorted({k for s in steps for k, x in by_step[s].items() if k.startswith("aux/") and _f(x) is None})
    resume_ok = [
        s
        for s in resume_starts
        if s in by_step and s + 1 in by_step and s + 1 not in broken and v(s, "aux/replay_zero_size") > 0
    ]
    checks = {
        "smoke_pass": smoke_verdict == "PASS",
        "aux_logged": bool(steps),
        "aux_batches_applied": bool(applied),
        "aux_loss_nonzero": any(v(s, "aux/loss") != 0.0 for s in applied),
        "aux_batches_after_resume": bool(after_resume),
        "replay_conserved_each_iteration": bool(steps) and not broken,
        "replay_nonempty_at_resume": bool(resume_starts) and len(resume_ok) == len(resume_starts),
        "aux_values_finite": not nonfinite,
        "ram_logged": bool(steps) and all(_f(by_step[s].get("aux/ram_available_mb")) is not None for s in steps),
    }
    return {
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "iterations": len(steps),
        "iterations_with_aux_batches": applied,
        "resume_starts": list(resume_starts),
        "replay_pos_size_by_iteration": {s: v(s, "aux/replay_pos_size") for s in steps},
        "grad_share_by_iteration": {s: v(s, "aux/grad_share") for s in steps},
        "replay_breaks": broken,
        "nonfinite_keys": nonfinite,
    }


# ====================================================================================================== lecture


def _aux_cfg(metas: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for m in metas:
        aux = (m.get("config") or {}).get("aux")
        if isinstance(aux, Mapping):
            return aux
    return {}


def summarize_bench_dir(d: Path, *, window_s: float = WINDOW_S, capacity: int | None = None) -> dict[str, Any]:
    meta = br._read_json(d / br.BENCH_META)
    if not meta:
        raise FileNotFoundError(f"{br.BENCH_META} absent de {d} : pas un dossier de banc")
    end = br._read_json(d / br.BENCH_END)
    metas = sbs._run_metas(d)
    aux = _aux_cfg(metas)
    cap = capacity if capacity is not None else br.integer(aux.get("capacity_per_class"))
    metrics_rows = br.read_jsonl(d / "metrics.jsonl")
    events = br.read_jsonl(d / "events.jsonl")
    ticks = br.read_jsonl(d / "bench_ticks.jsonl")
    win = aux_window(metrics_rows, window_s=window_s, capacity=cap)
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
    if not aux.get("enabled"):
        warnings.append("config du run sans tâche auxiliaire activée (run_meta.json)")
    if aux.get("anneal_end_timesteps") is not None:
        warnings.append("calendrier de λ présent : la règle D5 suppose λ constant pendant le banc")
    ram = ram_summary(br.read_jsonl(d / "bench_samples.jsonl"), metrics_rows, events)
    warnings += ram["warnings"]
    return {
        "label": "résumé d'un banc D5 de la tâche auxiliaire (ADR-010), calculé hors ligne depuis les fichiers bruts",
        "dir": str(d),
        "provenance": prov,
        "aux_config": dict(aux),
        "aux": win,
        "throughput": throughput(meta, end, ticks, metrics_rows, window_s=window_s),
        "learner_occupancy": learner_occupancy(ticks, window_s=window_s),
        "ram": ram,
        "ram_rule": ram_rule(br.integer(aux.get("horizon_steps")), ram["avail_min_mb"]),
        "checkpoint_probe": br._read_json(d / br.BENCH_PROBE) or None,
        "warnings": warnings,
    }


def summarize_benches(dirs: Sequence[Path], *, window_s: float = WINDOW_S) -> dict[str, Any]:
    benches = [summarize_bench_dir(d, window_s=window_s) for d in dirs]
    results = [
        (float(b["aux"]["coef"]), float(b["aux"]["grad_share"]["median"]))
        for b in benches
        if b["aux"]["conclusive"]
        and isinstance(b["aux"]["coef"], float)
        and b["aux"]["grad_share"]["median"] is not None
    ]
    return {
        "label": "bancs D5 de la tâche auxiliaire (ADR-010) : choix de λ, débit, occupation du learner, RAM",
        "computed_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "benches": benches,
        "ignored_not_conclusive": [b["dir"] for b in benches if not b["aux"]["conclusive"]],
        "lambda": choose_lambda(results),
    }


def smoke_dir_checks(d: Path) -> dict[str, Any]:
    metas = sbs._run_metas(d)
    if not metas:
        raise FileNotFoundError(f"run_meta.json absent de {d} : pas un dossier de run")
    aux = _aux_cfg(metas)
    cap = br.integer(aux.get("capacity_per_class"))
    if cap is None:
        raise ValueError("config sans section `aux` (run_meta.json) : pas un smoke de la tâche auxiliaire")
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
        "label": "vérification du chemin de la tâche auxiliaire d'un smoke test (ADR-010 D8)",
        "dir": str(d),
        "computed_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "git_commit": metas[0].get("git_commit"),
        "aux_config": dict(aux),
        **res,
    }


# ====================================================================================================== rendu


def _n(x: Any, fmt: str = ".3g") -> str:
    return sbs._n(x, fmt)


def render_benches(s: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for b in s["benches"]:
        w, t, r, occ = b["aux"], b["throughput"], b["ram"], b["learner_occupancy"]
        gs = w["grad_share"]
        lines += [
            f"Banc {b['dir']} (commit {b['provenance'].get('git_commit')}, sale={b['provenance'].get('git_dirty')})",
            f"  grad_share, {w['window_s'] / 60:.0f} dernières min ({w['rows']} itérations, λ {w['coef']}) : médiane "
            f"{_n(gs['median'])} (p10 {_n(gs['p10'])}, p90 {_n(gs['p90'])}) ; disp_share médiane "
            f"{_n(w['disp_share']['median'])} ; rappel ± {_n(w['recall_pos']['median'])} / "
            f"{_n(w['recall_neg']['median'])}, 0 {_n(w['recall_zero']['median'])}",
            f"  rejeu : ± min {_n(w['replay_pos_size']['min'], '.0f')}, "
            f"0 min {_n(w['replay_zero_size']['min'], '.0f')} "
            f"/ {w['capacity_per_class']} ; frac ± {_n(w['frac_pos']['median'])} + {_n(w['frac_neg']['median'])}, "
            f"masqués {_n(w['frac_masked']['median'])}",
        ]
        if w["not_conclusive_because"]:
            lines.append("  non concluant : " + " ; ".join(w["not_conclusive_because"]))
        g, tw, k4 = t["global"], t["window"], t["ks4"]
        lines.append(
            f"  débit appris mural : global {_n(g['learned_wall'], '.0f')}/s, "
            f"fenêtre {_n(tw['learned_wall'], '.0f')}/s ;"
            f" KS4 {'ALERTE' if k4['alert'] else 'non'} ({'concluant' if k4['conclusive'] else 'non concluant'}) ; "
            f"fil learner occupé {_n(occ['global'])} (fenêtre {_n(occ['window'])})"
        )
        ck = r["checkpoints"]
        lines.append(
            f"  RAM disponible min {_n(r['avail_min_mb'], '.0f')} Mo ; pendant les checkpoints "
            f"{_n(ck['avail_min_mb'], '.0f')} Mo, chute max {_n(ck['drop_max_mb'], '.0f')} Mo ; KS1 "
            f"{r['ks1_triggered']} ; règle D2 : {b['ram_rule']['decision']} — {b['ram_rule']['action']}"
        )
        lines += [f"  ! {x}" for x in b["warnings"]]
    lam = s["lambda"]
    lines.append(f"Règle D5 : {lam['decision']} — {lam['action']}" + (f" ({lam['note']})" if lam.get("note") else ""))
    return "\n".join(lines) + "\n"


def render_smoke(s: Mapping[str, Any]) -> str:
    lines = [f"Smoke aux {s['dir']} (commit {s['git_commit']}) : {s['verdict']}"]
    lines += [f"  {'ok ' if ok else 'NON'} {k}" for k, ok in s["checks"].items()]
    lines.append(f"  itérations avec lots : {s['iterations_with_aux_batches']} ; reprises : {s['resume_starts']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+", help="dossiers de bancs (ou un dossier de smoke avec --smoke)")
    ap.add_argument("--smoke", action="store_true", help="vérifie le chemin de la tâche auxiliaire d'un smoke test")
    ap.add_argument("--window-min", type=float, default=WINDOW_S / 60.0, help="fenêtre finale (minutes, défaut 10)")
    ap.add_argument("--no-write", action="store_true", help="n'écrit pas le JSON dans le dossier")
    args = ap.parse_args(argv)
    if args.smoke and len(args.dirs) != 1:
        ap.error("--smoke : un seul dossier")
    try:
        if args.smoke:
            s = smoke_dir_checks(args.dirs[0])
            text, name = render_smoke(s), "aux_smoke_check.json"
        else:
            s = summarize_benches(args.dirs, window_s=args.window_min * 60.0)
            text, name = render_benches(s), "aux_summary.json"
    except (OSError, ValueError, KeyError) as exc:
        print(f"lecture impossible : {type(exc).__name__} : {exc}", file=sys.stderr)
        return 2
    print(text, end="")
    if not args.no_write:
        path = br.write_unique(args.dirs[0] / name, br._dump(s))
        print(f"[aux] résumé : {path}")
    if args.smoke:
        return 0 if s["verdict"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consoles Windows en cp1252 par défaut
    raise SystemExit(main())
