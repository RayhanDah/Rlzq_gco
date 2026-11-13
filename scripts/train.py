"""Lance un run d'entraînement (`training.run`) depuis une config YAML. Script mince : arguments, appel, résumé.

    python scripts/train.py configs/smoke.yaml                  # run neuf (refuse un dossier déjà utilisé)
    python scripts/train.py configs/smoke.yaml --resume         # reprise du dernier checkpoint sain
    python scripts/train.py configs/smoke.yaml --smoke          # smoke test : 1re moitié, arrêt, reprise, 2e moitié

Sorties (jamais supprimées ni écrasées) : journaux dans `runs/<name>/` (`--out-dir`), checkpoints dans
`checkpoints/<name>/` (`--checkpoint-dir`). Variables d'environnement RocketGPU : `ROCKETGPU_PY_DIR`,
`ROCKETGPU_COLLISION_MESH_DIR` (et `ROCKETGPU_DLL_DIR` si besoin). XPU : lancer avec `vendor/venv-xpu/Scripts/python`
(le device `auto` choisit cuda, sinon xpu, sinon cpu). Ctrl+C : l'itération en cours finit, checkpoint final.

Code de sortie : 0 si le run se termine (budget atteint ou pause demandée) ou si le smoke test passe ; 1 sinon.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import gc
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from training import ConfigError, RunConfig, RunError, RunSummary, run  # noqa: E402
from training.checkpoint import git_commit  # noqa: E402
from training.run import rocketgpu_commit  # noqa: E402


def _stats(xs: list[float]) -> dict[str, float | int | None]:
    ok = [x for x in xs if math.isfinite(x) and x > 0]
    if not ok:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(ok),
        "mean": statistics.fmean(ok),
        "median": statistics.median(ok),
        "min": min(ok),
        "max": max(ok),
    }


def git_dirty(repo: Path) -> bool:
    """Vrai si l'arbre de travail diffère du commit, ou si git est indisponible (propreté non prouvable)."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return res.returncode != 0 or bool(res.stdout.strip())


def smoke_report(cfg: RunConfig, first: RunSummary, second: RunSummary | None, command: list[str]) -> dict[str, Any]:
    """Résumé JSON du smoke test et verdict (critères de `docs/PLAN.md` Phase 3 et de la section `smoke`)."""
    assert cfg.smoke is not None
    segments = [s for s in (first, second) if s is not None]
    ckpts = [c for s in segments for c in s.checkpoints]
    decisions = [d for s in segments for d in s.decisions]
    nan = second.nan_detected if second is not None else first.nan_detected  # cumulé dans les checkpoints
    last = second if second is not None else first
    steady_first = first.steps_per_s[1:]  # la 1re itération d'un run neuf collecte deux lots (démarrage)
    wall = sum(s.wall_s for s in segments)
    loop = sum(s.loop_s for s in segments)
    collected_in_segments = sum(s.segment_collected_timesteps for s in segments)
    checks = {
        "completed_budget": last.status == "completed",
        "timesteps_at_least_budget": last.trained_timesteps >= cfg.budget.total_timesteps,
        "no_nan": nan <= cfg.smoke.max_nan,
        "no_guard_stop": all(d["action"] != "stop" for d in decisions),
        "no_restore": all(s.restores == 0 for s in segments),
        "resumed_mid_run": (
            second is not None
            and second.resumed_from is not None
            and 0 < second.start_iteration < last.total_iterations
            and second.start_iteration == first.end_iteration
        ),
        "resume_exact_env": second is not None and second.exact_env is True,
        "checkpoints_written": len(ckpts) >= 2,
    }
    required = [
        "completed_budget",
        "timesteps_at_least_budget",
        "no_nan",
        "no_guard_stop",
        "no_restore",
        "checkpoints_written",
    ]
    if cfg.smoke.require_resume:
        required += ["resumed_mid_run", "resume_exact_env"]
    return {
        "command": command,
        "config": cfg.name,
        "config_hash": cfg.trajectory_hash(),
        "git_commit": git_commit(ROOT),
        "git_dirty": git_dirty(ROOT),
        "date_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "rocketgpu_commit": rocketgpu_commit(),
        "device": cfg.device,
        "device_resolved": _resolved_device(cfg),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "iterations_total": last.end_iteration,
        "iterations_planned": last.total_iterations,
        "trained_timesteps": last.trained_timesteps,
        "collected_timesteps": last.collected_timesteps,
        "duration_wall_s": wall,
        "duration_loop_s": loop,
        "mean_steps_per_s_loop": (collected_in_segments / loop) if loop > 0 else None,
        "mean_steps_per_s_wall": (collected_in_segments / wall) if wall > 0 else None,
        "steps_per_s_by_slice": {
            "first_half_steady": _stats(steady_first),
            "second_half": _stats(second.steps_per_s if second is not None else []),
        },
        "nan_detected": nan,
        "restores": last.restores,
        "guard_decisions": decisions,
        "checkpoints": {
            "count": len(ckpts),
            "bytes_max": max((c["bytes"] for c in ckpts), default=None),
            "write_s": _stats([c["write_s"] for c in ckpts]),
            "capture_s": _stats([c["capture_s"] for c in ckpts]),
            "files": ckpts,
        },
        "goals_total": last.goals_total,
        "p_kick_final": last.p_kick_final,
        "segments": [dataclasses.replace(s, steps_per_s=[]).as_dict() for s in segments],
        "checks": checks,
        "required_checks": required,
        "verdict": "PASS" if all(checks[k] for k in required) else "FAIL",
    }


def _resolved_device(cfg: RunConfig) -> str:
    from training import resolve_device

    return str(resolve_device(cfg.device))


def _write_new(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "x", encoding="utf-8") as f:  # mode exclusif : jamais d'écrasement
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="fichier YAML (voir configs/smoke.yaml)")
    ap.add_argument("--out-dir", type=Path, help="journaux (défaut : runs/<name>)")
    ap.add_argument("--checkpoint-dir", type=Path, help="checkpoints (défaut : checkpoints/<name>)")
    ap.add_argument("--resume", action="store_true", help="reprendre du dernier checkpoint sain")
    ap.add_argument("--max-iterations", type=int, help="plafond absolu du compteur d'itérations (arrêt `paused`)")
    ap.add_argument("--device", help="remplace `device` de la config (change le hash de config)")
    ap.add_argument(
        "--smoke", action="store_true", help="smoke test : 1re moitié, arrêt, reprise, 2e moitié, résumé JSON"
    )
    ap.add_argument(
        "--summary", type=Path, help="fichier de résumé JSON du smoke test (défaut : <out-dir>/smoke_summary.json)"
    )
    args = ap.parse_args(argv)
    command = [sys.executable, *sys.argv]

    try:
        cfg = RunConfig.from_yaml(args.config)
        if args.device:
            cfg = dataclasses.replace(cfg, device=args.device)
    except (ConfigError, OSError) as exc:
        print(f"configuration invalide : {exc}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or ROOT / "runs" / cfg.name
    ck_dir = args.checkpoint_dir or ROOT / "checkpoints" / cfg.name

    try:
        if not args.smoke:
            summary = run(
                cfg,
                out_dir,
                resume=args.resume,
                max_iterations=args.max_iterations,
                checkpoint_dir=ck_dir,
                command=command,
            )
            print(json.dumps({k: v for k, v in summary.as_dict().items() if k != "steps_per_s"}, indent=2, default=str))
            return 0 if summary.status in ("completed", "paused", "interrupted") else 1

        if cfg.smoke is None:
            print("--smoke exige une section `smoke` dans la config", file=sys.stderr)
            return 1
        summary_path = args.summary or out_dir / "smoke_summary.json"
        if summary_path.exists():
            print(f"{summary_path} existe déjà : jamais écrasé", file=sys.stderr)
            return 1
        half = max(1, math.ceil(cfg.smoke.resume_at_fraction * cfg.total_iterations))
        first = run(cfg, out_dir, max_iterations=half, checkpoint_dir=ck_dir, command=command)
        print(
            f"[smoke] 1re partie : {first.status} à l'itération {first.end_iteration}/{first.total_iterations}",
            flush=True,
        )
        second: RunSummary | None = None
        if first.status == "paused":
            gc.collect()  # libère le simulateur de la 1re partie avant d'en construire un second
            second = run(cfg, out_dir, resume=True, checkpoint_dir=ck_dir, command=command)
            print(f"[smoke] reprise : {second.status} à l'itération {second.end_iteration}", flush=True)
        report = smoke_report(cfg, first, second, command)
        _write_new(summary_path, report)
        print(
            json.dumps({k: report[k] for k in ("verdict", "checks", "nan_detected", "mean_steps_per_s_loop")}, indent=2)
        )
        print(f"[smoke] résumé : {summary_path}")
        return 0 if report["verdict"] == "PASS" else 1
    except RunError as exc:
        print(f"run impossible : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consoles Windows en cp1252 par défaut
    raise SystemExit(main())
