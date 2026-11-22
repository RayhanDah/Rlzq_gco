"""Lance un run d'entraînement (`training.run`) depuis une config YAML. Script mince : arguments, appel, résumé.

    python scripts/train.py configs/smoke.yaml                  # run neuf (refuse un dossier déjà utilisé)
    python scripts/train.py configs/smoke.yaml --resume         # reprise du dernier checkpoint sain
    python scripts/train.py configs/smoke.yaml --smoke          # smoke test : 1re moitié, arrêt, reprise, 2e moitié
    python scripts/train.py <config> --eval                     # + évaluation en cours de run (S1-léger, ADR-005 D2)
    python scripts/train.py <config> --eval-config eval.yaml    # idem, paramètres de `EvalHookConfig` en YAML

Évaluation en cours de run (`training.evalhook`) : alimente `s1_reached` de `run` via `eval_hook` ; journal
`<out-dir>/evals.jsonl`, état dans les checkpoints (copie dans `evalhook_state.jsonl`), instantanés dans
`<checkpoint-dir>/eval_snapshots/` (hors rotation). Désactivée par défaut (smoke test inchangé).

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
from training.checkpoint import CheckpointIdentity, git_commit  # noqa: E402
from training.evalhook import (  # noqa: E402
    EvalHook,
    EvalHookConfig,
    load_latest_state,
)
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


def smoke_resume_iteration(cfg: RunConfig) -> int:
    """Itération (1-based) où `--smoke` arrête la 1re partie puis reprend : ⌈resume_at_fraction · total_iterations⌉,
    au moins 1. La 1re partie fait donc exactement ce nombre de mises à jour (et ingère autant de lots)."""
    assert cfg.smoke is not None
    return max(1, math.ceil(cfg.smoke.resume_at_fraction * cfg.total_iterations))


def _resolved_device(cfg: RunConfig) -> str:
    from training import resolve_device

    return str(resolve_device(cfg.device))


def _write_new(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "x", encoding="utf-8") as f:  # mode exclusif : jamais d'écrasement
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def build_eval_hook(
    cfg: RunConfig,
    hook_cfg: EvalHookConfig,
    out_dir: Path,
    ck_dir: Path,
    *,
    resume: bool,
    snapshot_dir: Path | None = None,
    sim_factory: Any = None,
    build_obs: Any = None,
    obs_dim: int | None = None,
) -> EvalHook:
    """Hook S1-léger d'un run : politique de même architecture, simulateur RocketGPU dédié (1v1, `n_envs` du hook),
    confirmation par le dernier checkpoint sain notifié par `run`. `resume` : l'état vient du checkpoint repris
    (`extra["evalhook"]`) ; run neuf : refuse un `evalhook_state.jsonl` déjà présent."""
    from actions import N_ACTIONS
    from models import PolicyNet
    from obs import obs_dim as schema_obs_dim
    from rewards import RewardFunction

    if cfg.env.n_cars != 2:
        raise RunError(f"évaluation en cours de run : 1v1 seulement (n_cars={cfg.env.n_cars})")
    reward_cfg = cfg.reward.build()
    d = obs_dim if obs_dim is not None else schema_obs_dim(cfg.env.n_cars)
    identity = CheckpointIdentity.current(
        reward_config_hash=RewardFunction(reward_cfg).config_hash, obs_dim=d, tick_skip=cfg.env.tick_skip
    )
    p_kw: dict[str, Any] = {} if cfg.model.policy_hidden is None else {"hidden": cfg.model.policy_hidden}
    tick_skip = cfg.env.tick_skip

    def default_sim(n_envs: int) -> Any:
        from env import RocketGPUSimulator

        return RocketGPUSimulator(n_envs, 2, tick_skip)

    hook = EvalHook(
        hook_cfg,
        learned_steps_per_iteration=cfg.batch_learned_steps,
        run_seed=cfg.seed,
        out_dir=out_dir,
        snapshot_dir=snapshot_dir or ck_dir / "eval_snapshots",
        policy_factory=lambda: PolicyNet(d, N_ACTIONS, **p_kw),
        sim_factory=sim_factory or default_sim,
        identity=identity,
        reward_config=reward_cfg,
        build_obs=build_obs,
        obs_dim=obs_dim,
        train_config_hash=cfg.trajectory_hash(),
        git_commit_id=git_commit(ROOT),
    )
    # à la reprise, l'état de référence est celui du checkpoint repris (`run` le recharge depuis extra["evalhook"])
    if not resume and load_latest_state(out_dir) is not None:
        raise RunError(f"{out_dir} contient déjà un état d'évaluation : relancer avec --resume")
    return hook


def _eval_summary(hook: EvalHook | None) -> dict[str, Any] | None:
    if hook is None:
        return None
    return {
        "evaluations_retained": len(hook.history),
        "evaluations_this_process": len(hook.eval_seconds),
        "eval_seconds": hook.eval_seconds,
        "streak": hook.streak(),
        "s1_reached": hook.s1_reached,
        "s1_iteration": hook.s1_iteration,
        "s1_eval_iteration": hook.s1_eval_iteration,
        "rollbacks": hook.rollbacks,
    }


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
    ap.add_argument("--eval", action="store_true", help="évaluation en cours de run, paramètres de l'ADR-005 D2")
    ap.add_argument("--eval-config", type=Path, help="YAML de `EvalHookConfig` (implique --eval)")
    ap.add_argument("--eval-snapshot-dir", type=Path, help="instantanés (défaut : <checkpoint-dir>/eval_snapshots)")
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
        hook: EvalHook | None = None
        hook_kw: dict[str, Any] = {}
        if args.eval or args.eval_config:
            try:
                hook_cfg = EvalHookConfig.from_yaml(args.eval_config) if args.eval_config else EvalHookConfig()
            except (ValueError, OSError) as exc:
                print(f"configuration d'évaluation invalide : {exc}", file=sys.stderr)
                return 1
            hook = build_eval_hook(
                cfg, hook_cfg, out_dir, ck_dir, resume=args.resume, snapshot_dir=args.eval_snapshot_dir
            )
            hook_kw = {"eval_hook": hook}
        if not args.smoke:
            summary = run(
                cfg,
                out_dir,
                resume=args.resume,
                max_iterations=args.max_iterations,
                checkpoint_dir=ck_dir,
                command=command,
                **hook_kw,
            )
            out = {k: v for k, v in summary.as_dict().items() if k != "steps_per_s"}
            out["evaluation"] = _eval_summary(hook)
            print(json.dumps(out, indent=2, default=str))
            return 0 if summary.status in ("completed", "paused", "interrupted") else 1

        if cfg.smoke is None:
            print("--smoke exige une section `smoke` dans la config", file=sys.stderr)
            return 1
        summary_path = args.summary or out_dir / "smoke_summary.json"
        if summary_path.exists():
            print(f"{summary_path} existe déjà : jamais écrasé", file=sys.stderr)
            return 1
        half = smoke_resume_iteration(cfg)
        first = run(cfg, out_dir, max_iterations=half, checkpoint_dir=ck_dir, command=command, **hook_kw)
        print(
            f"[smoke] 1re partie : {first.status} à l'itération {first.end_iteration}/{first.total_iterations}",
            flush=True,
        )
        second: RunSummary | None = None
        if first.status == "paused":
            gc.collect()  # libère le simulateur de la 1re partie avant d'en construire un second
            second = run(cfg, out_dir, resume=True, checkpoint_dir=ck_dir, command=command, **hook_kw)
            print(f"[smoke] reprise : {second.status} à l'itération {second.end_iteration}", flush=True)
        report = smoke_report(cfg, first, second, command)
        if hook is not None:
            report["evaluation"] = _eval_summary(hook)
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
