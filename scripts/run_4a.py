"""Lanceur des runs courts 4a (ADR-005, tâche A6) : une session d'un run, reprise automatique entre sessions.

    vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant std --seed 0

Une session = un appel : le run `run_4a_<variante>_s<seed>` est créé s'il n'existe pas, repris du dernier checkpoint
sain sinon. La session s'arrête proprement (checkpoint final) à la fin du budget, après `--max-hours` heures de mur, ou
sur Ctrl+C ; relancer la même commande reprend le run. Chaque session est un segment (`segments.jsonl`). Un run dont
la dernière session a fini par un STOP ou une erreur n'est pas repris sans `--resume-after-stop` ; un run terminé
n'est pas relancé.

Ce que le lanceur ajoute à `scripts/train.py` :
- seed et nom du run fixés par la ligne de commande (configs `configs/run_4a_{std,s}.yaml`, identiques hors variante) ;
- évaluation en cours de run toujours active (`configs/run_4a_eval.yaml`, ADR-005 D2) ;
- arrêt à durée murale (`--max-hours`, vérifié à chaque frontière d'itération) ;
- mise en veille par inactivité bloquée pendant la session (Windows, `SetThreadExecutionState` : rien n'est modifié
  dans les réglages du système ; un capot fermé met quand même la machine en veille) ;
- refus d'un arbre git sale (mesure non attribuable à un commit) et de GCO-S sans `--std-s1-observed` (ADR-005 :
  GCO-S seulement si au moins une seed GCO-std a S1-léger).

Code de sortie : 0 = session terminée proprement (budget atteint, durée atteinte ou Ctrl+C) ; 3 = STOP d'un kill
criterion (K2, K3, K5 : le plan est suspendu, ADR-005 D6) ; 1 = erreur ou refus.
Aucun run de plus de 30 min sans smoke test vert sur le même commit (CLAUDE.md, règle 5) : le lanceur ne le vérifie pas.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

VARIANTS = {"std": "run_4a_std.yaml", "s": "run_4a_s.yaml"}
EVAL_CONFIG = "run_4a_eval.yaml"
EXIT_OK, EXIT_ERROR, EXIT_KILL = 0, 1, 3
_CKPT_RE = re.compile(r"^ckpt_\d{6}_it\d{8}\.pt$")  # `training.checkpoint`

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def run_name(variant: str, seed: int) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variante {variant!r} : {sorted(VARIANTS)} attendu")
    if seed < 0:
        raise ValueError(f"seed={seed} : >= 0 attendu")
    return f"run_4a_{variant}_s{seed}"


def has_checkpoint(ck_dir: Path) -> bool:
    return ck_dir.is_dir() and any(_CKPT_RE.match(p.name) for p in ck_dir.iterdir())


def session_mode(out_dir: Path, ck_dir: Path) -> str:
    """« new » (rien n'existe), « resume » (au moins un checkpoint) ; lève si des journaux existent sans checkpoint
    (run cassé avant son checkpoint 0 : à examiner à la main, rien n'est écrasé ni supprimé)."""
    if has_checkpoint(ck_dir):
        return "resume"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"{out_dir} contient des journaux mais {ck_dir} n'a aucun checkpoint : état à examiner")
    return "new"


class SessionRefused(RuntimeError):
    """La session ne doit pas démarrer (décision humaine requise)."""


def last_segment(out_dir: Path) -> dict[str, Any] | None:
    """Dernière ligne lisible de `segments.jsonl` (une ligne par session, écrite par `training.run`)."""
    path = out_dir / "segments.jsonl"
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(ValueError):
            row = json.loads(line)
            if isinstance(row, dict):
                last = row
    return last


def check_resumable(out_dir: Path, *, force: bool) -> str:
    """État d'un run existant d'après sa dernière session : « resumable », ou « completed » (budget atteint : rien à
    relancer). Lève `SessionRefused` si elle s'est terminée par un STOP de kill criterion ou une erreur : `run`
    reprendrait le dernier checkpoint SAIN, antérieur au STOP, et continuerait, alors que l'ADR-005 D6 dit « STOP =
    fin du run ». `force` (`--resume-after-stop`) lève ce refus : décision humaine, à consigner."""
    seg = last_segment(out_dir)
    status = None if seg is None else seg.get("status")
    if status == "completed":
        return "completed"
    if status in ("stopped", "error") and not force:
        raise SessionRefused(
            f"dernière session de {out_dir.name} : {status} ({seg.get('cause') if seg else None}). Reprise refusée "
            "(ADR-005 D6) ; après décision, relancer avec --resume-after-stop"
        )
    return "resumable"


def deadline(max_hours: float | None, clock: Callable[[], float] = time.monotonic) -> Callable[[], bool] | None:
    """`should_stop` de `training.run` : vrai dès que `max_hours` heures de mur sont écoulées depuis cet appel."""
    if max_hours is None:
        return None
    if not max_hours > 0.0:
        raise ValueError(f"max_hours={max_hours} : > 0 attendu")
    end = clock() + max_hours * 3600.0
    return lambda: clock() >= end


def exit_code(status: str) -> int:
    if status in ("completed", "paused", "interrupted"):
        return EXIT_OK
    return EXIT_KILL if status == "stopped" else EXIT_ERROR


@contextlib.contextmanager
def keep_awake(enabled: bool = True) -> Iterator[bool]:
    """Bloque la veille par inactivité le temps du bloc (Windows). Rend True si la demande a été acceptée."""
    if not enabled or os.name != "nt":
        yield False
        return
    import ctypes

    k32 = ctypes.WinDLL("kernel32")
    k32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    k32.SetThreadExecutionState.restype = ctypes.c_uint
    ok = bool(k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    try:
        yield ok
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)


def launch(
    variant: str,
    seed: int,
    *,
    max_hours: float | None,
    runs_root: Path,
    checkpoints_root: Path,
    config_dir: Path,
    device: str | None = None,
    max_iterations: int | None = None,
    command: list[str] | None = None,
    run_kwargs: dict[str, Any] | None = None,
    hook_kwargs: dict[str, Any] | None = None,
    config_edit: Callable[[Any], Any] | None = None,
    resume_after_stop: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Une session du run `run_name(variant, seed)`. `run_kwargs`, `hook_kwargs`, `config_edit` : tests seulement
    (env et simulateur factices, config réduite)."""
    import train as train_script

    from training import RunConfig, run
    from training.evalhook import EvalHookConfig

    name = run_name(variant, seed)
    cfg = RunConfig.from_yaml(config_dir / VARIANTS[variant])
    cfg = dataclasses.replace(cfg, name=name, seed=seed)
    if device:
        cfg = dataclasses.replace(cfg, device=device)
    if config_edit is not None:
        cfg = config_edit(cfg)
    hook_cfg = EvalHookConfig.from_yaml(config_dir / EVAL_CONFIG)
    out_dir, ck_dir = runs_root / name, checkpoints_root / name
    mode = session_mode(out_dir, ck_dir)
    resume = mode == "resume"
    if resume and check_resumable(out_dir, force=resume_after_stop) == "completed":
        # budget déjà atteint : ne rien reconstruire, ne rien réécrire (un checkpoint final non sain ferait sinon
        # rejouer les dernières itérations depuis un checkpoint antérieur)
        return EXIT_OK, {"run": name, "session_mode": "none", "status": "completed", "remaining_iterations": 0}
    hook = train_script.build_eval_hook(cfg, hook_cfg, out_dir, ck_dir, resume=resume, **(hook_kwargs or {}))
    print(
        f"[run_4a] {name} : session {'de reprise' if resume else 'neuve'}, budget {cfg.total_iterations} itérations, "
        f"durée max {max_hours} h, journaux {out_dir}, checkpoints {ck_dir}",
        flush=True,
    )
    summary = run(
        cfg,
        out_dir,
        resume=resume,
        max_iterations=max_iterations,
        checkpoint_dir=ck_dir,
        command=command,
        eval_hook=hook,
        should_stop=deadline(max_hours),
        **(run_kwargs or {}),
    )
    report = {k: v for k, v in summary.as_dict().items() if k != "steps_per_s"}
    report["run"] = name
    report["session_mode"] = mode
    report["evaluation"] = train_script._eval_summary(hook)
    report["remaining_iterations"] = summary.total_iterations - summary.end_iteration
    return exit_code(summary.status), report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True, help="std = GCO-std, s = GCO-S")
    ap.add_argument("--seed", type=int, required=True, help="0, 1 ou 2 (ADR-005 D3)")
    ap.add_argument("--max-hours", type=float, default=7.5, help="durée murale de la session (défaut 7,5 h)")
    ap.add_argument("--device", help="remplace `device` de la config (change le hash : même valeur à chaque session)")
    ap.add_argument("--max-iterations", type=int, help="plafond absolu du compteur d'itérations (essais)")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--checkpoints-root", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--allow-dirty", action="store_true", help="accepte un arbre git sale (essais seulement)")
    ap.add_argument("--no-keep-awake", action="store_true", help="ne bloque pas la mise en veille")
    ap.add_argument(
        "--std-s1-observed",
        action="store_true",
        help="exigé pour --variant s : atteste qu'au moins une seed GCO-std a S1-léger (ADR-005)",
    )
    ap.add_argument(
        "--resume-after-stop",
        action="store_true",
        help="reprend un run dont la dernière session a fini par un STOP ou une erreur (décision humaine, ADR-005 D6)",
    )
    args = ap.parse_args(argv)

    if args.variant == "s" and not args.std_s1_observed:
        print(
            "GCO-S refusé : ADR-005 ne l'autorise que si au moins une seed GCO-std a S1-léger (--std-s1-observed)",
            file=sys.stderr,
        )
        return EXIT_ERROR

    import bench_run
    import train as train_script

    from training import ConfigError, RunError

    if train_script.git_dirty(ROOT) and not args.allow_dirty:
        print("arbre git sale : run non attribuable à un commit (committer, ou --allow-dirty)", file=sys.stderr)
        return EXIT_ERROR
    bench_run._set_rocketgpu_defaults()
    try:
        with keep_awake(not args.no_keep_awake) as awake:
            print(f"[run_4a] veille par inactivité bloquée : {awake}", flush=True)
            code, report = launch(
                args.variant,
                args.seed,
                max_hours=args.max_hours,
                runs_root=args.runs_root,
                checkpoints_root=args.checkpoints_root,
                config_dir=ROOT / "configs",
                device=args.device,
                max_iterations=args.max_iterations,
                command=[sys.executable, *sys.argv],
                resume_after_stop=args.resume_after_stop,
            )
    except (ConfigError, RunError, RuntimeError, ValueError, OSError) as exc:
        print(f"run impossible : {type(exc).__name__} : {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(report, indent=2, default=str))
    left = report["remaining_iterations"]
    if code == EXIT_KILL:
        print(f"[run_4a] STOP ({report.get('cause')}) : plan suspendu, ne pas relancer sans décision (ADR-005 D6)")
    elif left > 0:
        print(f"[run_4a] session finie ({report['status']}), {left} itérations restantes : relancer la même commande")
    else:
        print("[run_4a] budget atteint : run terminé")
    return code


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consoles Windows en cp1252 par défaut
    raise SystemExit(main())
