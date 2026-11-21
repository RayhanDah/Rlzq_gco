"""Lanceur des runs courts 4a (ADR-005, tâche A6) : une session d'un run, reprise automatique entre sessions.

    vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant std --seed 0

Une session = un appel : le run `run_4a_<variante>_s<seed>` est créé s'il n'existe pas, repris du dernier checkpoint
sain sinon. La session s'arrête proprement (checkpoint final) à la fin du budget, après `--max-hours` heures de mur, ou
sur Ctrl+C ; relancer la même commande reprend le run. Chaque session est un segment (`segments.jsonl`). Un run dont
la dernière session a fini par un STOP ou une erreur n'est pas repris sans `--resume-after-stop` ; un run terminé
n'est pas relancé.

Ce que le lanceur ajoute à `scripts/train.py` :
- seed et nom du run fixés par la ligne de commande (configs `configs/run_4a_{std,s}.yaml`, identiques hors variante) ;
- évaluation en cours de run toujours active (`configs/run_4a_eval.yaml`, ADR-005 D2 ; `--eval-config
  run_s1v2_eval.yaml` pour le protocole S1-seed v2 de l'ADR-006) ;
- arrêt à durée murale (`--max-hours`, vérifié à chaque frontière d'itération) ;
- mise en veille par inactivité bloquée pendant la session (Windows, `SetThreadExecutionState` : rien n'est modifié
  dans les réglages du système ; un capot fermé met quand même la machine en veille) ;
- refus d'un arbre git sale (mesure non attribuable à un commit) et de GCO-S sans `--std-s1-observed` (ADR-005 :
  GCO-S seulement si au moins une seed GCO-std a S1-léger).

Interventions (ADR-006 D5, ADR-007, ADR-008, ADR-009) : `--variant int_gamma998`, `int_slow_anneal` ou `int_sil`
(self-imitation, bloc `sil`) lance la config `configs/<variante>.yaml` (GCO-std, une seule modification par rapport à
`run_4a_std.yaml`) sous le nom de run
`<variante>_s<seed>`, avec les mêmes garanties (sessions enfants, reprise après perte du GPU, refus après STOP, arbre
propre). Évaluation par défaut : `run_s1v2_eval.yaml` (protocole d'ADR-006). Pas de garde-fou GCO-S : des GCO-std.

Témoin HORS SPEC (ADR-011) : `--variant diag_dense` lance `configs/diag_dense.yaml`
(récompense `dense-diag`, seul levier
contre std s0) sous le nom `diag_dense_s<seed>`, mêmes garanties de session ; run étiqueté « HORS SPEC » dans run_meta,
events et segments, jamais promu (ses checkpoints ne servent qu'à l'évaluation, `training.checkpoint.require_in_spec`).
Audit des réglages PPO (ADR-012 D1, D2) : `--variant diag_dense_lr3` lance `configs/diag_dense_lr3.yaml` (le même
témoin, seul levier : calendrier de LR ×3) sous le nom `diag_dense_lr3_s<seed>`, même traitement que `diag_dense`
(HORS SPEC, éval `run_s1v2_eval.yaml`). Arrêt à 0,5B : `--max-iterations 9688` (le budget de la config reste 1B pour
que la LR à l'itération t soit exactement 3× celle du témoin).

Enchaînement et reprise automatique (décision de l'utilisateur du 2026-11-15, runs lancés la nuit) :
`--seed 1 2` enchaîne les runs ; le suivant ne démarre que si le précédent est terminé (budget atteint). Chaque
session tourne dans un processus enfant (`--child`) : après une perte du GPU par le pilote
(`UR_RESULT_ERROR_DEVICE_LOST`, contexte XPU inutilisable dans le processus), le superviseur attend 60 s et relance
un enfant qui reprend du dernier checkpoint sain, sans limite tant que le run progresse ; il abandonne après
`MAX_NO_PROGRESS` pannes de suite sans nouveau checkpoint. Un STOP de kill criterion ou toute autre erreur arrête
l'enchaînement (ADR-005 D6). `--max-hours` borne chaque run (0 = aucune limite).

Code de sortie : 0 = terminé proprement (budget atteint, durée atteinte ou Ctrl+C) ; 3 = STOP d'un kill criterion
(K2, K3, K5 : le plan est suspendu, ADR-005 D6) ; 4 = perte du GPU (enfant) ; 1 = autre erreur, refus, ou reprises
abandonnées.
Aucun run de plus de 30 min sans smoke test vert sur le même commit (CLAUDE.md, règle 5) : le lanceur ne le vérifie pas.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

VARIANTS = {"std": "run_4a_std.yaml", "s": "run_4a_s.yaml"}
# Interventions après GCO-S (ADR-006 D5) : nom de variante = nom de config = préfixe du run ; toutes en GCO-std.
INTERVENTIONS = {
    "int_gamma998": "int_gamma998.yaml",
    "int_slow_anneal": "int_slow_anneal.yaml",
    "int_sil": "int_sil.yaml",  # ADR-009 (proposé) : self-imitation, bloc `sil` seul ajouté
}
# Témoins de diagnostic HORS SPEC (ADR-011) : récompense dense-diag, une seule modification par rapport à std s0 ;
# diag_dense_lr3 (ADR-012 D1, D2) : une seule modification par rapport à diag_dense (calendrier de LR ×3).
DIAGNOSTICS = {"diag_dense": "diag_dense.yaml", "diag_dense_lr3": "diag_dense_lr3.yaml"}
EVAL_CONFIG = "run_4a_eval.yaml"
INTERVENTION_EVAL_CONFIG = "run_s1v2_eval.yaml"  # ADR-006 D2, D3 : protocole S1-seed v2
EXIT_OK, EXIT_ERROR, EXIT_KILL, EXIT_DEVICE_LOST = 0, 1, 3, 4
DEVICE_LOST = "UR_RESULT_ERROR_DEVICE_LOST"  # message du runtime Level Zero (torch XPU) quand le pilote perd le GPU
MAX_NO_PROGRESS = 3  # pannes de suite sans nouveau checkpoint : le GPU ne revient pas, on abandonne
RETRY_WAIT_S = 60.0  # laisse le pilote se rétablir avant de relancer
_CKPT_RE = re.compile(r"^ckpt_\d{6}_it\d{8}\.pt$")  # `training.checkpoint`

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def config_file(variant: str) -> str:
    """Nom du fichier de config (dans `configs/`) de la variante ou de l'intervention ; lève si inconnue."""
    if variant in VARIANTS:
        return VARIANTS[variant]
    if variant in INTERVENTIONS:
        return INTERVENTIONS[variant]
    if variant in DIAGNOSTICS:
        return DIAGNOSTICS[variant]
    raise ValueError(f"variante {variant!r} : {sorted(VARIANTS) + sorted(INTERVENTIONS) + sorted(DIAGNOSTICS)} attendu")


def run_name(variant: str, seed: int) -> str:
    """`run_4a_<variante>_s<seed>` pour std et s (inchangé) ; `<intervention>_s<seed>` pour une intervention."""
    config_file(variant)
    if seed < 0:
        raise ValueError(f"seed={seed} : >= 0 attendu")
    return f"{variant}_s{seed}" if variant in INTERVENTIONS or variant in DIAGNOSTICS else f"run_4a_{variant}_s{seed}"


def default_eval_config(variant: str) -> str:
    """Config d'évaluation par défaut : S1-léger pour std et s (inchangé), S1-seed v2 pour une intervention."""
    return INTERVENTION_EVAL_CONFIG if variant in INTERVENTIONS or variant in DIAGNOSTICS else EVAL_CONFIG


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


def resolve_eval_config(config_dir: Path, name: str | Path) -> Path:
    """Fichier de config d'évaluation : un nom dans `config_dir`, sinon un chemin existant (relatif au répertoire
    courant, ou absolu). Lève `FileNotFoundError` si aucun des deux n'existe."""
    p = Path(name)
    if p.is_absolute():
        if p.is_file():
            return p
    else:
        for cand in (config_dir / p, p):
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"config d'évaluation introuvable : {name} (cherchée dans {config_dir} puis comme chemin)")


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


def is_device_lost(exc: BaseException | None) -> bool:
    """Vrai si l'exception, ou une de ses causes (`LearnerError` enveloppe celle du thread learner), est une perte
    du GPU par le pilote."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if DEVICE_LOST in str(exc):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def segment_device_lost(seg: dict[str, Any] | None) -> bool:
    """La session décrite par `seg` (ligne de `segments.jsonl`) a fini sur une perte du GPU."""
    return seg is not None and seg.get("status") == "error" and DEVICE_LOST in str(seg.get("error") or "")


def latest_checkpoint_iteration(ck_dir: Path) -> int | None:
    """Itération du checkpoint le plus récent de `ck_dir` (nom `ckpt_<index>_it<itération>.pt`), None si aucun."""
    if not ck_dir.is_dir():
        return None
    its = [int(p.name[14:22]) for p in ck_dir.iterdir() if _CKPT_RE.match(p.name)]
    return max(its) if its else None


def supervise_run(
    spawn: Callable[[bool, float | None], int],
    state: Callable[[], tuple[dict[str, Any] | None, int | None]],
    *,
    max_hours: float | None,
    resume_first: bool = False,
    auto_resume: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> str:
    """Mène un run jusqu'au bout à travers des sessions enfants. `spawn(resume_after_stop, heures_restantes)` lance
    une session et rend son code de sortie ; `state()` rend (dernier segment, itération du dernier checkpoint).
    Rend « completed », « paused » (durée atteinte, Ctrl+C, plafond d'itérations), « stopped » (kill criterion),
    « error » (autre erreur, ou reprises abandonnées)."""
    start = clock()
    resume = resume_first
    no_progress = 0
    while True:
        remaining = None
        if max_hours is not None:
            remaining = max_hours - (clock() - start) / 3600.0
            if remaining <= 0.0:
                return "paused"
        _, ck_before = state()
        code = spawn(resume, remaining)
        seg, ck_after = state()
        if code == EXIT_KILL:
            return "stopped"
        if code == EXIT_OK:
            return "completed" if seg is not None and seg.get("status") == "completed" else "paused"
        if code != EXIT_DEVICE_LOST or not auto_resume:
            return "error"
        progressed = ck_after is not None and (ck_before is None or ck_after > ck_before)
        no_progress = 0 if progressed else no_progress + 1
        if no_progress >= MAX_NO_PROGRESS:
            log(f"[run_4a] {no_progress} pertes du GPU de suite sans nouveau checkpoint : reprises abandonnées")
            return "error"
        log(f"[run_4a] GPU perdu (dernier checkpoint : itération {ck_after}) : reprise dans {RETRY_WAIT_S:.0f} s")
        sleep(RETRY_WAIT_S)
        resume = True


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
    eval_config: str | Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Une session du run `run_name(variant, seed)`. `eval_config` : None = `default_eval_config(variant)`.
    `run_kwargs`, `hook_kwargs`, `config_edit` : tests seulement (env et simulateur factices, config réduite)."""
    import train as train_script

    from training import RunConfig, run
    from training.evalhook import EvalHookConfig

    name = run_name(variant, seed)
    cfg = RunConfig.from_yaml(config_dir / config_file(variant))
    if variant in INTERVENTIONS and cfg.reward.variant != "GCO-std":
        raise ValueError(f"intervention {variant} : GCO-std attendu (ADR-006 D5), {cfg.reward.variant} trouvé")
    if variant in DIAGNOSTICS and cfg.reward.variant != "dense-diag":
        raise ValueError(f"diagnostic {variant} : dense-diag attendu (ADR-011), {cfg.reward.variant} trouvé")
    cfg = dataclasses.replace(cfg, name=name, seed=seed)
    if device:
        cfg = dataclasses.replace(cfg, device=device)
    if config_edit is not None:
        cfg = config_edit(cfg)
    if eval_config is None:
        eval_config = default_eval_config(variant)
    hook_cfg = EvalHookConfig.from_yaml(resolve_eval_config(config_dir, eval_config))
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
    ap.add_argument(
        "--variant",
        choices=sorted(VARIANTS) + sorted(INTERVENTIONS) + sorted(DIAGNOSTICS),
        required=True,
        help="std = GCO-std, s = GCO-S ; int_gamma998 (ADR-007), int_slow_anneal (ADR-008), int_sil (ADR-009) : "
        "interventions GCO-std ; diag_dense (ADR-011) : témoin à récompense dense, HORS SPEC, jamais promu ; "
        "diag_dense_lr3 (ADR-012) : le même témoin, LR ×3",
    )
    ap.add_argument("--seed", type=int, nargs="+", required=True, help="0, 1 ou 2 (ADR-005 D3) ; plusieurs : enchaînés")
    ap.add_argument("--max-hours", type=float, default=7.5, help="durée murale par run (défaut 7,5 h ; 0 = aucune)")
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
    ap.add_argument(
        "--eval-config",
        help=f"YAML de l'évaluation en cours de run, dans configs/ ou chemin (défaut {EVAL_CONFIG} : S1-léger, pour "
        f"std et s ; {INTERVENTION_EVAL_CONFIG} : S1-seed v2, ADR-006, défaut des interventions)",
    )
    ap.add_argument("--no-auto-resume", action="store_true", help="pas de reprise automatique après perte du GPU")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)  # une session, lancée par le superviseur
    args = ap.parse_args(argv)
    if args.eval_config is None:
        args.eval_config = default_eval_config(args.variant)
    max_hours = args.max_hours if args.max_hours > 0 else None

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
    if not args.child:
        return _supervise_main(args, max_hours)
    if len(args.seed) != 1:
        print("--child : une seule seed", file=sys.stderr)
        return EXIT_ERROR
    bench_run._set_rocketgpu_defaults()
    try:
        with keep_awake(not args.no_keep_awake) as awake:
            print(f"[run_4a] veille par inactivité bloquée : {awake}", flush=True)
            code, report = launch(
                args.variant,
                args.seed[0],
                max_hours=max_hours,
                runs_root=args.runs_root,
                checkpoints_root=args.checkpoints_root,
                config_dir=ROOT / "configs",
                device=args.device,
                max_iterations=args.max_iterations,
                command=[sys.executable, *sys.argv],
                resume_after_stop=args.resume_after_stop,
                eval_config=args.eval_config,
            )
    except (ConfigError, RunError, RuntimeError, ValueError, OSError) as exc:
        print(f"run impossible : {type(exc).__name__} : {exc}", file=sys.stderr)
        return EXIT_DEVICE_LOST if is_device_lost(exc) else EXIT_ERROR
    print(json.dumps(report, indent=2, default=str))
    left = report["remaining_iterations"]
    if code == EXIT_KILL:
        print(f"[run_4a] STOP ({report.get('cause')}) : plan suspendu, ne pas relancer sans décision (ADR-005 D6)")
    elif left > 0:
        print(f"[run_4a] session finie ({report['status']}), {left} itérations restantes : relancer la même commande")
    else:
        print("[run_4a] budget atteint : run terminé")
    return code


def child_command(args: argparse.Namespace, seed: int, resume_after_stop: bool, hours: float | None) -> list[str]:
    """Ligne de commande d'une session enfant (mêmes options que le superviseur, une seule seed)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--variant", args.variant, "--seed", str(seed)]
    cmd += ["--max-hours", repr(hours if hours is not None else 0.0)]
    cmd += ["--runs-root", str(args.runs_root), "--checkpoints-root", str(args.checkpoints_root)]
    if args.device:
        cmd += ["--device", args.device]
    if args.max_iterations is not None:
        cmd += ["--max-iterations", str(args.max_iterations)]
    for flag in ("allow_dirty", "no_keep_awake", "std_s1_observed"):
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))
    # toujours transmise : le défaut dépend de la variante (`default_eval_config`), un choix explicite égal au défaut
    # d'une autre variante ne doit pas se perdre (revue de 933a8e8, majeur)
    eval_config = getattr(args, "eval_config", None) or default_eval_config(args.variant)
    cmd += ["--eval-config", str(eval_config)]
    if resume_after_stop:
        cmd.append("--resume-after-stop")
    return cmd


def run_child(cmd: list[str]) -> int:
    """Lance une session enfant et attend sa fin. Sur Ctrl+C (reçu aussi par l'enfant, même console), on ne tue pas
    l'enfant : il s'arrête proprement (checkpoint et segment `interrupted`), puis le superviseur sort (revue, M1)."""
    proc = subprocess.Popen(cmd)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            print("[run_4a] Ctrl+C : attente de l'arrêt propre de la session enfant", flush=True)


def _supervise_main(args: argparse.Namespace, max_hours: float | None) -> int:
    """Enchaîne les seeds ; chaque run passe par `supervise_run` (sessions enfants, reprise après perte du GPU)."""
    with keep_awake(not args.no_keep_awake) as awake:
        print(f"[run_4a] superviseur : seeds {args.seed}, veille par inactivité bloquée : {awake}", flush=True)
        for k, seed in enumerate(args.seed):
            name = run_name(args.variant, seed)
            # le forçage de l'utilisateur ne vaut que pour la 1re session de la 1re seed (revue, mineur 1)
            forced = [bool(args.resume_after_stop) and k == 0]
            out_dir, ck_dir = args.runs_root / name, args.checkpoints_root / name

            def spawn(resume: bool, hours: float | None, seed: int = seed, forced: list[bool] = forced) -> int:
                cmd = child_command(args, seed, resume or forced[0], hours)
                forced[0] = False
                print(f"[run_4a] session enfant : {' '.join(cmd[2:])}", flush=True)
                return run_child(cmd)

            def state(out_dir: Path = out_dir, ck_dir: Path = ck_dir) -> tuple[dict[str, Any] | None, int | None]:
                return last_segment(out_dir), latest_checkpoint_iteration(ck_dir)

            # run déjà arrêté par une perte du GPU avant ce lancement : reprise directe, même règle que la nuit
            result = supervise_run(
                spawn,
                state,
                max_hours=max_hours,
                resume_first=not args.no_auto_resume and segment_device_lost(last_segment(out_dir)),
                auto_resume=not args.no_auto_resume,
            )
            print(f"[run_4a] {name} : {result}", flush=True)
            if result != "completed":
                print("[run_4a] enchaînement arrêté : les seeds suivantes ne sont pas lancées", flush=True)
                return {"paused": EXIT_OK, "stopped": EXIT_KILL}.get(result, EXIT_ERROR)
    return EXIT_OK


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consoles Windows en cp1252 par défaut
    raise SystemExit(main())
