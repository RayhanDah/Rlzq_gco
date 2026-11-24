"""Relais jour/nuit du run long de démo (ADR-014 D6) : la nuit sur le laptop, le jour sur Kaggle T4, un seul run.

Sous-commandes (à blanc par défaut : rien n'est envoyé ni écrit dans runs/ ou checkpoints/ sans `--push` / `--apply`) :
  pack    run en pause → build/kaggle/long/data/ : bundle.tar (journaux du run, dernier checkpoint sain, instantanés
          d'éval), long-bundle.json, dataset-metadata.json ; empreinte de l'état local dans last_pack.json.
          `--push` publie une version du dataset privé `<user>/rlgco-long-s0` (`--create` la 1re fois).
  start   rend kaggle/train/long_kernel.py (commit poussé, empreinte du bundle) et le pousse (`--push`).
          `--smoke` : smoke T4 seul (configs/smoke.yaml en CUDA), sans dataset ni reprise.
  status  statut du kernel.
  fetch   télécharge les sorties du kernel dans build/kaggle/long/out/<horodatage>/ et vérifie l'archive.
  sync    réintègre une sortie dans runs/<run> et checkpoints/<run> (`--apply`). Refuse si l'état local a changé
          depuis `pack` (une session locale aurait divergé), si l'archive est corrompue ou si le run a reculé.
          Sauvegarde runs/<run> avant d'écrire ; ne supprime jamais rien ; un checkpoint de même nom doit être
          identique octet pour octet.

Le jeton Kaggle n'est jamais lu : le CLI `kaggle` (installé par l'utilisateur) s'en charge, via `KaggleCli` de
`scripts/kaggle_run.py` (motifs de secret refusés dans tout ce qui est écrit ou affiché).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import kaggle_run as kr  # noqa: E402

TEMPLATE = ROOT / "kaggle" / "train" / "long_kernel.py"
BUILD = ROOT / "build" / "kaggle" / "long"
RUN = "long_s0"
SEED = 0
DATASET_SLUG = "rlgco-long-s0"
BUNDLE_TAR = "bundle.tar"
BUNDLE_MANIFEST = "long-bundle.json"
SESSION_S = 12 * 3600 - 10 * 60  # session GPU ≤ 12 h (docs/research/kaggle.md) moins 10 min de garde
MARGIN_S = 45 * 60  # fin du run 45 min avant SESSION_S : dernière éval, archive, écriture des sorties
SMOKE_SESSION_S = 30 * 60  # règle 5 de CLAUDE.md : smoke ≤ 30 min (smoke T4 et essai de reprise)
SMOKE_MARGIN_S = 5 * 60
MACHINE = kr.ACCELERATORS["t4"].machine_shape


class LongRelayError(RuntimeError):
    """Refus explicite ; rien n'a été écrit dans runs/ ni checkpoints/."""


# ------------------------------------------------------------------------------------------------- état local


def run_dirs(runs_root: Path, ck_root: Path, run: str = RUN) -> tuple[Path, Path]:
    return runs_root / run, ck_root / run


def fingerprint(*dirs: Path) -> dict[str, list[Any]]:
    """{chemin relatif à la racine du dossier parent : [taille, sha256]} de tous les fichiers (ordre trié)."""
    out: dict[str, list[Any]] = {}
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                out[f"{d.parent.name}/{d.name}/{p.relative_to(d).as_posix()}"] = [p.stat().st_size, kr.sha256_file(p)]
    return out


def session_running(runs_dir: Path) -> bool:
    """Vrai si une session a démarré sans segment de fin (processus en cours ou tué sans trace)."""
    ev, seg = runs_dir / "events.jsonl", runs_dir / "segments.jsonl"
    starts = sum(1 for line in _lines(ev) if json.loads(line).get("event") == "start")
    return starts > len(_lines(seg))


def _lines(p: Path) -> list[str]:
    return [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def latest_healthy(ck_dir: Path) -> tuple[Path, int]:
    from training import CheckpointManager

    found = CheckpointManager(ck_dir).load_latest(healthy_only=True)
    if found is None:
        raise LongRelayError(f"aucun checkpoint sain et intègre dans {ck_dir}")
    return found[0], int(found[1]["iteration"])


# ------------------------------------------------------------------------------------------------------ pack


def pack(
    runs_root: Path, ck_root: Path, out: Path, user: str, commit: str, now: float, assume_stopped: bool = False
) -> dict[str, Any]:
    """`assume_stopped` : l'utilisateur atteste qu'aucune session ne tourne (dernière session tuée sans segment de fin,
    par exemple au timeout d'un kernel)."""
    runs_dir, ck_dir = run_dirs(runs_root, ck_root)
    if not runs_dir.is_dir() or not ck_dir.is_dir():
        raise LongRelayError(f"run {RUN} introuvable ({runs_dir}, {ck_dir})")
    if session_running(runs_dir) and not assume_stopped:
        raise LongRelayError(
            "une session du run semble en cours (start sans segment de fin) : l'arrêter d'abord, ou --assume-stopped "
            "si elle a été tuée"
        )
    ckpt, iteration = latest_healthy(ck_dir)
    kr.fresh_dir(out)
    tar = out / BUNDLE_TAR
    with tarfile.open(tar, "w") as tf:
        tf.add(runs_dir, arcname=f"runs/{RUN}")
        tf.add(ckpt, arcname=f"checkpoints/{RUN}/{ckpt.name}")
        snaps = ck_dir / "eval_snapshots"
        if snaps.is_dir():
            tf.add(snaps, arcname=f"checkpoints/{RUN}/eval_snapshots")
    manifest = {
        "schema": 1,
        "run": RUN,
        "iteration": iteration,
        "checkpoint": ckpt.name,
        "sha256": kr.sha256_file(tar),
        "size": tar.stat().st_size,
        "files": tar_file_hashes(tar),  # Kaggle décompresse l'archive téléversée (mesuré le 2026-11-24)
        "commit": commit,
        "created_at": kr.iso(now),
    }
    kr.write_json(out / BUNDLE_MANIFEST, manifest)
    kr.write_json(
        out / kr.DATASET_META, {"id": f"{user}/{DATASET_SLUG}", "title": DATASET_SLUG, "licenses": [{"name": "other"}]}
    )
    return {**manifest, "local_fingerprint": fingerprint(runs_dir, ck_dir)}


def tar_file_hashes(tar: Path) -> dict[str, str]:
    """{chemin dans l'archive : sha256} des fichiers réguliers : le kernel vérifie chaque fichier quand Kaggle a
    décompressé l'archive à l'import (dossier `bundle/` au lieu de `bundle.tar`)."""
    import hashlib

    out: dict[str, str] = {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            if m.isfile():
                f = tf.extractfile(m)
                assert f is not None
                out[m.name] = hashlib.file_digest(f, "sha256").hexdigest()
    return out


# --------------------------------------------------------------------------------------------------- kernel


def render_kernel(config: dict[str, Any]) -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    marker = "CONFIG: dict[str, object] = {}"
    if src.count(marker) != 1:
        raise LongRelayError("marqueur CONFIG introuvable dans le modèle de kernel")
    out = src.replace(marker, f"CONFIG: dict[str, object] = json.loads({json.dumps(config, ensure_ascii=False)!r})", 1)
    kr.assert_no_secret(out, "le kernel rendu")
    return out


def kernel_slug(smoke: bool) -> str:
    return "rlgco-long-smoke-t4" if smoke else "rlgco-long-s0-t4"


def kernel_metadata(user: str, smoke: bool) -> dict[str, Any]:
    slug = kernel_slug(smoke)
    return {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "long_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",  # clone du dépôt public, pip, meshes RLGym, sources RocketGPU
        "machine_shape": MACHINE,
        "dataset_sources": [] if smoke else [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def kernel_config(
    repo: str, commit: str, smoke: bool, bundle_sha256: str | None, max_iterations: int | None = None
) -> dict[str, Any]:
    """`max_iterations` : mode essai (plafond ABSOLU d'itérations, session courte) ; None : vraie session de jour."""
    if not repo.startswith("https://github.com/"):
        raise LongRelayError(f"dépôt {repo!r} : URL https GitHub publique attendue (le kernel la clone)")
    if not smoke and not bundle_sha256:
        raise LongRelayError("aucun pack à reprendre : lancer `pack --push` d'abord")
    short = smoke or max_iterations is not None
    return {
        "repo": repo,
        "commit": commit,
        "run": RUN,
        "seed": SEED,
        "smoke": smoke,
        "session_s": SMOKE_SESSION_S if short else SESSION_S,
        "margin_s": SMOKE_MARGIN_S if short else MARGIN_S,
        "bundle_sha256": bundle_sha256,
        "max_iterations": max_iterations,
    }


def build_kernel(dest: Path, config: dict[str, Any], meta: dict[str, Any]) -> Path:
    kr.fresh_dir(dest)
    (dest / "long_kernel.py").write_text(render_kernel(config), encoding="utf-8")
    kr.write_json(dest / kr.KERNEL_META, meta)
    return dest


# ----------------------------------------------------------------------------------------------------- sync


def check_output(out_dir: Path) -> dict[str, Any]:
    rep = kr.read_json(out_dir / "relay-out.json")
    if rep is None:
        raise LongRelayError(f"relay-out.json absent de {out_dir}")
    bundle = rep.get("bundle")
    if not bundle:
        raise LongRelayError(f"aucune archive dans la sortie (statut {rep.get('status')!r}, {rep.get('error')!r})")
    tar = out_dir / str(bundle["file"])
    if not tar.is_file() or kr.sha256_file(tar) != bundle["sha256"]:
        raise LongRelayError(f"archive {tar.name} absente ou d'empreinte différente de relay-out.json")
    return rep


def sync(
    out_dir: Path,
    runs_root: Path,
    ck_root: Path,
    pack_record: dict[str, Any],
    *,
    apply: bool,
    stamp: str,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Plan (et application si `apply`) de la réintégration d'une sortie Kaggle. Rend un résumé."""
    rep = check_output(out_dir)
    if rep.get("start_sha256") != pack_record["sha256"]:
        raise LongRelayError("cette sortie ne part pas du pack en attente (empreinte d'entrée différente) : refusée")
    runs_dir, ck_dir = run_dirs(runs_root, ck_root)
    if fingerprint(runs_dir, ck_dir) != pack_record["local_fingerprint"]:
        raise LongRelayError("l'état local du run a changé depuis `pack` (session locale ?) : sync refusée, à trancher")
    staging = kr.fresh_dir(out_dir / "staging")  # à côté de la sortie téléchargée, hors de runs/ et checkpoints/
    with tarfile.open(out_dir / str(rep["bundle"]["file"])) as tf:
        tf.extractall(staging, filter="data")
    s_runs, s_ck = run_dirs(staging / "runs", staging / "checkpoints")
    if not s_runs.is_dir() or not s_ck.is_dir():
        raise LongRelayError("archive sans runs/<run> ou checkpoints/<run>")
    _, it_new = latest_healthy(s_ck)
    if it_new < int(pack_record["iteration"]):
        raise LongRelayError(f"le run a reculé : itération {it_new} < {pack_record['iteration']} au pack")
    copies: list[tuple[Path, Path]] = []
    for src in sorted(s_ck.rglob("*")):
        if not src.is_file():
            continue
        dst = ck_dir / src.relative_to(s_ck)
        if dst.exists():
            if kr.sha256_file(dst) != kr.sha256_file(src):
                raise LongRelayError(f"{dst.name} existe localement avec un autre contenu : sync refusée")
            continue
        copies.append((src, dst))
    for src in sorted(s_runs.rglob("*")):
        if src.is_file():
            copies.append((src, runs_dir / src.relative_to(s_runs)))
    summary = {
        "iteration_before": pack_record["iteration"],
        "iteration_after": it_new,
        "files": len(copies),
        "status": rep.get("status"),
        "applied": apply,
    }
    echo(
        f"sync {RUN} : itération {pack_record['iteration']} → {it_new}, {len(copies)} fichiers"
        + ("" if apply else " (à blanc : --apply pour écrire)")
    )
    if not apply:
        return summary
    backup = BUILD / "backup" / f"{stamp}-{RUN}-runs.tar"
    backup.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(backup, "w") as tf:
        tf.add(runs_dir, arcname=f"runs/{RUN}")
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".sync-tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
    summary["backup"] = str(backup)
    return summary


# ------------------------------------------------------------------------------------------------------ CLI


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def pushed_commit() -> str:
    """HEAD, à condition que l'arbre soit propre et que HEAD soit sur origin/main (le kernel clone GitHub)."""
    if git("status", "--porcelain"):
        raise LongRelayError("arbre git sale : committer et pousser d'abord")
    git("fetch", "--quiet", "origin")
    head = git("rev-parse", "HEAD")
    try:
        git("merge-base", "--is-ancestor", head, "origin/main")
    except subprocess.CalledProcessError as e:
        raise LongRelayError(f"HEAD {head[:7]} n'est pas sur origin/main : pousser d'abord") from e
    return head


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "start", "status", "fetch", "sync"):
        p = sub.add_parser(name)
        if name != "sync":
            p.add_argument("--user", required=True, help="pseudo Kaggle du titulaire du compte (R11)")
        if name in ("pack", "start"):
            p.add_argument("--push", action="store_true", help="exécute réellement les commandes kaggle")
        if name in ("start", "status", "fetch"):
            p.add_argument("--smoke", action="store_true", help="kernel de smoke T4")
        if name == "start":
            p.add_argument("--trial", type=int, help="essai de reprise : N itérations depuis le pack, session ≤ 30 min")
        if name == "pack":
            p.add_argument("--create", action="store_true", help="1re publication : crée le dataset (privé)")
            p.add_argument("--assume-stopped", action="store_true", help="aucune session ne tourne (dernière tuée)")
        if name == "sync":
            p.add_argument("--from", dest="src", type=Path, required=True, help="dossier écrit par `fetch`")
            p.add_argument("--apply", action="store_true", help="écrit dans runs/ et checkpoints/")
    args = ap.parse_args(argv)
    runs_root, ck_root = ROOT / "runs", ROOT / "checkpoints"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    last_pack = BUILD / "last_pack.json"
    try:
        if args.cmd == "sync":
            record = kr.read_json(last_pack)
            if record is None:
                raise LongRelayError("aucun pack en attente (build/kaggle/long/last_pack.json)")
            sync(args.src, runs_root, ck_root, record, apply=args.apply, stamp=stamp)
            if args.apply:
                last_pack.replace(BUILD / f"pack-{stamp}-synced.json")  # un pack ne se réintègre qu'une fois
            return 0
        cli = kr.KaggleCli(kr.subprocess_executor if getattr(args, "push", True) else None, exe=kr.kaggle_exe())
        if args.cmd == "pack":
            if last_pack.exists():
                raise LongRelayError("un pack attend déjà sa sync (build/kaggle/long/last_pack.json)")
            data = BUILD / "data"
            record = pack(
                runs_root, ck_root, data, args.user, git("rev-parse", "HEAD"), time.time(), args.assume_stopped
            )
            mo, sha = record["size"] / 1e6, record["sha256"][:12]
            print(f"bundle : itération {record['iteration']}, {mo:.0f} Mo, sha {sha}")
            if args.create:
                cli.datasets_create(data)
            else:
                cli.datasets_version(data, f"{RUN} it. {record['iteration']}")
            if args.push:
                kr.write_json(last_pack, record)
            return 0
        if args.cmd == "start":
            record = kr.read_json(last_pack)
            sha = None if args.smoke else (record or {}).get("sha256")
            cap = None if args.trial is None or record is None else int(record["iteration"]) + args.trial
            config = kernel_config(git("remote", "get-url", "origin"), pushed_commit(), args.smoke, sha, cap)
            folder = build_kernel(
                BUILD / ("kernel-smoke" if args.smoke else "kernel"), config, kernel_metadata(args.user, args.smoke)
            )
            cli.kernels_push(folder, MACHINE, int(config["session_s"]) + 10 * 60)
            return 0
        kernel = f"{args.user}/{kernel_slug(args.smoke)}"
        cli = kr.KaggleCli(kr.subprocess_executor, exe=kr.kaggle_exe())
        if args.cmd == "status":
            cli.kernels_status(kernel)
            return 0
        dest = BUILD / "out" / f"{stamp}{'-smoke' if args.smoke else ''}"
        dest.mkdir(parents=True)
        cli.kernels_output(kernel, dest)
        rep = kr.read_json(dest / "relay-out.json") or {}
        print(
            json.dumps(
                {
                    k: rep.get(k)
                    for k in ("status", "error", "start_iteration", "max_hours", "versions", "elapsed_s", "steps")
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if not args.smoke and rep.get("bundle"):
            check_output(dest)
            print(f"sortie vérifiée : python scripts/kaggle_long.py sync --from {dest.relative_to(ROOT).as_posix()}")
        return 0
    except (LongRelayError, kr.RelayError, kr.KaggleError) as e:
        print(f"refus : {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
