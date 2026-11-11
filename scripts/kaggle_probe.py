"""Construit (et, avec --push, lance) le kernel « sonde » sur le compte Kaggle de la personne qui exécute ce script.

Chaque membre l'exécute depuis son PC avec son propre jeton (installé par lui, jamais lu par ce script).
Usage :
  python scripts/kaggle_probe.py --user <pseudo_kaggle> --acc t4,tpu            # construit seulement (build/kaggle/)
  python scripts/kaggle_probe.py --user <pseudo_kaggle> --acc t4,tpu --push     # pousse et lance les kernels
  .venv/Scripts/kaggle kernels status <pseudo>/rlgco-probe-t4                   # suivi
  .venv/Scripts/kaggle kernels output <pseudo>/rlgco-probe-t4 -p kaggle_out/probe-t4
Pousser deux accélérateurs à la suite teste aussi la concurrence GPU + TPU sur un même compte.
Colab (aucun appel Kaggle) :
  python scripts/kaggle_probe.py --colab      # écrit build/colab/rlgco_probe.py, à téléverser dans Colab
  puis, dans une cellule Colab : !python rlgco_probe.py   (résultats : /content/probe.json, /content/bench/)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "kaggle" / "probe" / "probe_kernel.py"
EMBED = ("scripts/benchmark_sim.py", "scripts/fetch_meshes.py", "scripts/build_rocketgpu.py")
ACCELERATORS = {"t4": "NvidiaTeslaT4", "tpu": "TpuV5E8"}
TIMEOUT_S = 50 * 60  # budget interne du kernel : 40 min


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


PROBE_PARTS = ("rg_cuda", "rg_cuda12", "rocketsim", "rg_cpu")  # miroir de PARTS dans probe_kernel.py


def render_kernel(commit: str, accelerator: str, parts: str = "") -> str:
    """Rend le kernel autonome. `parts` (ex. "rg_cuda12") fige les parties par défaut, un kernel Kaggle ne recevant
    pas d'arguments ; vide = choix automatique du kernel."""
    src = TEMPLATE.read_text(encoding="utf-8")
    embedded = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in EMBED}
    marker = "EMBEDDED: dict[str, str] = {}"
    if marker not in src:
        raise ValueError("marqueur EMBEDDED introuvable dans le modèle")
    src = src.replace(marker, f"EMBEDDED: dict[str, str] = {json.dumps(embedded, ensure_ascii=False)}", 1)
    if parts:
        names = [p.strip() for p in parts.split(",") if p.strip()]
        unknown = [n for n in names if n not in PROBE_PARTS]
        if unknown or not names:
            raise ValueError(f"parties inconnues : {unknown or parts!r} (choix : {PROBE_PARTS})")
        assign = 'DEFAULT_PARTS = "__PARTS__"'
        if assign not in src:
            raise ValueError("marqueur DEFAULT_PARTS introuvable dans le modèle")
        src = src.replace(assign, f'DEFAULT_PARTS = "{",".join(names)}"', 1)
    return src.replace("__COMMIT__", commit).replace("__ACCELERATOR__", accelerator)


def metadata(user: str, key: str) -> dict[str, object]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,48}", user):
        raise ValueError(f"pseudo Kaggle invalide : {user!r}")
    slug = f"rlgco-probe-{key}"
    return {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "probe_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if key == "t4" else "false",
        "enable_tpu": "true" if key == "tpu" else "false",  # lu par kaggle 2.2.5 (kaggle_api_extended.py:7267)
        "enable_internet": "true",
        "machine_shape": ACCELERATORS[key],
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="pseudo Kaggle (minuscules) ; requis sauf avec --colab")
    ap.add_argument("--colab", action="store_true", help="fichier unique pour Colab, sans métadonnées ni push")
    ap.add_argument("--parts", default="", help="parties figées dans le kernel, ex. rg_cuda12 ; vide = auto")
    ap.add_argument("--acc", default="t4", help="liste parmi t4,tpu")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    if git("status", "--porcelain", "--untracked-files=no"):
        print("arbre git modifié : commiter avant de lancer une sonde (traçabilité du commit)", file=sys.stderr)
        return 1
    commit = git("rev-parse", "--short", "HEAD")
    if args.colab:
        if args.push:
            print("--push est réservé à Kaggle", file=sys.stderr)
            return 2
        out = ROOT / "build" / "colab"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "rlgco_probe.py"
        path.write_text(render_kernel(commit, "colab", args.parts), encoding="utf-8")
        print(f"construit : {path} (commit {commit}) ; dans Colab : !python rlgco_probe.py")
        return 0
    if not args.user:
        print("--user requis (ou --colab)", file=sys.stderr)
        return 2
    keys = [k.strip() for k in args.acc.split(",")]
    for key in keys:
        if key not in ACCELERATORS:
            print(f"accélérateur inconnu : {key}", file=sys.stderr)
            return 2
        out = ROOT / "build" / "kaggle" / f"probe-{key}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        (out / "probe_kernel.py").write_text(render_kernel(commit, ACCELERATORS[key], args.parts), encoding="utf-8")
        meta = metadata(args.user, key)
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"construit : {out} ({meta['id']}, commit {commit})")
        if args.push:
            kaggle = ROOT / ".venv" / "Scripts" / "kaggle.exe"
            exe = str(kaggle) if kaggle.exists() else "kaggle"
            cmd = [exe, "kernels", "push", "-p", str(out), "--accelerator", ACCELERATORS[key], "-t", str(TIMEOUT_S)]
            print("+", " ".join(cmd), flush=True)
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
