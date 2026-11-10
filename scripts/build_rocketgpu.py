"""Construit RocketGPU (simulateur tiers, MIT) à un commit épinglé : CPU (option PGO) ou CUDA.

Le même script sert au laptop, aux PC des membres et aux kernels Kaggle (Linux).
Usage :
  python scripts/build_rocketgpu.py                    # CPU, -march=native → vendor/build/rocketgpu-cpu
  python scripts/build_rocketgpu.py --pgo              # CPU + PGO (profil = notre benchmark) → rocketgpu-cpu-pgo
  python scripts/build_rocketgpu.py --variant cuda     # CUDA (toolkit requis) → rocketgpu-cuda
Ensuite : ROCKETGPU_PY_DIR=<dossier python affiché> python scripts/benchmark_sim.py --engine rocketgpu ...
Prérequis : git, cmake ≥ 3.21, ninja, pybind11 (`uv sync` installe le groupe `native`), un compilateur C++17
(MinGW g++ sous Windows : l'option PGO et -march=native du projet ne s'appliquent qu'à gcc/clang).
Le code source et les binaires restent dans vendor/ (non versionné). Stdlib uniquement.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/thecosmicskye/RocketGPU.git"
PINNED_SHA = "f0508a184829d1cb32522e3e37debe5c1bcd895d"  # 2026-11-08, « Update README », inspecté le 2026-11-10
PGO_TRAIN = ("--procs", "1", "--arenas", "256", "--engine", "rocketgpu", "--seconds", "30")


def tool(name: str) -> str:
    """Préfère l'outil du venv courant (groupe uv `native`), sinon le PATH."""
    bindir = Path(sys.executable).parent
    for cand in (bindir / f"{name}.exe", bindir / name):
        if cand.exists():
            return str(cand)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} introuvable (lancer `uv sync` ou `pip install cmake ninja pybind11`)")
    return found


def find_nvcc() -> str | None:
    """nvcc du PATH, sinon /usr/local/cuda*/bin/nvcc (images Kaggle : le toolkit n'est pas toujours dans le PATH)."""
    found = shutil.which("nvcc")
    if found:
        return found
    cands = sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))
    return cands[-1] if cands else None


def has_profile_data(profile: Path) -> bool:
    """Profil PGO non vide : *.gcda (gcc) ou *.profraw (clang) de taille > 0, comme le vérifie tools/pgo_build.sh."""
    return any(f.stat().st_size > 0 for pat in ("*.gcda", "*.profraw") for f in profile.rglob(pat))


def cmake_args(
    src: Path,
    build: Path,
    *,
    variant: str,
    pgo: str,
    profile: Path | None,
    portable: bool,
    python: str,
    windows: bool,
    cuda_compiler: str | None = None,
) -> list[str]:
    """Arguments de configuration CMake. `pgo` ∈ {OFF, GENERATE, USE}."""
    if variant not in ("cpu", "cuda"):
        raise ValueError(f"variante inconnue : {variant}")
    if pgo not in ("OFF", "GENERATE", "USE"):
        raise ValueError(f"mode PGO inconnu : {pgo}")
    if variant == "cuda" and pgo != "OFF":
        raise ValueError("PGO réservé aux builds CPU")
    if pgo != "OFF" and profile is None:
        raise ValueError("PGO sans dossier de profil")
    args = [
        "-S",
        str(src),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DROCKETGPU_BUILD_PYTHON=ON",
        f"-DPython_EXECUTABLE={python}",
        f"-DROCKETGPU_USE_CUDA={'ON' if variant == 'cuda' else 'OFF'}",
        f"-DROCKETGPU_NATIVE_ARCH={'OFF' if portable else 'ON'}",
    ]
    if variant == "cuda" and cuda_compiler:
        args.append(f"-DCMAKE_CUDA_COMPILER={cuda_compiler}")
    if windows:
        args.append("-DCMAKE_CXX_COMPILER=g++")  # MSVC : ni -march=native ni PGO dans le CMake du projet
    if pgo != "OFF" and profile is not None:
        args += [f"-DROCKETGPU_PGO={pgo}", f"-DROCKETGPU_PGO_PROFILE={profile.as_posix()}"]
    return args


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def ensure_source(src: Path, sha: str) -> None:
    if not (src / ".git").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", REPO_URL, str(src)])
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if head.stdout.strip() != sha:
        run(["git", "-C", str(src), "fetch", "--quiet", "origin", sha])
        run(["git", "-C", str(src), "checkout", "--quiet", "--detach", sha])
    dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True, check=True)
    if dirty.stdout.strip():
        raise SystemExit(f"source RocketGPU modifiée localement : {src}")


def configure_and_build(args: list[str], build: Path, jobs: int) -> None:
    shutil.rmtree(build, ignore_errors=True)
    env = {**os.environ, "PATH": f"{Path(tool('ninja')).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    run([tool("cmake"), *args], env)
    run([tool("cmake"), "--build", str(build), "-j", str(jobs)], env)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--pgo", action="store_true", help="CPU seulement : build instrumenté, profil, rebuild")
    ap.add_argument("--portable", action="store_true", help="sans -march=native (binaire pour d'autres machines)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--sha", default=PINNED_SHA)
    ap.add_argument("--src", type=Path, default=ROOT / "vendor" / "src" / "RocketGPU")
    ap.add_argument("--out", type=Path, default=ROOT / "vendor" / "build")
    ap.add_argument("--meshes", default=str(ROOT / "vendor" / "collision_meshes"))
    a = ap.parse_args()

    ensure_source(a.src, a.sha)
    windows = os.name == "nt"
    name = f"rocketgpu-{a.variant}" + ("-pgo" if a.pgo else "") + ("-portable" if a.portable else "")
    build = a.out / name
    nvcc = find_nvcc() if a.variant == "cuda" else None
    if a.variant == "cuda" and nvcc is None:
        print("nvcc introuvable (PATH, /usr/local/cuda*/bin)", file=sys.stderr)
        return 1
    common = {
        "variant": a.variant,
        "portable": a.portable,
        "python": sys.executable,
        "windows": windows,
        "cuda_compiler": nvcc,
    }
    if a.pgo:
        profile = a.out / f"{name}-profile"
        shutil.rmtree(profile, ignore_errors=True)
        profile.mkdir(parents=True)
        gen = a.out / f"{name}-gen"
        configure_and_build(cmake_args(a.src, gen, pgo="GENERATE", profile=profile, **common), gen, a.jobs)
        # Mêmes réglages d'entraînement que tools/pgo_build.sh du projet.
        env = {
            **os.environ,
            "ROCKETGPU_PY_DIR": str(gen / "python"),
            "ROCKETGPU_CPU_THREADS": "1",
            "ROCKETGPU_CPU_BLOCK": "4",
            "CCACHE_DISABLE": "1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            bench = [sys.executable, str(ROOT / "scripts" / "benchmark_sim.py"), *PGO_TRAIN]
            run([*bench, "--meshes", a.meshes, "--out", tmp], env)
        if not has_profile_data(profile):
            print(f"profil PGO vide dans {profile} : abandon (sinon binaire « -pgo » sans PGO)", file=sys.stderr)
            return 1
        configure_and_build(cmake_args(a.src, build, pgo="USE", profile=profile, **common), build, a.jobs)
    else:
        configure_and_build(cmake_args(a.src, build, pgo="OFF", profile=None, **common), build, a.jobs)

    py_dir = build / "python"
    if not any(py_dir.glob("rocketgpu_py*")):
        print(f"module rocketgpu_py absent de {py_dir}", file=sys.stderr)
        return 1
    info = {"sha": a.sha, "variant": a.variant, "pgo": a.pgo, "portable": a.portable, "py_dir": str(py_dir)}
    (build / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
