"""Kernel Kaggle « sonde » (≤ 40 min) : matériel réel de la session, débit RocketSim, build et débit RocketGPU.

Généré par `scripts/kaggle_probe.py`, qui remplace EMBEDDED par le source de nos scripts (benchmark, meshes, build
RocketGPU) et COMMIT par le commit git local. Aucune donnée privée, aucun secret. Sortie : <WORK>/probe.json et
les rapports JSON du benchmark dans <WORK>/bench/, avec WORK = /kaggle/working sur Kaggle, /content sur Colab,
sinon le dossier courant. Le même fichier s'exécute tel quel dans une cellule Colab : `!python rlgco_probe.py`.
Parties (`--parts`, ordre conservé) : `rg_cuda` (build + bench RocketGPU CUDA), `rocketsim` (bench RocketSim),
`rg_cpu` (build + bench RocketGPU CPU). Défaut : `rg_cuda,rocketsim,rg_cpu` si un GPU est visible, sinon
`rocketsim,rg_cpu`. Exemples Colab : T4 → `!python rlgco_probe.py --parts rg_cuda` ;
TPU v5e-1 → `!python rlgco_probe.py --parts rocketsim,rg_cpu`.
Verdict `ok` : installation et meshes, plus le benchmark RocketSim s'il est demandé. RocketGPU : `ok_rocketgpu`.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

EMBEDDED: dict[str, str] = {}  # remplacé au build : {"scripts/benchmark_sim.py": "...", ...}
COMMIT = "__COMMIT__"
ACCELERATOR = "__ACCELERATOR__"


def _work_dir() -> Path:
    for cand in (Path("/kaggle/working"), Path("/content")):
        if cand.is_dir():
            return cand
    return Path.cwd()


WORK = _work_dir()
PLATFORM = "kaggle" if str(WORK).startswith("/kaggle") else "colab" if str(WORK) == "/content" else "autre"
APP = Path("/tmp/rlgco")  # hors de WORK : meshes, sources et binaires tiers ne font pas partie des sorties
T0 = time.time()
BUDGET_S = 40 * 60
BENCH_S = 15
REQUIRED = ("pip_install", "meshes", "bench_rocketsim")


def run(cmd: str, timeout: int = 120) -> dict[str, object]:
    """Commande shell dans son propre groupe de processus : au timeout, tout le groupe est tué (cmake, ninja, nvcc
    gardent sinon le pipe ouvert et bloquent bien au-delà du timeout)."""
    t = time.time()
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, _ = proc.communicate()
        code = -1
    text = (stdout or "").strip()
    out: dict[str, object] = {"code": code, "out": (f"timeout {timeout}s\n" if code == -1 else "") + text[-6000:]}
    out["full"] = text  # journal complet, écrit dans <WORK>/logs/ par step() puis retiré de probe.json
    out["seconds"] = round(time.time() - t, 1)
    out["cmd"] = cmd
    return out


def error_lines(text: str, limit: int = 40) -> list[str]:
    """Premières lignes d'erreur du compilateur ou de CMake (le traceback Python final les masque sinon)."""
    keys = ("error", "Error", "ERROR", "fatal", "undefined reference", "does not match", "FAILED:")
    return [line[:300] for line in text.splitlines() if any(k in line for k in keys)][:limit]


def sh(cmd: str, timeout: int = 120) -> str:
    return str(run(cmd, timeout)["out"])


def remaining() -> int:
    return int(BUDGET_S - (time.time() - T0))


def py_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("numpy", "torch", "jax", "jaxlib", "torch_xla", "libtpu", "rocketsim", "cmake", "ninja", "pybind11"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "absent"
    return out


PARTS = ("rg_cuda", "rg_cuda12", "rocketsim", "rg_cpu")
DEFAULT_PARTS = "__PARTS__"  # remplacé au build (kaggle_probe.py --parts) ; un kernel Kaggle ne reçoit pas d'arguments
CUDA12 = "12.9"
CUDA12_INSTALL = (
    "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
    " -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update -q"
    " && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends"
    " cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-cccl-12-9"
)


def parse_parts(argv: list[str], has_gpu: bool) -> tuple[list[str], int]:
    """`--parts a,b` (ordre conservé) et `--budget-min N`. Par défaut : la partie GPU d'abord quand un GPU est
    visible (sinon elle passe après les étapes CPU et peut manquer de budget sur une machine à 2 cœurs)."""
    import argparse

    default = DEFAULT_PARTS if DEFAULT_PARTS != "__PARTS__" else "auto"
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=default, help=f"auto ou liste parmi {','.join(PARTS)}")
    ap.add_argument("--budget-min", type=int, default=40)
    a = ap.parse_args(argv)
    if a.parts == "auto":
        parts = ["rg_cuda", "rocketsim", "rg_cpu"] if has_gpu else ["rocketsim", "rg_cpu"]  # rg_cuda12 : sur demande
    else:
        parts = [p.strip() for p in a.parts.split(",") if p.strip()]
        unknown = [p for p in parts if p not in PARTS]
        if unknown:
            raise SystemExit(f"parties inconnues : {unknown} (choix : {PARTS})")
    return parts, a.budget_min * 60


def main() -> int:
    global BUDGET_S
    has_gpu = run("nvidia-smi -L")["code"] == 0
    parts, BUDGET_S = parse_parts(sys.argv[1:], has_gpu)
    start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report: dict[str, object] = {"commit": COMMIT, "accelerator": ACCELERATOR, "platform": PLATFORM, "start": start}
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count() or 1
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv")
    report["system"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "sched_affinity": affinity,
        "nproc": sh("nproc"),
        "lscpu": sh("lscpu"),
        "free_g": sh("free -g"),
        "df_working": sh(f"df -h {WORK}"),
        "tpu_env": sh("env | grep -i -E '^(TPU|COLAB_TPU|XRT)' | cut -c1-120"),
        "nvidia_smi": gpu,
        "nvcc": sh("nvcc --version || ls /usr/local/cuda*/bin/nvcc"),
        "gcc": sh("g++ --version | head -1"),
        "cgroup_cpu_max": sh("cat /sys/fs/cgroup/cpu.max"),
    }
    report["versions_before"] = py_versions()
    report["jax_devices"] = sh(f"{sys.executable} -c 'import jax; print(jax.devices())'", timeout=180)

    for rel, src in EMBEDDED.items():
        p = APP / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    py = sys.executable
    scripts = APP / "scripts"
    meshes = APP / "vendor" / "collision_meshes"
    bench = f"{py} {scripts / 'benchmark_sim.py'} --meshes {meshes} --out {WORK / 'bench'} --seconds {BENCH_S}"
    steps: dict[str, dict[str, object]] = {}
    rg_ok: dict[str, object] = {"cpu": "non tenté", "cuda": "non tenté"}

    def save() -> None:
        """Écrit probe.json après chaque étape : un kernel tué par Kaggle laisse quand même ses résultats partiels."""
        report["steps"] = steps
        required = REQUIRED if "rocketsim" in parts else REQUIRED[:2]
        report["ok"] = all(steps.get(n, {}).get("code") == 0 for n in required)
        report["ok_rocketgpu"] = rg_ok
        report["elapsed_s"] = round(time.time() - T0, 1)
        (WORK / "probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def step(name: str, cmd: str, timeout: int) -> bool:
        if remaining() < 60:
            steps[name] = {"code": -2, "out": "budget épuisé, étape sautée", "cmd": cmd}
        else:
            steps[name] = run(cmd, timeout=min(timeout, remaining()))
            full = str(steps[name].pop("full", ""))
            logs = WORK / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{name}.log").write_text(full, encoding="utf-8")
            if steps[name]["code"] != 0:
                steps[name]["errors"] = error_lines(full)
        print(f"== {name} (code {steps[name]['code']}, {steps[name].get('seconds')} s)", flush=True)
        for line in steps[name].get("errors", []):  # type: ignore[union-attr]
            print("   ", line, flush=True)
        save()
        return steps[name]["code"] == 0

    report["parts"] = parts
    ok = step("pip_install", f"{py} -m pip install -q rocketsim==2.2.1 cmake ninja pybind11", 600)
    report["versions_after"] = py_versions()
    ok = step("meshes", f"{py} {scripts / 'fetch_meshes.py'} --dest {meshes}", 120) and ok
    build = f"{py} {scripts / 'build_rocketgpu.py'} --src {APP / 'vendor/src/RocketGPU'} --out {APP / 'vendor/build'}"

    def part_rocketsim() -> None:
        procs = ",".join(map(str, sorted({1, 2, 4, max(1, affinity // 2), affinity, affinity * 2})))
        step("bench_rocketsim", f"{bench} --procs {procs}", 600)
        step("bench_rocketsim_multistep", f"{bench} --procs 1 --arenas 256 --engine multi_step", 120)

    def part_rg_cpu() -> None:
        rg_ok["cpu"] = False
        if step("build_rocketgpu_cpu", f"{build} --variant cpu", remaining()):
            env = f"ROCKETGPU_PY_DIR={APP / 'vendor/build/rocketgpu-cpu/python'}"
            rg_ok["cpu"] = all(
                step(f"bench_rocketgpu_cpu_{n}", f"{env} {bench} --procs 1 --arenas {n} --engine rocketgpu", 120)
                for n in (1024, 4096)
            )

    def part_rg_cuda() -> None:
        if not has_gpu:
            rg_ok["cuda"] = "pas de GPU"
            return
        rg_ok["cuda"] = False
        if step("build_rocketgpu_cuda", f"{build} --variant cuda", remaining()):
            env = f"ROCKETGPU_PY_DIR={APP / 'vendor/build/rocketgpu-cuda/python'}"
            rg_ok["cuda"] = all(
                step(f"bench_rocketgpu_cuda_{n}", f"{env} {bench} --procs 1 --arenas {n} --engine rocketgpu", 180)
                for n in (4096, 16384)
            )

    def part_rg_cuda12() -> None:
        """Toolkit CUDA 12.9 installé par apt (dépôt officiel NVIDIA, Ubuntu 24.04) à côté du CUDA de l'image :
        contourne l'échec de cicc (CUDA 13.0, sm_75) constaté le 2026-11-11."""
        if not has_gpu:
            rg_ok["cuda12"] = "pas de GPU"
            return
        rg_ok["cuda12"] = False
        nvcc12 = Path(f"/usr/local/cuda-{CUDA12}/bin/nvcc")
        if not step("install_cuda12", CUDA12_INSTALL, 900) or not nvcc12.exists():
            return
        tag = "cu12"
        if step("build_rocketgpu_cuda12", f"{build} --variant cuda --nvcc {nvcc12} --suffix {tag}", remaining()):
            env = f"ROCKETGPU_PY_DIR={APP / f'vendor/build/rocketgpu-cuda-{tag}/python'}"
            rg_ok["cuda12"] = all(
                step(f"bench_rocketgpu_cuda12_{n}", f"{env} {bench} --procs 1 --arenas {n} --engine rocketgpu", 180)
                for n in (4096, 16384)
            )

    runners = {
        "rg_cuda": part_rg_cuda,
        "rg_cuda12": part_rg_cuda12,
        "rocketsim": part_rocketsim,
        "rg_cpu": part_rg_cpu,
    }
    if ok:
        for name in parts:
            runners[name]()

    save()
    summary = {"ok": report["ok"], "ok_rocketgpu": rg_ok, "accelerator": ACCELERATOR, "elapsed_s": report["elapsed_s"]}
    print(json.dumps(summary))
    for name, st in steps.items():
        print(f"== {name} (code {st['code']})")
        print(st["out"])
    return 0 if report["ok"] else 1  # échec visible dans `kaggle kernels status`


if __name__ == "__main__":
    sys.exit(main())
