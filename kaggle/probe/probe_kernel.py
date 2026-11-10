"""Kernel Kaggle « sonde » (≤ 15 min) : matériel réel de la session + débit brut RocketSim.

Généré par `scripts/kaggle_probe.py`, qui remplace EMBEDDED par le source de nos scripts (benchmark, meshes) et
COMMIT par le commit git local. Aucune donnée privée, aucun secret. Sortie : /kaggle/working/probe.json.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

EMBEDDED: dict[str, str] = {}  # remplacé au build : {"scripts/benchmark_sim.py": "...", ...}
COMMIT = "__COMMIT__"
ACCELERATOR = "__ACCELERATOR__"
WORK = Path("/kaggle/working")
APP = Path("/tmp/rlgco")  # hors de /kaggle/working : les meshes ne font pas partie des sorties
T0 = time.time()
BUDGET_S = 14 * 60


def run(cmd: str, timeout: int = 120) -> dict[str, object]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"code": r.returncode, "out": (r.stdout + r.stderr).strip()[-4000:]}
    except subprocess.TimeoutExpired:
        return {"code": -1, "out": f"timeout {timeout}s"}


def sh(cmd: str, timeout: int = 120) -> str:
    return str(run(cmd, timeout)["out"])


def py_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("numpy", "torch", "jax", "jaxlib", "torch_xla", "libtpu", "rocketsim"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "absent"
    return out


def main() -> int:
    start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report: dict[str, object] = {"commit": COMMIT, "accelerator": ACCELERATOR, "start": start}
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count() or 1
    report["system"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "sched_affinity": affinity,
        "nproc": sh("nproc"),
        "lscpu": sh("lscpu"),
        "free_g": sh("free -g"),
        "df_working": sh("df -h /kaggle/working"),
        "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total --format=csv"),
        "cgroup_cpu_max": sh("cat /sys/fs/cgroup/cpu.max"),
    }
    report["versions_before"] = py_versions()
    report["jax_devices"] = sh(f"{sys.executable} -c 'import jax; print(jax.devices())'", timeout=180)

    for rel, src in EMBEDDED.items():
        p = APP / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    steps: dict[str, dict[str, object]] = {}
    steps["pip_install"] = run(f"{sys.executable} -m pip install -q rocketsim==2.2.1", timeout=300)
    report["versions_after"] = py_versions()
    steps["meshes"] = run(f"{sys.executable} {APP / 'scripts/fetch_meshes.py'}", timeout=120)

    procs = sorted({1, 2, 4, max(1, affinity // 2), affinity, affinity * 2})
    remaining = BUDGET_S - (time.time() - T0)
    seconds = max(5, min(20, int(remaining / (len(procs) + 2)) - 5))
    bench_cmd = (
        f"{sys.executable} {APP / 'scripts/benchmark_sim.py'} --procs {','.join(map(str, procs))} "
        f"--seconds {seconds} --meshes {APP / 'vendor/collision_meshes'} --out {WORK / 'bench'}"
    )
    report["bench_cmd"] = bench_cmd
    if all(st["code"] == 0 for st in steps.values()):
        steps["bench"] = run(bench_cmd, timeout=max(60, int(remaining)))
    report["steps"] = steps
    report["ok"] = all(st["code"] == 0 for st in steps.values()) and "bench" in steps
    report["elapsed_s"] = round(time.time() - T0, 1)
    (WORK / "probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "accelerator": ACCELERATOR, "elapsed_s": report["elapsed_s"]}))
    for name, st in steps.items():
        print(f"== {name} (code {st['code']})")
        print(st["out"])
    return 0 if report["ok"] else 1  # échec visible dans `kaggle kernels status`


if __name__ == "__main__":
    sys.exit(main())
