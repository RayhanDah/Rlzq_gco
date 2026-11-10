"""Micro-benchmark RocketSim 1v1 : débit brut du simulateur, sans obs, sans politique, sans PPO.

Mesure un **plafond de collecte** (pas le simulateur seul) : chaque step = tirage numpy + pose en Python des contrôles
des 2 voitures, puis `arena.step(tick_skip)`. Reset au kickoff sur but ou après `--episode-s` secondes de jeu.
Les workers démarrent ensemble (barrière) et le débit est calculé sur la fenêtre réelle min(début) → max(fin).
Au-delà du nombre de cœurs logiques, la mesure est en sursouscription volontaire (le JSON donne `cpu_count`).
Unités rapportées :
- env-steps/s : steps d'arène (1 step = tick_skip ticks) ;
- agent-steps/s = 2 × env-steps/s en 1v1 self-play (les deux voitures produisent de l'expérience) ;
- ticks/s = env-steps/s × tick_skip.

Usage : python scripts/benchmark_sim.py --procs 1,2,4,8 --seconds 20 [--arenas 4] [--tick-skip 8] [--engine multi_step]
Écrit un JSON dans runs/bench/ (commit, matériel, date, arguments, résultats) et l'affiche.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESHES = ROOT / "vendor" / "collision_meshes"
_BARRIER: Any = None
_RS_INIT_DIR: str | None = None
ENGINES = ("step", "multi_step", "rocketgpu")
# RocketGPU (build local, non versionné) : dossier du module et dossier des DLL du compilateur (MinGW sous Windows).
RG_PY_DIR = os.environ.get("ROCKETGPU_PY_DIR", str(ROOT / "vendor" / "build" / "rocketgpu-gcc" / "python"))
RG_DLL_DIR = os.environ.get("ROCKETGPU_DLL_DIR", r"C:\ProgramData\mingw64\mingw64\bin")


def _init_worker(barrier: Any) -> None:
    global _BARRIER
    _BARRIER = barrier


def worker(args: tuple[int, float, int, int, float, str, str]) -> dict[str, float]:
    seed, seconds, n_arenas, tick_skip, episode_s, mesh_dir, engine = args
    if engine not in ENGINES:
        raise ValueError(f"engine inconnu : {engine}")
    if engine == "rocketgpu":
        return worker_rocketgpu(seed, seconds, n_arenas, tick_skip, episode_s, mesh_dir)
    import numpy as np
    import RocketSim as rs

    global _RS_INIT_DIR
    if _RS_INIT_DIR is None:  # rs.init lève « Already inited » au second appel dans un même processus
        rs.init(mesh_dir)
        _RS_INIT_DIR = mesh_dir
    elif _RS_INIT_DIR != mesh_dir:
        raise RuntimeError(f"RocketSim déjà initialisé avec {_RS_INIT_DIR}")
    rng = np.random.default_rng(seed)
    arenas, cars = [], []
    for i in range(n_arenas):
        a = rs.Arena(rs.GameMode.SOCCAR)
        cars.append((a.add_car(rs.Team.BLUE), a.add_car(rs.Team.ORANGE)))
        a.reset_kickoff(seed * 1000 + i)
        arenas.append(a)
    max_ticks = int(episode_s * 120)
    ep_ticks = [0] * n_arenas
    scores = [0] * n_arenas
    steps = goals = resets = 0
    controls = rs.CarControls()
    if _BARRIER is not None:
        _BARRIER.wait(timeout=300)  # tous les workers ont fini leur init avant le départ
    t_start = time.time()
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        for _ in range(64):  # vérifier l'horloge toutes les 64 itérations
            axes = rng.uniform(-1.0, 1.0, (n_arenas, 2, 5)).tolist()
            bits = (rng.random((n_arenas, 2, 3)) < 0.5).tolist()
            for i in range(n_arenas):
                for k, car in enumerate(cars[i]):
                    controls.throttle, controls.steer, controls.pitch, controls.yaw, controls.roll = axes[i][k]
                    controls.jump, controls.boost, controls.handbrake = bits[i][k]
                    car.set_controls(controls)
            if engine == "multi_step":
                rs.Arena.multi_step(arenas, tick_skip)  # C++ multithreadé sur les arènes
            else:
                for a in arenas:
                    a.step(tick_skip)
            for i, a in enumerate(arenas):
                ep_ticks[i] += tick_skip
                s = a.blue_score + a.orange_score
                if s != scores[i] or ep_ticks[i] >= max_ticks:
                    goals += s != scores[i]
                    scores[i] = s
                    ep_ticks[i] = 0
                    resets += 1
                    a.reset_kickoff(int(rng.integers(1 << 30)))
                    scores[i] = a.blue_score + a.orange_score  # robuste si reset_kickoff remet le score à 0
            steps += n_arenas
    elapsed = time.perf_counter() - t0
    t_end = time.time()
    # Garde de cohérence : une voiture qui traverse le sol signale des meshes non chargés.
    min_z = min(float(c.get_state().pos.z) for pair in cars for c in pair)
    return {
        "env_steps": steps,
        "elapsed_s": elapsed,
        "t_start": t_start,
        "t_end": t_end,
        "goals": goals,
        "resets": resets,
        "min_car_z": min_z,
    }


def _import_rocketgpu() -> tuple[Any, Any]:
    if os.name == "nt" and Path(RG_DLL_DIR).is_dir():
        os.add_dll_directory(RG_DLL_DIR)
    if RG_PY_DIR not in sys.path:
        sys.path.insert(0, RG_PY_DIR)
    import rocketgpu_py as rg
    import rocketgpu_py_cuda as views

    return rg, views


def worker_rocketgpu(
    seed: int, seconds: float, n_envs: int, tick_skip: int, episode_s: float, mesh_dir: str
) -> dict[str, float]:
    """Même charge que `worker` (1v1, contrôles aléatoires tenus `tick_skip` ticks, reset au but ou au timeout),
    mais en un appel batché : contrôles en tableau numpy, états lus par vue numpy, sans objet Python par voiture."""
    import numpy as np

    os.environ["ROCKETGPU_COLLISION_MESH_DIR"] = str(Path(mesh_dir).resolve())
    rg, views = _import_rocketgpu()
    cfg = rg.SimConfig()
    cfg.env_count = n_envs
    cfg.max_cars_per_env = 2
    sim = rg.BatchSim(cfg)
    sim.set_env_car_count([2] * n_envs)
    sim.set_teams([rg.Team.Blue, rg.Team.Orange] * n_envs)
    sim.reset_to_kickoff(seed + 1)
    commands = [rg.EnvCommand() for _ in range(n_envs)]
    max_ticks = int(episode_s * 120)
    if rg.is_cuda_build:
        return _rocketgpu_cuda_loop(rg, views, sim, seed, seconds, n_envs, tick_skip, max_ticks, commands)
    ctrl = np.zeros(sim.car_capacity(), dtype=views.state_dtype("controls"))
    env = views.host_struct_array(sim, "env_states")
    cars = views.host_struct_array(sim, "car_states")
    rng = np.random.default_rng(seed)
    ep_ticks = np.zeros(n_envs, dtype=np.int64)
    scores = env["blue_score"].astype(np.int64) + env["orange_score"]
    steps = goals = resets = 0
    axes = ("throttle", "steer", "pitch", "yaw", "roll")
    bits = ("jump", "boost", "handbrake")
    t_start = time.time()
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        for _ in range(16):
            for name in axes:
                ctrl[name] = rng.uniform(-1.0, 1.0, ctrl.shape[0])
            for name in bits:
                ctrl[name] = rng.random(ctrl.shape[0]) < 0.5
            sim.step_with_device_controls(ctrl.ctypes.data, tick_skip, 0)  # stride 0 : mêmes contrôles à chaque tick
            ep_ticks += tick_skip
            s = env["blue_score"].astype(np.int64) + env["orange_score"]
            scored = s != scores
            done = np.flatnonzero(scored | (ep_ticks >= max_ticks))
            if done.size:
                goals += int(scored.sum())
                for i in done:
                    commands[i].reset_to_kickoff = 1
                sim.apply_env_commands(commands)
                for i in done:
                    commands[i].reset_to_kickoff = 0
                ep_ticks[done] = 0
                resets += int(done.size)
                scores = env["blue_score"].astype(np.int64) + env["orange_score"]
            steps += n_envs
    elapsed = time.perf_counter() - t0
    t_end = time.time()
    return {
        "env_steps": steps,
        "elapsed_s": elapsed,
        "t_start": t_start,
        "t_end": t_end,
        "goals": goals,
        "resets": resets,
        "min_car_z": float(cars["pos"][:, 2].min()),
    }


class _CudaArray:
    """Expose un pointeur device au protocole `__cuda_array_interface__` (lu par torch.as_tensor)."""

    def __init__(self, cai: dict[str, Any]) -> None:
        self.__cuda_array_interface__ = cai


CONTROL_OFFSETS = {
    "throttle": 0,
    "steer": 4,
    "pitch": 8,
    "yaw": 12,
    "roll": 16,
    "jump": 20,
    "boost": 21,
    "handbrake": 22,
}


def _rocketgpu_cuda_loop(
    rg: Any,
    views: Any,
    sim: Any,
    seed: int,
    seconds: float,
    n_envs: int,
    tick_skip: int,
    max_ticks: int,
    commands: list[Any],
) -> dict[str, float]:
    """Build CUDA : contrôles tirés sur le GPU (torch), scores lus sur le GPU, seuls les indices de reset passent
    par l'hôte. NON TESTÉ en local (pas de CUDA sur le laptop) : première exécution prévue dans la sonde Kaggle T4.
    Le débit obtenu est une borne basse : synchronisation hôte à chaque step (indices de reset) et liste complète
    d'EnvCommand reconvertie à chaque reset."""
    import torch

    dt = views.state_dtype("controls")
    offsets = {n: dt.fields[n][1] for n in dt.names}
    if dt.itemsize != 24 or offsets != CONTROL_OFFSETS:
        raise RuntimeError(f"layout CarControls inattendu : {offsets}, itemsize {dt.itemsize}")
    dev = torch.device("cuda")
    cap = sim.car_capacity()
    ctrl = torch.zeros((cap, 6), dtype=torch.float32, device=dev)  # 24 octets par voiture, comme CarControls
    raw = ctrl.view(torch.uint8)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    fields = views.field_arrays(sim, "env_states")

    def as_i32(name: str) -> Any:
        cai = dict(fields[name].__cuda_array_interface__)
        cai["typestr"] = "<i4"  # scores < 2**31 ; torch ne lit pas toujours uint32
        return torch.as_tensor(_CudaArray(cai), device=dev)

    blue, orange = as_i32("blue_score"), as_i32("orange_score")
    ep = torch.zeros(n_envs, dtype=torch.int64, device=dev)
    scores = blue.long() + orange.long()
    steps = goals = resets = 0
    torch.cuda.synchronize()
    t_start = time.time()
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        for _ in range(16):
            ctrl[:, :5].uniform_(-1.0, 1.0, generator=gen)
            raw[:, 20:23] = (torch.rand((cap, 3), device=dev, generator=gen) < 0.5).to(torch.uint8)
            sim.step_with_device_controls(ctrl.data_ptr(), tick_skip, 0)
            ep += tick_skip
            s = blue.long() + orange.long()
            scored = s != scores
            mask = scored | (ep >= max_ticks)
            done = mask.nonzero().flatten().cpu().tolist()  # synchronise à chaque step
            if done:
                goals += int(scored.sum().item())
                for i in done:
                    commands[i].reset_to_kickoff = 1
                sim.apply_env_commands(commands)
                for i in done:
                    commands[i].reset_to_kickoff = 0
                ep[mask] = 0
                resets += len(done)
                scores = blue.long() + orange.long()
            steps += n_envs
    sim.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    t_end = time.time()
    min_z = min(float(c.pos.z) for c in sim.download_car_states(min(cap, 64)))
    return {
        "env_steps": steps,
        "elapsed_s": elapsed,
        "t_start": t_start,
        "t_end": t_end,
        "goals": goals,
        "resets": resets,
        "min_car_z": min_z,
    }


def aggregate(results: list[dict[str, float]], tick_skip: int) -> dict[str, float]:
    """Somme des steps divisée par la fenêtre réelle d'exécution : max(fin) - min(début).

    Sans horodatages (ancien format), retombe sur la durée du worker le plus lent.
    """
    steps = sum(r["env_steps"] for r in results)
    if all("t_start" in r and "t_end" in r for r in results):
        wall = max(r["t_end"] for r in results) - min(r["t_start"] for r in results)
    else:
        wall = max(r["elapsed_s"] for r in results)
    env_sps = steps / wall if wall > 0 else 0.0
    return {
        "env_steps": steps,
        "wall_s": round(wall, 3),
        "env_sps": round(env_sps, 1),
        "agent_sps": round(2 * env_sps, 1),
        "ticks_per_s": round(env_sps * tick_skip, 1),
        "goals": sum(r["goals"] for r in results),
        "min_car_z": min(r["min_car_z"] for r in results),
    }


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return sha.strip() + ("-dirty" if dirty.strip() else "")
    except OSError:
        return "inconnu"


def machine() -> dict[str, Any]:
    import numpy as np

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rocketsim": _dist_version("rocketsim"),
    }


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", default="1", help="liste de nombres de processus, ex. 1,2,4,8")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--arenas", type=int, default=4, help="arènes par processus")
    ap.add_argument("--engine", choices=ENGINES, default="step", help="step par arène ou multi_step (threads C++)")
    ap.add_argument("--tick-skip", type=int, default=8)
    ap.add_argument("--episode-s", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meshes", default=os.environ.get("RS_COLLISION_MESHES", str(DEFAULT_MESHES)))
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bench")
    args = ap.parse_args()

    if not (Path(args.meshes) / "soccar").is_dir():
        print(f"meshes absents : {args.meshes} (lancer scripts/fetch_meshes.py)", file=sys.stderr)
        return 1
    report: dict[str, Any] = {
        "kind": "benchmark_sim",
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "machine": machine(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": [],
    }
    for n in [int(x) for x in args.procs.split(",")]:
        base = (args.seconds, args.arenas, args.tick_skip, args.episode_s, args.meshes, args.engine)
        jobs = [(args.seed + k, *base) for k in range(n)]
        if n == 1:
            res = [worker(jobs[0])]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(n, initializer=_init_worker, initargs=(ctx.Barrier(n),)) as pool:
                res = pool.map(worker, jobs)
        row = {"procs": n, **aggregate(res, args.tick_skip)}
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        if row["min_car_z"] < -100:
            print("ALERTE : voiture sous le sol, meshes probablement non chargés", file=sys.stderr)
            return 1
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench_sim_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rapport : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
