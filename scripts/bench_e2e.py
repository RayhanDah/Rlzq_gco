"""Prototype bout-en-bout (mesure M3 de l'ADR-infra) : collecte RocketGPU batchée + politique torch + PPO synchrone.

PROTOTYPE DE MESURE, pas le code d'entraînement (Phase 3) :
- obs provisoire de 89 dimensions par agent, calculée en numpy vectorisé : balle 9, soi 19, adversaire 19,
  action précédente 8, pads 34, avec miroir bleu↔orange par rotation de 180° autour de z (ordre des pads non
  remappé : simplification assumée). Ni la taille ni le contenu ne sont figés ;
- table de 90 actions discrètes **reconstruite pour la mesure**, sans parité vérifiée avec RLGym (Phase 3,
  obs-action-engineer) : seul son coût de gather compte ici ;
- récompense GCO minimale (±1 par équipe au but, 0 sinon), terminaison au but, troncature à `--episode-s` ;
- PPO v0 (`docs/01_papers.md` §2) : π 3×256, V 3×512, GAE γ=0,99 λ=0,95, 2 epochs, minibatch 25k.

Mesure, par itération : temps de collecte (obs, inférence, sim, comptabilité), temps d'apprentissage (valeurs, GAE,
update), et débit bout-en-bout = timesteps agents / (collecte + apprentissage), synchrone, sans recouvrement.
Usage : python scripts/bench_e2e.py [--envs 1024] [--steps 25] [--iters 4] [--device cpu] [--threads 0]
Avec torch XPU : vendor/venv-xpu/Scripts/python scripts/bench_e2e.py --device xpu
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_learner import make_nets, ppo_update, sync  # noqa: E402
from benchmark_sim import _import_rocketgpu, git_commit, machine  # noqa: E402

OBS_DIM = 89
POS_SCALE = np.array([4096.0, 5120.0, 2044.0], dtype=np.float32)
VEL_SCALE, ANGVEL_SCALE = 2300.0, 5.5
CONTROL_FIELDS = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
MIRROR = np.array([-1.0, -1.0, 1.0], dtype=np.float32)  # rotation de 180° autour de z


def action_table() -> np.ndarray:
    """90 actions discrètes [throttle, steer, pitch, yaw, roll, jump, boost, handbrake] (prototype, voir docstring)."""
    rows: list[list[float]] = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    rows.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = int(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        rows.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    table = np.asarray(rows, dtype=np.float32)
    if table.shape != (90, 8):
        raise AssertionError(f"table d'actions : {table.shape}")
    return table


def quat_axes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quaternions (w, x, y, z) → axes avant (R·x̂) et haut (R·ẑ), vectorisé."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    fwd = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=-1)
    up = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=-1)
    return fwd, up


def car_features(cars: np.ndarray, flip: np.ndarray) -> np.ndarray:
    """19 features par voiture ; `flip` (n, 1) vaut −1 pour les voitures vues depuis le camp orange."""
    m = np.where(flip < 0, MIRROR, 1.0).astype(np.float32)
    fwd, up = quat_axes(cars["rot"].astype(np.float32))
    can_flip = (cars["has_flipped"] == 0) & (cars["has_double_jumped"] == 0)
    return np.concatenate(
        [
            cars["pos"] * m / POS_SCALE,
            fwd * m,
            up * m,
            cars["vel"] * m / VEL_SCALE,
            cars["ang_vel"] * m / ANGVEL_SCALE,
            (cars["boost"] / 100.0)[:, None],
            cars["is_on_ground"][:, None],
            can_flip[:, None],
            cars["is_demoed"][:, None],
        ],
        axis=1,
        dtype=np.float32,
    )


def build_obs(cars: np.ndarray, balls: np.ndarray, pads: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Obs (2·n_envs, 89). Agents rangés [env0 bleu, env0 orange, env1 bleu, ...] comme les slots de voitures."""
    n_agents = cars.shape[0]
    flip = np.tile(np.array([1.0, -1.0], dtype=np.float32), n_agents // 2)[:, None]
    m = np.where(flip < 0, MIRROR, 1.0).astype(np.float32)
    ball = np.repeat(balls, 2)
    ball_f = np.concatenate(
        [ball["pos"] * m / POS_SCALE, ball["vel"] * m / VEL_SCALE, ball["ang_vel"] * m / ANGVEL_SCALE], axis=1
    )
    me = car_features(cars, flip)
    opp_idx = np.arange(n_agents) ^ 1
    opp = car_features(cars[opp_idx], flip)
    pad_f = np.repeat(pads["is_active"].reshape(n_agents // 2, -1).astype(np.float32), 2, axis=0)
    return np.concatenate([ball_f, me, opp, prev, pad_f], axis=1, dtype=np.float32)


def gae(rew: np.ndarray, val: np.ndarray, last_val: np.ndarray, term: np.ndarray, trunc: np.ndarray,
        gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """GAE (T, A). `term` : pas de bootstrap ; `trunc` : bootstrap simplifié sur V(s_t) (prototype de mesure)."""
    t_len = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1], dtype=np.float32)
    for t in reversed(range(t_len)):
        nxt = last_val if t == t_len - 1 else val[t + 1]
        nxt = np.where(trunc[t], val[t], nxt)
        nonterm = 1.0 - term[t]
        cont = nonterm * (1.0 - trunc[t])
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        last = delta + gamma * lam * cont * last
        adv[t] = last
    return adv, adv + val


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25, help="steps par itération ; timesteps = envs × 2 × steps")
    ap.add_argument("--iters", type=int, default=4, help="la première itération (échauffement) est exclue")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--minibatch", type=int, default=25_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tick-skip", type=int, default=8)
    ap.add_argument("--episode-s", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meshes", default=str(ROOT / "vendor" / "collision_meshes"))
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bench")
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    os.environ["ROCKETGPU_COLLISION_MESH_DIR"] = str(Path(args.meshes).resolve())
    rg, views = _import_rocketgpu()

    n_env, n_ag = args.envs, 2 * args.envs
    cfg = rg.SimConfig()
    cfg.env_count = n_env
    cfg.max_cars_per_env = 2
    sim = rg.BatchSim(cfg)
    sim.set_env_car_count([2] * n_env)
    sim.set_teams([rg.Team.Blue, rg.Team.Orange] * n_env)
    sim.reset_to_kickoff(args.seed + 1)
    cars = views.host_struct_array(sim, "car_states")
    balls = views.host_struct_array(sim, "ball_states")
    pads = views.host_struct_array(sim, "boost_pad_states")
    env = views.host_struct_array(sim, "env_states")
    if pads.shape[0] != 34 * n_env:
        raise RuntimeError(f"{pads.shape[0]} pads pour {n_env} arènes, 34 attendus par arène")
    ctrl = np.zeros(sim.car_capacity(), dtype=views.state_dtype("controls"))
    commands = [rg.EnvCommand() for _ in range(n_env)]
    table = action_table()
    policy, value = make_nets(OBS_DIM, device)
    opt = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=2e-4, eps=1e-5)

    prev = np.zeros((n_ag, 8), dtype=np.float32)
    ep_ticks = np.zeros(n_env, dtype=np.int64)
    scores = env["blue_score"].astype(np.int64), env["orange_score"].astype(np.int64)
    max_ticks = int(args.episode_s * 120)
    team_sign = np.tile(np.array([1.0, -1.0], dtype=np.float32), n_env)
    rows: list[dict[str, Any]] = []
    goals_total = 0

    for it in range(args.iters):
        t_parts = dict.fromkeys(("obs", "infer", "sim", "book", "learn"), 0.0)
        buf: dict[str, list[np.ndarray]] = {k: [] for k in ("obs", "act", "logp", "rew", "term", "trunc")}
        for _ in range(args.steps):
            t = time.perf_counter()
            obs = build_obs(cars, balls, pads, prev)
            t_parts["obs"] += time.perf_counter() - t
            t = time.perf_counter()
            with torch.inference_mode():
                dist = torch.distributions.Categorical(logits=policy(torch.from_numpy(obs).to(device)))
                act_t = dist.sample()
                logp = dist.log_prob(act_t).cpu().numpy()
                act = act_t.cpu().numpy()
            t_parts["infer"] += time.perf_counter() - t
            t = time.perf_counter()
            chosen = table[act]
            for j, name in enumerate(CONTROL_FIELDS):
                ctrl[name] = chosen[:, j]
            sim.step_with_device_controls(ctrl.ctypes.data, args.tick_skip, 0)
            t_parts["sim"] += time.perf_counter() - t
            t = time.perf_counter()
            ep_ticks += args.tick_skip
            blue, orange = env["blue_score"].astype(np.int64), env["orange_score"].astype(np.int64)
            d_blue, d_orange = blue - scores[0], orange - scores[1]
            scored = (d_blue + d_orange) > 0
            r_env = (d_blue - d_orange).astype(np.float32)  # +1 si but bleu, −1 si but orange
            rew = np.repeat(r_env, 2) * team_sign
            trunc_env = (~scored) & (ep_ticks >= max_ticks)
            done = np.flatnonzero(scored | trunc_env)
            if done.size:
                for i in done:
                    commands[i].reset_to_kickoff = 1
                sim.apply_env_commands(commands)
                for i in done:
                    commands[i].reset_to_kickoff = 0
                ep_ticks[done] = 0
                goals_total += int(scored.sum())
            scores = env["blue_score"].astype(np.int64), env["orange_score"].astype(np.int64)
            prev = chosen
            prev[np.repeat(scored | trunc_env, 2)] = 0.0
            for k, v in (("obs", obs), ("act", act), ("logp", logp), ("rew", rew)):
                buf[k].append(v)
            buf["term"].append(np.repeat(scored, 2).astype(np.float32))
            buf["trunc"].append(np.repeat(trunc_env, 2).astype(np.float32))
            t_parts["book"] += time.perf_counter() - t

        t = time.perf_counter()
        obs_all = np.stack(buf["obs"])  # (T, A, D)
        last_obs = build_obs(cars, balls, pads, prev)
        with torch.no_grad():
            v_all = value(torch.from_numpy(obs_all.reshape(-1, OBS_DIM)).to(device)).squeeze(-1).cpu().numpy()
            v_last = value(torch.from_numpy(last_obs).to(device)).squeeze(-1).cpu().numpy()
        val = v_all.reshape(args.steps, n_ag)
        adv, ret = gae(np.stack(buf["rew"]), val, v_last, np.stack(buf["term"]), np.stack(buf["trunc"]), 0.99, 0.95)
        data = {
            "obs": torch.from_numpy(obs_all.reshape(-1, OBS_DIM)).to(device),
            "act": torch.from_numpy(np.stack(buf["act"]).reshape(-1)).to(device),
            "logp": torch.from_numpy(np.stack(buf["logp"]).reshape(-1)).to(device),
            "adv": torch.from_numpy(adv.reshape(-1)).to(device),
            "ret": torch.from_numpy(ret.reshape(-1)).to(device),
        }
        stats = ppo_update(policy, value, opt, data, args.epochs, args.minibatch)
        sync(device)
        t_parts["learn"] = time.perf_counter() - t

        timesteps = args.steps * n_ag
        collect = sum(v for k, v in t_parts.items() if k != "learn")
        row = {
            "iter": it,
            "timesteps": timesteps,
            "collect_s": round(collect, 3),
            "learn_s": round(t_parts["learn"], 3),
            "collect_sps": round(timesteps / collect, 1),
            "learn_sps": round(timesteps / t_parts["learn"], 1),
            "e2e_sps": round(timesteps / (collect + t_parts["learn"]), 1),
            "parts_s": {k: round(v, 3) for k, v in t_parts.items()},
            "loss": round(stats["loss"], 4),
            "finite": bool(np.isfinite(stats["loss"]) and np.isfinite(obs_all).all()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    kept = rows[1:] or rows
    summary = {k: round(float(np.mean([r[k] for r in kept])), 1) for k in ("collect_sps", "learn_sps", "e2e_sps")}
    summary["goals_total"] = goals_total
    print(json.dumps({"summary_excluding_warmup": summary}))
    report = {
        "kind": "bench_e2e",
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "machine": machine(),
        "torch": {"version": torch.__version__, "threads": torch.get_num_threads(), "device": str(device)},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "iterations": rows,
        "summary_excluding_warmup": summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench_e2e_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rapport : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
