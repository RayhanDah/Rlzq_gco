"""Micro-benchmark du learner PPO (torch) : inférence batchée et mise à jour PPO sur données synthétiques.

Réseaux de la config PPO v0 (`docs/01_papers.md` §2) : politique 3×256 → 90 logits, critique séparé 3×512.
Mesures :
- inférence : obs/s pour un forward politique + échantillonnage d'action, par taille de lot (chemin de collecte) ;
- update : timesteps consommés par seconde d'update PPO (lot B, `epochs` passes, minibatch M), soit B / durée.
Données aléatoires : le coût ne dépend pas des valeurs. `--obs-dim` est une estimation de la taille d'obs (non figée).

Usage : python scripts/bench_learner.py [--device cpu] [--threads 0] [--batch 50000] [--minibatch 25000] [--epochs 2]
Écrit un JSON dans runs/bench/ (commit, matériel, date, arguments, résultats).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_sim import git_commit, machine  # noqa: E402

N_ACTIONS = 90


def mlp(n_in: int, width: int, depth: int, n_out: int) -> Any:
    import torch.nn as nn

    layers: list[Any] = []
    d = n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def make_nets(obs_dim: int, device: Any) -> tuple[Any, Any]:
    return mlp(obs_dim, 256, 3, N_ACTIONS).to(device), mlp(obs_dim, 512, 3, 1).to(device)


def sync(device: Any) -> None:
    import torch

    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_inference(policy: Any, obs_dim: int, n: int, device: Any, seconds: float) -> float:
    import torch

    obs = torch.randn(n, obs_dim, device=device)
    with torch.inference_mode():
        for _ in range(3):
            torch.distributions.Categorical(logits=policy(obs)).sample()
        sync(device)
        calls = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            torch.distributions.Categorical(logits=policy(obs)).sample().cpu()
            calls += 1
        sync(device)
        return calls * n / (time.perf_counter() - t0)


def ppo_update(
    policy: Any,
    value: Any,
    opt: Any,
    data: dict[str, Any],
    epochs: int,
    minibatch: int,
    clip: float = 0.2,
    amp: str = "none",
) -> dict[str, float]:
    """Une itération PPO-clip complète : epochs × minibatches, avantages normalisés par minibatch, grad clip 0,5.

    Les statistiques ne sont lues qu'à la fin (une seule synchronisation hôte par itération, pas deux par minibatch).
    `amp` ∈ {none, bf16, fp16} : autocast des passes avant (poids et optimiseur restent en float32).
    """
    import contextlib

    import torch

    n = data["obs"].shape[0]
    dev_type = data["obs"].device.type
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if amp not in ("none", *dtypes):
        raise ValueError(f"amp inconnu : {amp}")
    last_loss = last_kl = None
    for _ in range(epochs):
        perm = torch.randperm(n, device=data["obs"].device)
        for i in range(0, n, minibatch):
            idx = perm[i : i + minibatch]
            # index_select plutôt que x[idx] : sur XPU (Arc, torch 2.15.0+xpu), x[idx] lève
            # UR_RESULT_ERROR_OUT_OF_RESOURCES (constaté le 2026-11-10) ; résultat identique sur CPU.
            mb = {k: v.index_select(0, idx) for k, v in data.items()}
            obs, act = mb["obs"], mb["act"]
            adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
            ctx = torch.autocast(dev_type, dtype=dtypes[amp]) if amp != "none" else contextlib.nullcontext()
            with ctx:
                logits, v_pred = policy(obs), value(obs)
            dist = torch.distributions.Categorical(logits=logits.float())
            logp = dist.log_prob(act)
            ratio = torch.exp(logp - mb["logp"])
            pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            v_loss = 0.5 * (v_pred.float().squeeze(-1) - mb["ret"]).pow(2).mean()
            loss = pg + v_loss - 0.01 * dist.entropy().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*policy.parameters(), *value.parameters()], 0.5)
            opt.step()
            last_loss, last_kl = loss.detach(), (mb["logp"] - logp).mean().detach()
    if last_loss is None or last_kl is None:
        return {"loss": 0.0, "kl": 0.0}
    return {"loss": float(last_loss), "kl": float(last_kl)}


def bench_update(
    policy: Any,
    value: Any,
    obs_dim: int,
    batch: int,
    minibatch: int,
    epochs: int,
    device: Any,
    iters: int,
    amp: str = "none",
    fused: bool = False,
) -> float:
    import torch

    params = [*policy.parameters(), *value.parameters()]
    opt = torch.optim.Adam(params, lr=2e-4, eps=1e-5, **({"fused": True} if fused else {}))
    with torch.no_grad():
        obs = torch.randn(batch, obs_dim, device=device)
        dist = torch.distributions.Categorical(logits=policy(obs))
        act = dist.sample()
        data = {
            "obs": obs,
            "act": act,
            "logp": dist.log_prob(act),
            "adv": torch.randn(batch, device=device),
            "ret": torch.randn(batch, device=device),
        }
    for _ in range(2):  # échauffement (et compilation éventuelle)
        ppo_update(policy, value, opt, data, 1, minibatch, amp=amp)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        ppo_update(policy, value, opt, data, epochs, minibatch, amp=amp)
    sync(device)
    return iters * batch / (time.perf_counter() - t0)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads ; 0 = défaut torch")
    ap.add_argument("--obs-dim", type=int, default=107, help="estimation, non figée")
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--minibatch", type=int, default=25_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--infer-sizes", default="256,1024,4096,8192")
    ap.add_argument("--infer-seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    ap.add_argument("--fused", action="store_true", help="Adam fused=True")
    ap.add_argument("--compile", action="store_true", help="torch.compile des deux réseaux")
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bench")
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    policy, value = make_nets(args.obs_dim, device)
    if args.compile:
        policy, value = torch.compile(policy), torch.compile(value)
    results: dict[str, Any] = {"threads": torch.get_num_threads(), "torch": torch.__version__}
    sizes = [] if args.skip_inference else args.infer_sizes.split(",")
    results["inference_obs_per_s"] = {
        n: round(bench_inference(policy, args.obs_dim, int(n), device, args.infer_seconds), 1) for n in sizes
    }
    print(json.dumps({"inference_obs_per_s": results["inference_obs_per_s"]}), flush=True)
    sps = bench_update(
        policy, value, args.obs_dim, args.batch, args.minibatch, args.epochs, device, args.iters, args.amp, args.fused
    )
    results["update_timesteps_per_s"] = round(sps, 1)
    results["update_seconds_per_iteration"] = round(args.batch / sps, 3)
    print(json.dumps({k: results[k] for k in ("update_timesteps_per_s", "update_seconds_per_iteration")}), flush=True)

    report = {
        "kind": "bench_learner",
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "machine": machine(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench_learner_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rapport : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
