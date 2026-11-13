"""Évalue deux politiques par matchs appariés (kickoff -> premier but, côtés échangés) et écrit un JSON avec provenance.

    python scripts/eval.py --a <checkpoint|random> --b <checkpoint|random> --episodes N --out resultat.json
        [--sprt-elo 50] [--device cpu] [--seed 0] [--mode sample|argmax] [--n-envs 64]

A est la politique évaluée, B la référence : l'Elo rapporté est celui de A sur B (parties décisives, nuls exclus).
`--episodes` est le nombre d'épisodes joués (arrondi à une paire) ET le plafond du SPRT. `--sprt-elo X` rejoue après
coup, dans l'ordre des paires, un SPRT H0 : Δ = 0 contre H1 : Δ = +X (α, β : `--sprt-alpha`, `--sprt-beta`) : la
décision et le nombre d'épisodes qu'elle aurait consommés sont rapportés, les épisodes suivants restent dans le résumé.
Le fichier de sortie est le SEUL fichier écrit ; il n'est jamais écrasé (erreur s'il existe).

Variables d'environnement RocketGPU : `ROCKETGPU_PY_DIR`, `ROCKETGPU_COLLISION_MESH_DIR` (et `ROCKETGPU_DLL_DIR`).
Code de sortie : 0 si le match est joué et écrit ; 2 sinon.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env import RocketGPUSimulator, Simulator, SimulatorUnavailableError  # noqa: E402
from env.batch_env import ObsBuilder  # noqa: E402
from eval.match import load_policy, play_match, sprt_on_match, summarize  # noqa: E402
from eval.sprt import SPRTConfig  # noqa: E402
from obs.schema import obs_dim as schema_obs_dim  # noqa: E402
from rewards import RewardConfig  # noqa: E402
from training.checkpoint import git_commit  # noqa: E402

SCHEMA_VERSION = 1
TICK_SKIP = 8


def rocketgpu_commit() -> str:
    """SHA du build RocketGPU (`build_info.json`, dossier parent de `ROCKETGPU_PY_DIR`), sinon "unknown"."""
    py_dir = os.environ.get("ROCKETGPU_PY_DIR", "")
    if not py_dir:
        return "unknown"
    try:
        sha = json.loads((Path(py_dir).parent / "build_info.json").read_text(encoding="utf-8")).get("sha")
    except (OSError, ValueError):
        return "unknown"
    return str(sha) if sha else "unknown"


def git_dirty(repo: Path) -> bool | None:
    """Arbre de travail modifié (fichiers suivis ou non suivis) ? None si git est indisponible."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, timeout=20, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(res.stdout.strip())


def provenance(command: list[str], seed: int, device: str) -> dict[str, Any]:
    return {
        "command": command,
        "seed": seed,
        "date_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(ROOT),
        "git_dirty": git_dirty(ROOT),
        "rocketgpu_commit": rocketgpu_commit(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "device": device,
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
        "thermal_state": "non contrôlé (aucune régulation ni mesure de température)",
    }


def evaluate(
    a_spec: str,
    b_spec: str,
    episodes: int,
    *,
    seed: int,
    device: str = "cpu",
    mode: Literal["sample", "argmax"] = "sample",
    n_envs: int = 64,
    sprt_elo: float | None = None,
    sprt_alpha: float = 0.05,
    sprt_beta: float = 0.05,
    n_boot: int = 2000,
    reward_config: RewardConfig | None = None,
    sim: Simulator | None = None,
    build_obs: ObsBuilder | None = None,
    obs_dim: int | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Joue le match et rend le rapport (dictionnaire JSON-sérialisable). `sim` : simulateur dédié (défaut : RocketGPU
    CPU, `n_envs` arènes) ; `build_obs`/`obs_dim` : pour un simulateur factice (tests)."""
    if mode not in ("sample", "argmax"):
        raise ValueError("mode : 'sample' ou 'argmax'")
    sim = sim if sim is not None else RocketGPUSimulator(n_envs, 2, TICK_SKIP)
    d = obs_dim if obs_dim is not None else schema_obs_dim(2)
    cfg = reward_config or RewardConfig()
    # graines distinctes par rôle : A et B ne partagent pas de flux aléatoire (même checkpoint des deux côtés = deux
    # instances indépendantes, ce qui est l'évaluation « contre soi-même gelé »)
    pol_a, info_a = load_policy(a_spec, mode=mode, seed=seed + 1, obs_dim=d, tick_skip=sim.tick_skip, device=device)
    pol_b, info_b = load_policy(b_spec, mode=mode, seed=seed + 2, obs_dim=d, tick_skip=sim.tick_skip, device=device)
    res = play_match(pol_a, pol_b, sim, episodes, seed=seed, reward_config=cfg, build_obs=build_obs, obs_dim=obs_dim)
    summary = summarize(res, n_boot=n_boot, seed=seed)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance(command if command is not None else list(sys.argv), seed, device),
        "config": {
            "episodes_requested": episodes,
            "episodes_played": res.n_episodes,
            "n_pairs": res.n_pairs,
            "n_envs": sim.n_envs,
            "tick_skip": sim.tick_skip,
            "mode": mode,
            "reward_config": cfg.to_dict(),
            "p_kick": 1.0,
            "unit": "épisode kickoff -> premier but ; troncature = nul ; paires de même graine, côtés échangés",
        },
        "policies": {"a": dataclasses.asdict(info_a), "b": dataclasses.asdict(info_b)},
        "summary": summary.as_dict(),
        "throughput": {
            "wall_s": res.wall_s,
            "episodes_per_s": res.n_episodes / res.wall_s,
            "env_steps": res.env_steps,
            "env_steps_per_s": res.env_steps / res.wall_s,
            "note": "steps d'arène (arènes à vide en fin de quota comprises), politiques et simulateur inclus",
        },
        "per_pair": {"outcome": res.outcome.tolist(), "ticks": res.ticks.tolist()},
    }
    if sprt_elo is not None:
        sprt_cfg = SPRTConfig(elo0=0.0, elo1=sprt_elo, alpha=sprt_alpha, beta=sprt_beta, max_episodes=res.n_episodes)
        report["sprt"] = {
            "h0_elo": sprt_cfg.elo0,
            "h1_elo": sprt_cfg.elo1,
            "alpha": sprt_cfg.alpha,
            "beta": sprt_cfg.beta,
            "max_episodes": sprt_cfg.max_episodes,
            "draw_model": "nuls ignorés (Davidson) : test sur les parties décisives, voir eval.sprt",
            "mode": "rejoué après coup dans l'ordre des paires, évalué aux frontières de paires",
            **sprt_on_match(res, sprt_cfg).as_dict(),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint (.pt) ou 'random' : politique évaluée")
    ap.add_argument("--b", required=True, help="checkpoint (.pt) ou 'random' : référence")
    ap.add_argument(
        "--episodes", type=int, required=True, help="épisodes joués (arrondi à une paire) = plafond du SPRT"
    )
    ap.add_argument("--sprt-elo", type=float, default=None, help="H1 : +X Elo (H0 : 0) ; absent : pas de SPRT")
    ap.add_argument("--sprt-alpha", type=float, default=0.05)
    ap.add_argument("--sprt-beta", type=float, default=0.05)
    ap.add_argument("--device", default="cpu", help="device torch des politiques (cpu, xpu, cuda)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["sample", "argmax"], default="sample", help="choix d'action des checkpoints")
    ap.add_argument("--n-envs", type=int, default=64, help="arènes simulées (pair)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True, help="fichier JSON de sortie (créé, jamais écrasé)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refus : {out} existe déjà (jamais écrasé)", file=sys.stderr)
        return 2
    try:
        report = evaluate(
            args.a,
            args.b,
            args.episodes,
            seed=args.seed,
            device=args.device,
            mode=args.mode,
            n_envs=args.n_envs,
            sprt_elo=args.sprt_elo,
            sprt_alpha=args.sprt_alpha,
            sprt_beta=args.sprt_beta,
            n_boot=args.n_boot,
            command=[sys.executable, *(argv if argv is not None else sys.argv)],
        )
    except SimulatorUnavailableError as exc:
        print(f"RocketGPU indisponible : {exc}", file=sys.stderr)
        return 2
    with open(out, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    s, t = report["summary"], report["throughput"]
    print(f"{s['n_episodes']} épisodes : {s['wins']}V {s['draws']}N {s['losses']}D ; Elo décisif {s['elo_decisive']}")
    print(f"débit : {t['episodes_per_s']:.1f} épisodes/s, {t['env_steps_per_s']:.0f} env-steps/s")
    if "sprt" in report:
        print(f"SPRT : {report['sprt']['decision']} après {report['sprt']['n_episodes']} épisodes")
    print(f"écrit : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
