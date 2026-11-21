"""Parité float16 des observations sur un lot RÉEL (ADR-009 D7-5) : log π et V calculés sur les obs float32 du lot et
sur les mêmes obs arrondies en float16 puis relues en float32 (chemin exact du buffer SIL : affectation numpy dans un
tableau float16, puis `torch.from_numpy(...).float()`, `training.sil.ReplayBuffer.batch`).

    python scripts/sil_f16_parity.py --policy <ckpt|instantané> [--value <ckpt>] [--obs-from <ckpt>] --out <x.json>

- `--policy` : checkpoint de reprise ou instantané d'évaluation (poids de π seuls), chargé par
  `eval.match.load_checkpoint_policy` (somme de contrôle, identité obs/actions/tick-skip vérifiées) ;
- `--value` : checkpoint de reprise qui contient le critique (`trainer.learner.value`) ; défaut : `--policy` ;
- `--obs-from` : checkpoint de reprise dont on lit le lot déjà collecté en attente (`trainer.pending`, `RolloutBatch`
  de `collection.steps_per_batch` × agents, obs float32 de la vraie collecte), agents appris seulement (`learn_mask` :
  ceux que la SIL ingère) ; défaut : `--value`. Si ce fichier contient aussi π, l'égalité de ses poids à ceux de
  `--policy` est rapportée (instantané et checkpoint de la même itération).

Seuils (ADR-009 D7-5) : max |Δ log π| ≤ 1e-2 sur TOUTES les actions (plus strict que sur la seule action jouée, aussi
rapportée) et max |ΔV| ≤ 1e-3 ; une obs non finie après l'arrondi float16 (|x| > 65 504) fait échouer le test.
Sortie : JSON (commande, commit, arbre sale ou non, matériel, date, fichiers et SHA-256, valeurs, PASS/FAIL), écrit
en mode exclusif dans `--out` (jamais écrasé : erreur s'il existe). Lecture seule des checkpoints. CPU par défaut.
Mémoire : un checkpoint est chargé en entier (octets + état décodé, ≈ 2 × sa taille, estimé) ; le même fichier
passé à plusieurs options est relu (≈ 0,8 Go par lecture pour un checkpoint de banc SIL, estimé, non mesuré).

Code de sortie : 0 PASS, 1 FAIL, 2 erreur (fichier absent, incompatible, sans critique ou sans lot en attente).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import torch  # noqa: E402

SCHEMA_VERSION = 1
MAX_DLOGP = 1e-2
"""ADR-009 D7-5 : max |log π(·|s) float32 − log π(·|s) float16|, toutes actions."""
MAX_DV = 1e-3
"""ADR-009 D7-5 : max |V(s) float32 − V(s) float16|."""
F16_MAX = float(np.finfo(np.float16).max)
TICK_SKIP = 8  # configs/*.yaml, `scripts/eval.py` (TICK_SKIP) ; vérifié contre l'identité du checkpoint


# ====================================================================================================== partie pure


def to_f16_and_back(obs: npt.NDArray[np.float32]) -> torch.Tensor:
    """Obs telles que la SIL les relit : tableau float16 (arrondi au plus proche, numpy) puis `.float()` torch."""
    stored = np.empty(obs.shape, dtype=np.float16)
    with np.errstate(over="ignore"):
        stored[...] = obs  # même affectation que `ReplayBuffer.add` (dépassement -> inf, compté ensuite)
    return torch.from_numpy(stored).float()


def f16_parity(
    policy: torch.nn.Module,
    value: torch.nn.Module,
    obs: npt.NDArray[np.float32],
    act: npt.NDArray[np.int64] | None = None,
    *,
    chunk: int = 8192,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Écarts float32 / float16 sur `obs` (n, D) : log π (toutes actions, et action jouée si `act`), V, arrondi des
    obs. Seuils de D7-5 appliqués ; `passed` = tous les critères vrais."""
    if obs.ndim != 2 or obs.shape[0] == 0:
        raise ValueError(f"obs (n, D) non vide attendu, reçu {obs.shape}")
    if obs.dtype != np.float32:
        raise ValueError(f"obs float32 attendues, reçu {obs.dtype}")
    if not np.isfinite(obs).all():
        raise ValueError("obs float32 non finies dans le lot")
    dev = torch.device(device)
    policy, value = policy.to(dev).eval(), value.to(dev).eval()
    x16_all = to_f16_and_back(obs)
    nonfinite16 = int((~torch.isfinite(x16_all)).any(dim=1).sum())
    obs_err = (x16_all - torch.from_numpy(obs)).abs()
    obs_err = obs_err[torch.isfinite(obs_err)]
    if obs_err.numel() == 0:
        obs_err = torch.zeros(1)
    dlogp_all = dlogp_act = dv = 0.0
    sum_dlogp = sum_dv = 0.0
    nonfinite_out = 0  # lignes où un écart est NaN ou infini (max() de Python ignorerait un NaN : compté à part)
    n = obs.shape[0]
    with torch.inference_mode():
        for s in range(0, n, chunk):
            x32 = torch.from_numpy(obs[s : s + chunk]).to(dev)
            x16 = x16_all[s : s + chunk].to(dev)
            lp32 = torch.log_softmax(policy(x32), dim=-1)
            lp16 = torch.log_softmax(policy(x16), dim=-1)
            d = (lp32 - lp16).abs()
            dvs = (value(x32) - value(x16)).abs()
            bad = (~torch.isfinite(d)).any(dim=-1) | ~torch.isfinite(dvs)
            nonfinite_out += int(bad.sum())
            d, dvs = d[~bad], dvs[~bad]
            if d.numel():
                dlogp_all = max(dlogp_all, float(d.max()))
                sum_dlogp += float(d.max(dim=-1).values.double().sum())
                dv = max(dv, float(dvs.max()))
                sum_dv += float(dvs.double().sum())
            if act is not None and d.numel():
                a = torch.from_numpy(np.ascontiguousarray(act[s : s + chunk], dtype=np.int64)).to(dev)[~bad]
                dlogp_act = max(dlogp_act, float(d.gather(-1, a.unsqueeze(-1)).max()))
    checks = {
        "obs_f16_finite": nonfinite16 == 0,
        "outputs_finite": nonfinite_out == 0,
        "dlogp_all_max_le_1e-2": dlogp_all <= MAX_DLOGP,
        "dv_max_le_1e-3": dv <= MAX_DV,
    }
    return {
        "n_obs": int(n),
        "obs_dim": int(obs.shape[1]),
        "obs_abs_max": float(np.abs(obs).max()),
        "obs_f16_nonfinite_rows": nonfinite16,
        "outputs_nonfinite_rows": nonfinite_out,
        "obs_round_abs_max": float(obs_err.max()),  # sur les composantes restées finies
        "dlogp_all_max": dlogp_all,
        "dlogp_all_mean_of_row_max": sum_dlogp / n,
        "dlogp_taken_max": dlogp_act if act is not None else None,
        "dv_max": dv,
        "dv_mean": sum_dv / n,
        "thresholds": {"dlogp_all_max": MAX_DLOGP, "dv_max": MAX_DV, "source": "ADR-009 D7-5"},
        "checks": checks,
        "passed": all(checks.values()),
    }


def learned_obs(batch: Any) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """(obs (T·n, D), actions (T·n,)) des agents appris (`learn_mask`) d'un `RolloutBatch`, dans l'ordre (t, agent)."""
    mask = np.asarray(batch.learn_mask, dtype=bool)
    obs = np.asarray(batch.obs)[:, mask].reshape(-1, batch.obs.shape[2])
    act = np.asarray(batch.act)[:, mask].reshape(-1)
    return np.ascontiguousarray(obs, dtype=np.float32), np.ascontiguousarray(act, dtype=np.int64)


def mlp_hidden(weights: Mapping[str, torch.Tensor]) -> tuple[tuple[int, ...], int, int]:
    """(largeurs cachées, entrée, sortie) d'un `build_mlp` d'après les formes de ses poids (`net.<i>.weight`)."""
    layers = sorted(int(k.split(".")[1]) for k in weights if k.startswith("net.") and k.endswith(".weight"))
    if not layers:
        raise ValueError("poids `net.<i>.weight` introuvables")
    hidden = tuple(int(weights[f"net.{i}.weight"].shape[0]) for i in layers[:-1])
    return hidden, int(weights[f"net.{layers[0]}.weight"].shape[1]), int(weights[f"net.{layers[-1]}.weight"].shape[0])


def same_weights(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


# ====================================================================================================== chargement


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _checked_state(path: Path, obs_dim: int, tick_skip: int) -> dict[str, Any]:
    """État du checkpoint (somme de contrôle vérifiée), identité obs/actions/tick-skip vérifiée comme dans
    `eval.match.load_checkpoint_policy` (les champs de récompense sont recopiés du fichier : sans effet sur π et V)."""
    from training.checkpoint import CheckpointIdentity, CheckpointManager, check_identity

    state = CheckpointManager.load(path)
    saved = state["identity"]
    expected = replace(
        CheckpointIdentity.current(
            reward_config_hash=saved["reward_config_hash"], obs_dim=obs_dim, tick_skip=tick_skip
        ),
        reward_spec_version=saved["reward_spec_version"],
        reward_hash_schema_version=saved["reward_hash_schema_version"],
    )
    check_identity(state, expected)
    return state


def _file_info(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "kind": state.get("kind", "checkpoint"),
        "iteration": state.get("iteration"),
        "git_commit": (state.get("versions") or {}).get("git_commit"),
        "train_config_hash": state.get("train_config_hash"),
    }


def _git(args: list[str]) -> str | None:
    try:
        res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode == 0 else None


def provenance(argv: list[str], device: str) -> dict[str, Any]:
    from training.checkpoint import git_commit

    status = _git(["status", "--porcelain"])
    return {
        "command": [sys.executable, *argv],
        "git_commit": git_commit(ROOT),
        "git_dirty": True if status is None else bool(status.strip()),
        "date_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": device,
        "torch_threads": torch.get_num_threads(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True, help="checkpoint de reprise ou instantané d'évaluation")
    ap.add_argument("--value", type=Path, help="checkpoint de reprise contenant le critique (défaut : --policy)")
    ap.add_argument("--obs-from", type=Path, help="checkpoint de reprise avec un lot en attente (défaut : --value)")
    ap.add_argument("--out", type=Path, required=True, help="JSON de sortie (jamais écrasé)")
    ap.add_argument("--device", default="cpu", help="device du calcul (défaut cpu)")
    ap.add_argument("--max-rows", type=int, help="sous-échantillon régulier des obs (défaut : tout le lot)")
    ap.add_argument("--all-agents", action="store_true", help="toutes les obs du lot, camp gelé compris")
    args = ap.parse_args(argv)
    argv_used = list(sys.argv if argv is None else ["scripts/sil_f16_parity.py", *argv])
    if args.out.exists():
        print(f"{args.out} existe déjà : jamais écrasé", file=sys.stderr)
        return 2
    value_path = args.value or args.policy
    obs_path = args.obs_from or value_path

    from eval.match import load_checkpoint_policy
    from models import ValueNet
    from obs import obs_dim as schema_obs_dim
    from training.checkpoint import CheckpointError
    from training.rollout import RolloutBatch

    d = schema_obs_dim(2)  # 1v1 (ADR-005)
    try:
        policy, pinfo = load_checkpoint_policy(
            args.policy, mode="argmax", seed=0, obs_dim=d, tick_skip=TICK_SKIP, device="cpu"
        )
        pnet = policy.net
        pweights = {k: v.detach().clone() for k, v in pnet.state_dict().items()}
        vstate = _checked_state(value_path, d, TICK_SKIP)
        vweights = ((vstate.get("trainer") or {}).get("learner") or {}).get("value")
        if not vweights:
            raise CheckpointError(f"{value_path} : pas de critique (instantané d'évaluation ?) : passer --value")
        hidden, n_in, n_out = mlp_hidden(vweights)
        if n_in != d or n_out != 1:
            raise CheckpointError(f"critique {n_in}->{n_out} : {d}->1 attendu")
        vnet = ValueNet(d, hidden)
        vnet.load_state_dict(vweights)
        files: dict[str, Any] = {
            "policy": {
                "path": str(args.policy),
                "bytes": args.policy.stat().st_size,
                "sha256": pinfo.sha256,
                "iteration": pinfo.iteration,
                "git_commit": pinfo.git_commit,
            },
            "value": _file_info(value_path, vstate),
        }
        ostate = vstate if obs_path == value_path else _checked_state(obs_path, d, TICK_SKIP)
        if obs_path != value_path:
            files["obs"] = _file_info(obs_path, ostate)
        del vstate
        batch = (ostate.get("trainer") or {}).get("pending")
        if not isinstance(batch, RolloutBatch):
            raise CheckpointError(f"{obs_path} : pas de lot en attente (`trainer.pending`) : autre checkpoint requis")
        other_pi = ((ostate.get("trainer") or {}).get("learner") or {}).get("policy")
        policy_equal = None if not other_pi else same_weights(pweights, other_pi)
        obs_iteration = ostate.get("iteration")
        del ostate
    except (CheckpointError, OSError, ValueError, KeyError) as exc:
        print(f"chargement impossible : {type(exc).__name__} : {exc}", file=sys.stderr)
        return 2

    if args.all_agents:
        obs = np.ascontiguousarray(batch.obs.reshape(-1, batch.obs.shape[2]), dtype=np.float32)
        act = np.ascontiguousarray(batch.act.reshape(-1), dtype=np.int64)
    else:
        obs, act = learned_obs(batch)
    if args.max_rows is not None and 0 < args.max_rows < obs.shape[0]:
        idx = np.linspace(0, obs.shape[0] - 1, args.max_rows).round().astype(np.int64)
        obs, act = obs[idx], act[idx]
    res = f16_parity(pnet, vnet, obs, act, device=args.device)
    report = {
        "schema_version": SCHEMA_VERSION,
        "label": "mesuré (lot réel : trainer.pending du checkpoint --obs-from)",
        "provenance": provenance(argv_used, args.device),
        "files": files,
        "obs_source": {
            "field": "trainer.pending",
            "iteration": obs_iteration,
            "steps": int(batch.obs.shape[0]),
            "agents": int(batch.obs.shape[1]),
            "learned_agents": int(np.asarray(batch.learn_mask).sum()),
            "agents_used": "tous" if args.all_agents else "appris (learn_mask)",
            "max_rows": args.max_rows,
        },
        "policy_equals_obs_file_policy": policy_equal,
        **res,
        "verdict": "PASS" if res["passed"] else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as f:  # mode exclusif : jamais d'écrasement
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    keys = ("verdict", "n_obs", "dlogp_all_max", "dlogp_taken_max", "dv_max", "obs_f16_nonfinite_rows")
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    print(f"[f16] résultat : {args.out}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consoles Windows en cp1252 par défaut
    raise SystemExit(main())
