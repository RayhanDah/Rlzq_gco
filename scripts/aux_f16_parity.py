"""Parité float16 des observations pour la tête auxiliaire sur un lot RÉEL (ADR-010 D8-5) : log softmax de la tête
(sur le tronc de π) calculé sur les obs float32 du lot et sur les mêmes obs arrondies en float16 puis relues en float32
(chemin exact du rejeu auxiliaire : affectation numpy dans un tableau float16, puis `.float()` torch,
`training.auxtask.AuxIteration`).

    python scripts/aux_f16_parity.py --ckpt <checkpoint de reprise d'un run ou banc int_aux> --out <x.json>

Le checkpoint fournit π (`trainer.learner.policy`), la tête (`trainer.learner.aux.head`) et le lot déjà collecté en
attente (`trainer.pending`, agents de `learn_mask`, ceux que la tâche ingère). Seuil (ADR-010 D8-5, celui de π en
ADR-009 D7-5) : max |Δ log softmax(tête)| ≤ 1e-2 sur les 3 classes ; une obs non finie après l'arrondi float16
(|x| > 65 504) fait échouer le test. Sinon (D3) : float32 et H = 75.

Sortie : JSON (commande, commit, arbre sale ou non, matériel, date, fichier et SHA-256, valeurs, PASS/FAIL), écrit en
mode exclusif dans `--out` (jamais écrasé). Lecture seule. CPU par défaut. Mémoire : le checkpoint est chargé en
entier (≈ 2 × sa taille, estimé, non mesuré). Code de sortie : 0 PASS, 1 FAIL, 2 erreur.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import sil_f16_parity as sfp  # noqa: E402
import torch  # noqa: E402

SCHEMA_VERSION = 1
MAX_DLOGP = 1e-2
"""ADR-010 D8-5 : max |log softmax(tête) float32 − float16|, 3 classes (seuil de π, ADR-009 D7-5)."""


def aux_f16_parity(
    trunk: torch.nn.Module,
    head: torch.nn.Module,
    obs: npt.NDArray[np.float32],
    *,
    chunk: int = 8192,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Écarts float32 / float16 de log softmax(tête(tronc(obs))) sur `obs` (n, D) ; `passed` = tous les critères."""
    if obs.ndim != 2 or obs.shape[0] == 0:
        raise ValueError(f"obs (n, D) non vide attendu, reçu {obs.shape}")
    if obs.dtype != np.float32:
        raise ValueError(f"obs float32 attendues, reçu {obs.dtype}")
    if not np.isfinite(obs).all():
        raise ValueError("obs float32 non finies dans le lot")
    dev = torch.device(device)
    trunk, head = trunk.to(dev).eval(), head.to(dev).eval()
    x16_all = sfp.to_f16_and_back(obs)
    nonfinite16 = int((~torch.isfinite(x16_all)).any(dim=1).sum())
    dmax, dsum, bad_rows = 0.0, 0.0, 0
    n = obs.shape[0]
    with torch.inference_mode():
        for s in range(0, n, chunk):
            x32 = torch.from_numpy(obs[s : s + chunk]).to(dev)
            x16 = x16_all[s : s + chunk].to(dev)
            d = (torch.log_softmax(head(trunk(x32)), -1) - torch.log_softmax(head(trunk(x16)), -1)).abs()
            bad = (~torch.isfinite(d)).any(dim=-1)
            bad_rows += int(bad.sum())
            d = d[~bad]
            if d.numel():
                dmax = max(dmax, float(d.max()))
                dsum += float(d.max(dim=-1).values.double().sum())
    checks = {
        "obs_f16_finite": nonfinite16 == 0,
        "outputs_finite": bad_rows == 0,
        "dlogp_head_max_le_1e-2": dmax <= MAX_DLOGP,
    }
    return {
        "n_obs": int(n),
        "obs_dim": int(obs.shape[1]),
        "obs_f16_nonfinite_rows": nonfinite16,
        "outputs_nonfinite_rows": bad_rows,
        "dlogp_head_max": dmax,
        "dlogp_head_mean_of_row_max": dsum / n,
        "thresholds": {"dlogp_head_max": MAX_DLOGP, "source": "ADR-010 D8-5"},
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="checkpoint de reprise d'un run ou banc int_aux")
    ap.add_argument("--out", type=Path, required=True, help="JSON de sortie (jamais écrasé)")
    ap.add_argument("--device", default="cpu", help="device du calcul (défaut cpu)")
    ap.add_argument("--max-rows", type=int, help="sous-échantillon régulier des obs (défaut : tout le lot)")
    args = ap.parse_args(argv)
    argv_used = list(sys.argv if argv is None else ["scripts/aux_f16_parity.py", *argv])
    if args.out.exists():
        print(f"{args.out} existe déjà : jamais écrasé", file=sys.stderr)
        return 2

    from models import PolicyNet
    from obs import obs_dim as schema_obs_dim
    from training.auxtask import AuxHead, policy_trunk
    from training.checkpoint import CheckpointError
    from training.rollout import RolloutBatch

    d = schema_obs_dim(2)  # 1v1 (ADR-005)
    try:
        state = sfp._checked_state(args.ckpt, d, sfp.TICK_SKIP)
        learner = (state.get("trainer") or {}).get("learner") or {}
        pweights, aux = learner.get("policy"), learner.get("aux")
        if not pweights or not isinstance(aux, dict) or not aux.get("head"):
            raise CheckpointError(f"{args.ckpt} : π ou tête auxiliaire absente (checkpoint de reprise int_aux requis)")
        hidden, n_in, n_out = sfp.mlp_hidden(pweights)
        policy = PolicyNet(n_in, n_out, hidden)
        policy.load_state_dict(pweights)
        head = AuxHead(hidden[-1])
        head.load_state_dict(aux["head"])
        batch = (state.get("trainer") or {}).get("pending")
        if not isinstance(batch, RolloutBatch):
            raise CheckpointError(f"{args.ckpt} : pas de lot en attente (`trainer.pending`)")
        files = {"ckpt": sfp._file_info(args.ckpt, state)}
        del state
    except (CheckpointError, OSError, ValueError, KeyError) as exc:
        print(f"chargement impossible : {type(exc).__name__} : {exc}", file=sys.stderr)
        return 2
    obs, _ = sfp.learned_obs(batch)
    if args.max_rows is not None and 0 < args.max_rows < obs.shape[0]:
        idx = np.linspace(0, obs.shape[0] - 1, args.max_rows).round().astype(np.int64)
        obs = obs[idx]
    res = aux_f16_parity(policy_trunk(policy), head, obs, device=args.device)
    report = {
        "schema_version": SCHEMA_VERSION,
        "label": "mesuré (lot réel : trainer.pending du checkpoint --ckpt)",
        "provenance": sfp.provenance(argv_used, args.device),
        "files": files,
        "obs_source": {"field": "trainer.pending", "learned_agents": int(np.asarray(batch.learn_mask).sum())},
        **res,
        "verdict": "PASS" if res["passed"] else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as f:  # mode exclusif : jamais d'écrasement
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(
        json.dumps({k: report[k] for k in ("verdict", "n_obs", "dlogp_head_max", "obs_f16_nonfinite_rows")}, indent=2)
    )
    print(f"[f16] résultat : {args.out}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consoles Windows en cp1252 par défaut
    raise SystemExit(main())
