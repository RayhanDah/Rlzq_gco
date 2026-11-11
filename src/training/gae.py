"""GAE(γ, λ) avec terminaux (but) et troncatures (timeouts) distincts (ADR-003 §6, `GCO_SPEC.md` §2.2).

Conventions, pour un tampon (T, N) :
- `V_next[t]` = V(`obs[t+1]`), ou V(`last_obs`) à t = T−1 ;
- si `trunc[t]` (et pas `term[t]`) : `V_next[t]` = V(`final_obs`) de la ligne stockée pour (t, agent), calculée par le
  learner ; jamais V(s_t) (défaut du prototype `scripts/bench_e2e.py:126`, non repris) ;
- si `term[t]` : `V_next[t]` = 0 (un but est terminal, pas de bootstrap), prioritaire sur `trunc` ;
- la trace λ est coupée dans les deux cas : continuation = ¬(term ∨ trunc).
Fonctions en torch pur (CPU, XPU, CUDA) ; aucune indexation avancée `x[idx]` (cassée sur XPU, ADR-001 M3).
"""

from __future__ import annotations

import torch


def bootstrap_values(
    values: torch.Tensor,
    last_value: torch.Tensor,
    final_value: torch.Tensor,
    final_t: torch.Tensor,
    final_agent: torch.Tensor,
    term: torch.Tensor,
) -> torch.Tensor:
    """`V_next` (T, N) à partir de V(`obs`) (T, N), V(`last_obs`) (N,) et V(`final_obs`) (K,).

    `final_t`, `final_agent` (K,) i64 désignent les transitions tronquées (non terminales) dont `final_value` est la
    valeur de l'observation finale. `term` (T, N) bool force 0.
    """
    n = values.shape[1]
    if last_value.shape != (n,):
        raise ValueError(f"last_value {tuple(last_value.shape)} ≠ ({n},)")
    k = final_value.shape[0]
    if final_t.shape != (k,) or final_agent.shape != (k,):
        raise ValueError("final_value, final_t et final_agent doivent avoir la même longueur K")
    v_next = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0).contiguous()
    if k:
        flat = v_next.view(-1)
        flat.index_copy_(0, final_t * n + final_agent, final_value.to(flat.dtype))
    return torch.where(term, torch.zeros_like(v_next), v_next)


def gae(
    rew: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    term: torch.Tensor,
    trunc: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Avantages et retours (T, N) : δ_t = r_t + γ·V_next[t] − V[t] ; A_t = δ_t + γλ·c_t·A_{t+1}, c_t = ¬(term ∨ trunc).

    `next_values` vient de `bootstrap_values` (0 sur les terminaux). Retour = A + V (cible de la perte de valeur).
    """
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError(f"γ={gamma}, λ={lam} hors de [0, 1]")
    shape = rew.shape
    for name, x in (("values", values), ("next_values", next_values), ("term", term), ("trunc", trunc)):
        if x.shape != shape:
            raise ValueError(f"{name} {tuple(x.shape)} ≠ rew {tuple(shape)}")
    cont = (~(term | trunc)).to(rew.dtype)
    delta = rew + gamma * next_values - values
    adv = torch.empty_like(rew)
    last = torch.zeros_like(rew[0])
    for t in range(shape[0] - 1, -1, -1):
        last = delta[t] + (gamma * lam) * cont[t] * last
        adv[t] = last
    return adv, adv + values
