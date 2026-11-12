"""Gardes d'entraînement (kill criteria de `docs/PLAN.md`) : fonctions pures, décisions typées.

`evaluate(cfg, state, obs) -> (nouvel état, décision)` ne modifie rien ni ne fait d'E/S : l'appelant exécute la
décision (continuer, arrêter, restaurer le dernier checkpoint sain) et conserve le nouvel état. Les valeurs par
défaut viennent de `docs/PLAN.md` (§ Kill criteria préliminaires) :
- NaN/Inf (pertes, gradients, paramètres) -> RESTORE ;
- KL approx > 0,05 pendant 5 itérations consécutives -> RESTORE ;
- entropie < 10 % de ln(n_actions) avant S1 -> STOP ;
- explained variance < 0 pendant 50 itérations consécutives après S1 -> STOP ;
- plus de `max_restores` restaurations consécutives sans itération saine entre deux -> STOP (évite une boucle infinie).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum

import torch

N_ACTIONS_V0 = 90


class Action(Enum):
    CONTINUE = "continue"
    STOP = "stop"
    RESTORE = "restore"


@dataclass(frozen=True)
class GuardConfig:
    kl_threshold: float = 0.05  # PLAN.md : KL approx > 0,05
    kl_patience: int = 5  # ... pendant 5 itérations
    entropy_floor_frac: float = 0.10  # PLAN.md : entropie < 10 % de ln(90) avant S1
    n_actions: int = N_ACTIONS_V0
    entropy_patience: int = 1
    ev_floor: float = 0.0  # PLAN.md : explained variance < 0
    ev_patience: int = 50  # ... pendant 50 itérations après S1
    max_restores: int = 3

    def __post_init__(self) -> None:
        if min(self.kl_patience, self.entropy_patience, self.ev_patience) < 1 or self.max_restores < 0:
            raise ValueError("patiences >= 1 et max_restores >= 0")

    @property
    def entropy_floor(self) -> float:
        return self.entropy_floor_frac * math.log(self.n_actions)


@dataclass(frozen=True)
class GuardState:
    kl_streak: int = 0
    entropy_streak: int = 0
    ev_streak: int = 0
    restores: int = 0  # restaurations consécutives (remis à 0 par une itération saine)


@dataclass(frozen=True)
class GuardObservation:
    """Ce que voit la garde pour une itération. `params_finite` / `grads_finite` : résultats de `params_finite()` /
    `grads_finite()` (None = non vérifié).
    `s1_reached` : jalon S1 atteint (active la règle EV, coupe l'entropie)."""

    stats: Mapping[str, float]
    params_finite: bool | None = None
    grads_finite: bool | None = None
    s1_reached: bool = False


@dataclass(frozen=True)
class GuardDecision:
    action: Action
    rule: str | None  # "nonfinite", "kl", "entropy", "explained_variance", "max_restores" ; None si CONTINUE
    reason: str
    healthy: bool  # le checkpoint de cette itération peut devenir le « dernier sain »


LOSS_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm", "explained_variance")


def nonfinite_keys(stats: Mapping[str, float]) -> list[str]:
    return [k for k, v in stats.items() if isinstance(v, float) and not math.isfinite(v)]


def params_finite(params: Iterable[torch.nn.Parameter]) -> bool:
    """Tous les paramètres sont finis (une seule lecture hôte)."""
    flags = [torch.isfinite(p.detach()).all() for p in params]
    return bool(torch.stack(flags).all().item()) if flags else True


def grads_finite(params: Iterable[torch.nn.Parameter]) -> bool:
    """Tous les gradients présents sont finis (gradients du dernier minibatch)."""
    flags = [torch.isfinite(p.grad).all() for p in params if p.grad is not None]
    return bool(torch.stack(flags).all().item()) if flags else True


def evaluate(cfg: GuardConfig, state: GuardState, obs: GuardObservation) -> tuple[GuardState, GuardDecision]:
    s = obs.stats
    bad = nonfinite_keys(s)
    if obs.params_finite is False:
        bad.append("params")
    if obs.grads_finite is False:
        bad.append("grads")
    if bad:
        return _restore(cfg, state, "nonfinite", f"valeurs non finies : {', '.join(bad)}")

    kl_streak = state.kl_streak + 1 if s["approx_kl"] > cfg.kl_threshold else 0
    ent_low = (not obs.s1_reached) and s["entropy"] < cfg.entropy_floor
    entropy_streak = state.entropy_streak + 1 if ent_low else 0
    ev_neg = obs.s1_reached and s["explained_variance"] < cfg.ev_floor
    ev_streak = state.ev_streak + 1 if ev_neg else 0
    new = replace(state, kl_streak=kl_streak, entropy_streak=entropy_streak, ev_streak=ev_streak)

    if kl_streak >= cfg.kl_patience:
        return _restore(
            cfg,
            new,
            "kl",
            f"KL approx > {cfg.kl_threshold} depuis {kl_streak} itérations (dernier {s['approx_kl']:.4g})",
        )
    if entropy_streak >= cfg.entropy_patience:
        return new, GuardDecision(
            Action.STOP,
            "entropy",
            f"entropie {s['entropy']:.4g} < {cfg.entropy_floor:.4g} (effondrement avant S1)",
            healthy=False,
        )
    if ev_streak >= cfg.ev_patience:
        return new, GuardDecision(
            Action.STOP,
            "explained_variance",
            f"explained variance < {cfg.ev_floor} depuis {ev_streak} itérations après S1",
            healthy=False,
        )
    healthy = kl_streak == 0
    new = replace(new, restores=0) if healthy else new
    return new, GuardDecision(Action.CONTINUE, None, "ok", healthy=healthy)


def _restore(cfg: GuardConfig, state: GuardState, rule: str, reason: str) -> tuple[GuardState, GuardDecision]:
    if state.restores >= cfg.max_restores:
        return state, GuardDecision(
            Action.STOP, "max_restores", f"{reason} ; {state.restores} restaurations déjà tentées", healthy=False
        )
    fresh = GuardState(restores=state.restores + 1)  # séries remises à zéro : on repart du checkpoint sain
    return fresh, GuardDecision(Action.RESTORE, rule, reason, healthy=False)
