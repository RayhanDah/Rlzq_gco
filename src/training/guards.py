"""Gardes d'entraînement (kill criteria de `docs/PLAN.md`) : fonctions pures, décisions typées.

`evaluate(cfg, state, obs) -> (nouvel état, décision)` ne modifie rien ni ne fait d'E/S : l'appelant exécute la
décision (continuer, arrêter, restaurer le dernier checkpoint sain) et conserve le nouvel état. Les valeurs par
défaut viennent de `docs/PLAN.md` (§ Kill criteria préliminaires) :
- NaN/Inf (pertes, gradients, paramètres) -> RESTORE ;
- KL approx > 0,05 pendant 5 itérations consécutives -> RESTORE ;
- entropie < 10 % de ln(n_actions) avant S1 -> STOP ;
- ALERTE sans action (ADR-005 D6, K2) : entropie < 50 % de ln(n_actions) (2,25 pour 90 actions), avant comme après
  S1 ; portée par `GuardDecision.alerts`, journalisée par l'appelant ;
- explained variance < 0 pendant 50 itérations consécutives après S1 -> STOP ;
- plus de `max_restores` restaurations au total depuis le début du run (cumulées, une itération saine ne remet PAS ce
  compteur à zéro : un run qui diverge de temps en temps finit par s'arrêter) -> STOP, règle `max_restores` ;
- plus de `max_consecutive_restores` restaurations sans itération saine entre deux -> STOP, règle
  `max_consecutive_restores` (une divergence reproductible depuis le même checkpoint ne boucle pas).
Valeurs par défaut `max_restores = 3` et `max_consecutive_restores = 2` : choix de l'équipe (PLAN.md ne chiffre pas ces
plafonds), à réviser au G3. Les deux compteurs sont sauvegardés dans les checkpoints (`GuardState`).
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
    entropy_alert_frac: float = 0.50  # ADR-005 D6 K2 : alerte journalisée sous 50 % de ln(90) = 2,25
    ev_floor: float = 0.0  # PLAN.md : explained variance < 0
    ev_patience: int = 50  # ... pendant 50 itérations après S1
    max_restores: int = 3  # restaurations CUMULÉES sur tout le run
    max_consecutive_restores: int = 2  # restaurations sans itération saine entre deux

    def __post_init__(self) -> None:
        if min(self.kl_patience, self.entropy_patience, self.ev_patience) < 1:
            raise ValueError("patiences >= 1")
        if self.max_restores < 0 or self.max_consecutive_restores < 0:
            raise ValueError("max_restores >= 0 et max_consecutive_restores >= 0")
        if not 0.0 <= self.entropy_alert_frac <= 1.0:
            raise ValueError("entropy_alert_frac dans [0, 1]")

    @property
    def entropy_floor(self) -> float:
        return self.entropy_floor_frac * math.log(self.n_actions)

    @property
    def entropy_alert(self) -> float:
        return self.entropy_alert_frac * math.log(self.n_actions)


@dataclass(frozen=True)
class GuardState:
    kl_streak: int = 0
    entropy_streak: int = 0
    ev_streak: int = 0
    restores: int = 0  # restaurations consécutives (remis à 0 par une itération saine)
    restores_total: int = 0  # restaurations cumulées depuis le début du run (jamais remis à 0)


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
    # "nonfinite", "kl", "entropy", "explained_variance", "max_restores", "max_consecutive_restores" ; None si CONTINUE
    rule: str | None
    reason: str
    healthy: bool  # le checkpoint de cette itération peut devenir le « dernier sain »
    alerts: tuple[str, ...] = ()
    """Alertes sans action (`entropy_low`, K2 à 50 %), une chaîne « règle : détail » par alerte ; l'appelant les
    journalise. Vide sur une décision `nonfinite` (statistiques inexploitables)."""


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


def entropy_alert(cfg: GuardConfig, stats: Mapping[str, float]) -> str | None:
    """Alerte K2 (ADR-005 D6) si l'entropie est finie et sous `entropy_alert_frac` · ln(n_actions), sinon None."""
    ent = stats.get("entropy")
    if ent is None or not math.isfinite(ent) or ent >= cfg.entropy_alert:
        return None
    return (
        f"entropy_low : entropie {ent:.4g} < {cfg.entropy_alert:.4g} "
        f"({cfg.entropy_alert_frac:.0%} de ln {cfg.n_actions}, ADR-005 K2)"
    )


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
    alert = entropy_alert(cfg, s)
    alerts = (alert,) if alert is not None else ()

    if kl_streak >= cfg.kl_patience:
        st, dec = _restore(
            cfg,
            new,
            "kl",
            f"KL approx > {cfg.kl_threshold} depuis {kl_streak} itérations (dernier {s['approx_kl']:.4g})",
        )
        return st, replace(dec, alerts=alerts)
    if entropy_streak >= cfg.entropy_patience:
        return new, GuardDecision(
            Action.STOP,
            "entropy",
            f"entropie {s['entropy']:.4g} < {cfg.entropy_floor:.4g} (effondrement avant S1)",
            healthy=False,
            alerts=alerts,
        )
    if ev_streak >= cfg.ev_patience:
        return new, GuardDecision(
            Action.STOP,
            "explained_variance",
            f"explained variance < {cfg.ev_floor} depuis {ev_streak} itérations après S1",
            healthy=False,
            alerts=alerts,
        )
    healthy = kl_streak == 0
    new = replace(new, restores=0) if healthy else new
    return new, GuardDecision(Action.CONTINUE, None, "ok", healthy=healthy, alerts=alerts)


def _restore(cfg: GuardConfig, state: GuardState, rule: str, reason: str) -> tuple[GuardState, GuardDecision]:
    if state.restores_total >= cfg.max_restores:
        return state, GuardDecision(
            Action.STOP,
            "max_restores",
            f"{reason} ; {state.restores_total} restaurations déjà effectuées dans ce run (plafond {cfg.max_restores})",
            healthy=False,
        )
    if state.restores >= cfg.max_consecutive_restores:
        return state, GuardDecision(
            Action.STOP,
            "max_consecutive_restores",
            f"{reason} ; {state.restores} restaurations consécutives sans itération saine "
            f"(plafond {cfg.max_consecutive_restores})",
            healthy=False,
        )
    # séries remises à zéro : on repart du checkpoint sain
    fresh = GuardState(restores=state.restores + 1, restores_total=state.restores_total + 1)
    return fresh, GuardDecision(Action.RESTORE, rule, reason, healthy=False)
