"""SPRT de Wald sur des résultats victoire / nul / défaite, en différence d'Elo (`docs/01_papers.md` §1.9).

H0 : Δ = elo0 (0), H1 : Δ = elo1 (+50), erreurs α (accepter H1 à tort) et β (accepter H0 à tort), bornes de Wald
    A = ln((1 - β) / α) (accepter H1 si LLR >= A),   B = ln(β / (1 - α)) (accepter H0 si LLR <= B)
soit ±2,944 pour α = β = 0,05.

Modèle de nuls : les nuls (troncatures) sont IGNORÉS par la vraisemblance. Justification : sous un modèle de Davidson
(P(A gagne) = π_A / S, P(nul) = ν √(π_A π_B) / S, S = π_A + π_B + ν √(π_A π_B), π = 10^(Elo/400), ν > 0 inconnu), la
probabilité de gagner CONDITIONNELLEMENT à une partie décisive vaut π_A / (π_A + π_B) = elo_to_score(Δ), indépendante
de ν. La vraisemblance se factorise (nul ou non) × (victoire | décisif) ; le premier facteur ne dépend pas de Δ, donc
le rapport de vraisemblance entre H0 et H1 est exactement celui des parties décisives, avec ν en paramètre de nuisance
sans hypothèse sur sa valeur. Δ est alors « l'Elo des parties décisives ». Conséquence : à effet égal, le nombre moyen
d'épisodes vaut (nombre moyen de parties décisives du SPRT de Bernoulli) / (1 - taux de nuls) ; le plafond d'épisodes
compte les nuls.

Alimentation par paires (échange des côtés) : le test est évalué aux seules frontières de paires (`check_every=2`).
Un avantage de côté rend les deux parties d'une paire non identiquement distribuées ; la variance de leur somme est
alors inférieure à celle du modèle i.i.d. (p_b q_b + p_o q_o <= 2 p̄ q̄) : l'approximation i.i.d. est conservatrice sur
la variance ; l'espérance de la somme n'est pas exactement celle du modèle (biais de second ordre), non quantifié ici.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eval.rating import elo_to_score, wilson_interval

I8Array = npt.NDArray[np.int8]


class Decision(enum.Enum):
    CONTINUE = "continue"
    ACCEPT_H1 = "accept_h1"
    ACCEPT_H0 = "accept_h0"
    CAP_REACHED = "cap_reached"
    """Plafond (épisodes ou parties décisives) atteint sans franchir de borne : ni H0 ni H1 acceptée."""


@dataclass(frozen=True)
class SPRTConfig:
    elo0: float = 0.0
    elo1: float = 50.0
    alpha: float = 0.05
    beta: float = 0.05
    max_episodes: int = 1200
    """Plafond sur le NOMBRE TOTAL d'épisodes (nuls compris) ; `docs/01_papers.md` §1.9 : 1 200."""
    max_decisive: int | None = None
    """Plafond optionnel sur le nombre de parties DÉCISIVES (victoires + défaites). Le test s'arrête au premier des deux
    plafonds. Utile quand la part de nuls est très élevée (l'échantillon utile est alors rare)."""

    def __post_init__(self) -> None:
        if not self.elo1 > self.elo0:
            raise ValueError("H1 doit avoir un Elo strictement supérieur à H0")
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError("alpha et beta dans ]0, 1[")
        if self.max_episodes < 1:
            raise ValueError("max_episodes >= 1")
        if self.max_decisive is not None and self.max_decisive < 1:
            raise ValueError("max_decisive >= 1 ou None")

    @property
    def p0(self) -> float:
        return elo_to_score(self.elo0)

    @property
    def p1(self) -> float:
        return elo_to_score(self.elo1)

    @property
    def upper(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def win_increment(self) -> float:
        return math.log(self.p1 / self.p0)

    @property
    def loss_increment(self) -> float:
        return math.log((1.0 - self.p1) / (1.0 - self.p0))


@dataclass(frozen=True)
class SPRTResult:
    decision: Decision
    llr: float
    lower: float
    upper: float
    wins: int
    draws: int
    losses: int
    n_episodes: int
    n_decisive: int
    """Victoires + défaites (les deux compteurs de plafond : `n_episodes` et `n_decisive`)."""

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "llr": self.llr,
            "lower": self.lower,
            "upper": self.upper,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "n_episodes": self.n_episodes,
            "n_decisive": self.n_decisive,
        }


def llr(wins: int, losses: int, cfg: SPRTConfig) -> float:
    """Log-rapport de vraisemblance H1 / H0 des parties décisives."""
    return wins * cfg.win_increment + losses * cfg.loss_increment


def sprt_status(wins: int, draws: int, losses: int, cfg: SPRTConfig) -> SPRTResult:
    """Décision à partir des comptes cumulés (à appeler aux frontières de paires)."""
    if min(wins, draws, losses) < 0:
        raise ValueError("comptes négatifs")
    n = wins + draws + losses
    s = llr(wins, losses, cfg)
    if s >= cfg.upper:
        d = Decision.ACCEPT_H1
    elif s <= cfg.lower:
        d = Decision.ACCEPT_H0
    elif n >= cfg.max_episodes or (cfg.max_decisive is not None and wins + losses >= cfg.max_decisive):
        d = Decision.CAP_REACHED
    else:
        d = Decision.CONTINUE
    return SPRTResult(d, s, cfg.lower, cfg.upper, wins, draws, losses, n, wins + losses)


def sprt_scan(outcomes: Sequence[int] | npt.NDArray[np.integer], cfg: SPRTConfig, check_every: int = 1) -> SPRTResult:
    """Rejoue le test séquentiellement sur `outcomes` (+1 victoire, 0 nul, -1 défaite, dans l'ordre de jeu) en
    évaluant la décision après chaque bloc de `check_every` épisodes (2 = frontières de paires) et s'arrête au premier
    franchissement de borne ou au premier plafond (épisodes ou décisifs).
    Si les résultats s'épuisent avant toute décision : `CONTINUE`."""
    if check_every < 1:
        raise ValueError("check_every >= 1")
    o = np.asarray(outcomes)
    if o.size and not np.isin(o, (-1, 0, 1)).all():
        raise ValueError("résultats dans {-1, 0, +1}")
    o = o[: cfg.max_episodes]
    ends = np.arange(check_every, o.size + 1, check_every)
    if o.size == cfg.max_episodes and (ends.size == 0 or int(ends[-1]) != o.size):
        ends = np.append(ends, o.size)  # le plafond tombe hors frontière de bloc : évalué quand même au plafond
    if ends.size == 0:
        return sprt_status(0, 0, 0, cfg)
    cw = np.concatenate([[0], np.cumsum(o == 1)])
    cl = np.concatenate([[0], np.cumsum(o == -1)])
    s = cw[ends] * cfg.win_increment + cl[ends] * cfg.loss_increment
    stop = (s >= cfg.upper) | (s <= cfg.lower)
    if cfg.max_decisive is not None:
        stop |= (cw[ends] + cl[ends]) >= cfg.max_decisive
    hit = np.flatnonzero(stop)
    n = int(ends[hit[0]]) if hit.size else int(ends[-1])
    w, lo = int(cw[n]), int(cl[n])
    return sprt_status(w, n - w - lo, lo, cfg)


class DecisiveVerdict(enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class DecisiveTestResult:
    verdict: DecisiveVerdict
    wins: int
    losses: int
    decisive: int
    win_rate: float | None
    """victoires / décisifs ; None sans partie décisive."""
    wilson_low: float | None
    wilson_high: float | None
    min_decisive: int
    alpha: float

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "wins": self.wins,
            "losses": self.losses,
            "decisive": self.decisive,
            "win_rate": self.win_rate,
            "wilson_low": self.wilson_low,
            "wilson_high": self.wilson_high,
            "min_decisive": self.min_decisive,
            "alpha": self.alpha,
        }


def decisive_win_test(wins: int, losses: int, *, min_decisive: int = 20, alpha: float = 0.05) -> DecisiveTestResult:
    """Critère sur les seules parties décisives : A est meilleur que B si la borne BASSE de Wilson (unilatérale, niveau
    1 - alpha) de victoires / décisifs dépasse 0,5.

    Issues : `PASSED` (borne basse > 0,5), `FAILED` (décisifs >= `min_decisive` et borne basse <= 0,5, c'est-à-dire
    « pas démontré meilleur » et non « démontré moins bon »), `INDETERMINATE` (décisifs < `min_decisive`, y compris 0 :
    jamais « réussi » faute de données, même avec 5 victoires sur 5). Les nuls n'entrent pas."""
    if wins < 0 or losses < 0:
        raise ValueError("comptes négatifs")
    if min_decisive < 1:
        raise ValueError("min_decisive >= 1")
    n = wins + losses
    if n == 0:
        return DecisiveTestResult(DecisiveVerdict.INDETERMINATE, wins, losses, 0, None, None, None, min_decisive, alpha)
    low, high = wilson_interval(wins, n, alpha, two_sided=False)
    if n < min_decisive:
        verdict = DecisiveVerdict.INDETERMINATE
    else:
        verdict = DecisiveVerdict.PASSED if low > 0.5 else DecisiveVerdict.FAILED
    return DecisiveTestResult(verdict, wins, losses, n, wins / n, low, high, min_decisive, alpha)
