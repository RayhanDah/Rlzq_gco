"""Elo / Bradley-Terry par maximum de vraisemblance, ancré sur un joueur de référence (l'agent aléatoire = 0 Elo).

Convention (`docs/01_papers.md` §1.9, `docs/research/paper_logo_2202_and_refs.md` §13.3) : p(d) = 1 / (1 + 10^(-d/400)),
d(p) = 400 · log10(p / (1 - p)).

Les nuls (troncatures no-touch ou durée max) NE SONT PAS dans la matrice : le rating porte sur les parties décisives.
C'est cohérent avec le modèle de nuls du SPRT (`eval.sprt`) : sous un modèle de Davidson (nuls de rapport ν
arbitraire), P(A gagne | partie décisive) = 1 / (1 + 10^(-Δ/400)) quelle que soit ν : les nuls n'informent pas sur Δ.

Cas dégénérés (joueur invaincu, sans victoire, ou comparaisons non connexes) : le MLE n'existe pas (Elo infini).
`fit_bradley_terry` LÈVE `DegenerateRatingError` par défaut ; `prior > 0` ajoute `prior` victoires fictives dans chaque
sens à chaque paire ayant joué (régularisation vers l'égalité), l'Elo reste fini et le résultat porte le drapeau
`regularized`. Aucun infini n'est jamais renvoyé silencieusement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

ELO_SCALE = 400.0


class DegenerateRatingError(ValueError):
    """Le MLE Bradley-Terry n'existe pas (joueur invaincu ou sans victoire, graphe non fortement connexe)."""


def elo_to_score(diff: float) -> float:
    """Score attendu d'un joueur ayant `diff` Elo de plus que son adversaire."""
    return float(1.0 / (1.0 + 10.0 ** (-diff / ELO_SCALE)))


def score_to_elo(score: float) -> float:
    """Différence d'Elo pour un score attendu dans ]0, 1[. 0 et 1 (Elo infini) lèvent `DegenerateRatingError`."""
    if not 0.0 < score < 1.0:  # NaN => faux
        raise DegenerateRatingError(f"score {score} hors de ]0, 1[ : différence d'Elo infinie ou indéfinie")
    return ELO_SCALE * math.log10(score / (1.0 - score))


def regularized_elo(wins: float, losses: float, prior: float = 0.5) -> float:
    """Elo de `wins` victoires contre `losses` défaites avec `prior` victoire et `prior` défaite fictives (finie même à
    0 victoire ou 0 défaite ; l'appelant doit signaler la régularisation)."""
    if prior <= 0:
        raise ValueError("prior doit être > 0")
    return score_to_elo((wins + prior) / (wins + losses + 2 * prior))


@dataclass(frozen=True)
class BTFit:
    elo: F64Array
    """(n,) Elo, `elo[anchor] == 0`."""
    anchor: int
    iterations: int
    converged: bool
    regularized: bool
    """Vrai si des victoires fictives (`prior > 0`) ont été ajoutées."""


def _check_matrix(wins: npt.ArrayLike) -> F64Array:
    w = np.asarray(wins, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError(f"matrice carrée n >= 2 attendue, reçu {w.shape}")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("victoires finies et >= 0 attendues")
    if np.diag(w).any():
        raise ValueError("diagonale non nulle (un joueur ne joue pas contre lui-même)")
    return w


def _strongly_connected(played_win: BoolArray) -> bool:
    """Le graphe i -> j (i a battu j au moins une fois) est-il fortement connexe ? (condition d'existence du MLE)"""
    n = played_win.shape[0]
    for graph in (played_win, played_win.T):
        seen = np.zeros(n, dtype=bool)
        seen[0] = True
        frontier = [0]
        while frontier:
            i = frontier.pop()
            for j in np.flatnonzero(graph[i] & ~seen):
                seen[j] = True
                frontier.append(int(j))
        if not seen.all():
            return False
    return True


def fit_bradley_terry(
    wins: npt.ArrayLike,
    anchor: int = 0,
    *,
    prior: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 20000,
) -> BTFit:
    """MLE Bradley-Terry. `wins[i, j]` = victoires décisives de i sur j. Itération MM de Zermelo (Hunter 2004).

    Lève `DegenerateRatingError` si `prior == 0` et que le MLE n'existe pas, ou (toujours) si le graphe des paires
    ayant joué n'est pas connexe (les forces de deux composantes ne sont pas comparables).
    """
    w = _check_matrix(wins)
    n = w.shape[0]
    if not 0 <= anchor < n:
        raise ValueError(f"ancre {anchor} hors de [0, {n})")
    if prior < 0:
        raise ValueError("prior doit être >= 0")
    games = w + w.T
    played = games > 0
    # connexité non orientée : sans elle, aucun prior ne rend les Elo comparables
    undirected = played | played.T
    seen = np.zeros(n, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(undirected[i] & ~seen):
            seen[j] = True
            stack.append(int(j))
    if not seen.all():
        raise DegenerateRatingError(f"joueurs sans comparaison avec le reste : {np.flatnonzero(~seen).tolist()}")
    if prior == 0.0:
        if not _strongly_connected(w > 0):
            bad = [i for i in range(n) if w[i].sum() == 0 or w[:, i].sum() == 0]
            raise DegenerateRatingError(
                f"MLE inexistant (Elo infini) : joueur(s) invaincu(s) ou sans victoire {bad}, ou cycle de domination "
                "absent ; utiliser prior > 0 pour une estimation régularisée (signalée)"
            )
    else:
        w = w + prior * played
        games = w + w.T
    wins_i = w.sum(axis=1)
    pi = np.ones(n)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 (`it` rendu après la boucle)
        denom = (games / (pi[:, None] + pi[None, :])).sum(axis=1)
        new = wins_i / denom
        new /= np.exp(np.log(new).mean())  # normalisation (le MLE est défini à un facteur près)
        if float(np.abs(np.log(new) - np.log(pi)).max()) < tol:
            pi, converged = new, True
            break
        pi = new
    elo = ELO_SCALE * np.log10(pi)
    elo -= elo[anchor]
    return BTFit(elo=elo, anchor=anchor, iterations=it, converged=converged, regularized=prior > 0)


@dataclass(frozen=True)
class RatingResult:
    elo: F64Array
    ci_low: F64Array
    ci_high: F64Array
    anchor: int
    n_boot: int
    alpha: float
    regularized: bool
    degenerate: BoolArray
    """(n,) joueurs invaincus ou sans victoire (Elo borné par le prior, pas estimé)."""


def rate_players(
    wins: npt.ArrayLike,
    anchor: int = 0,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    prior: float = 0.0,
) -> RatingResult:
    """Elo MLE ancré + IC percentile (1 - alpha) par bootstrap par paire (chaque paire (i, j)
    rééchantillonnée en Binomial(n_ij, taux observé), ce qui équivaut au bootstrap des parties i.i.d.).

    `prior == 0` : lève si le point estimé est dégénéré (voir `fit_bradley_terry`). `prior > 0` : toutes les fits (point
    et réplicats) sont régularisées ; `degenerate` signale les joueurs dont l'Elo n'est alors qu'une borne.
    """
    w = _check_matrix(wins)
    n = w.shape[0]
    point = fit_bradley_terry(w, anchor, prior=prior)
    games = w + w.T
    degenerate = (w.sum(axis=1) == 0) | (w.sum(axis=0) == 0)
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, k=1)
    n_ij = games[iu].astype(np.int64)
    p_ij = np.divide(w[iu], games[iu], out=np.full(n_ij.shape, 0.5), where=n_ij > 0)
    samples = np.empty((n_boot, n))
    for b in range(n_boot):
        draw = rng.binomial(n_ij, p_ij).astype(np.float64)
        wb = np.zeros((n, n))
        wb[iu] = draw
        wb[(iu[1], iu[0])] = n_ij - draw
        try:
            samples[b] = fit_bradley_terry(wb, anchor, prior=prior).elo
        except DegenerateRatingError:
            samples[b] = np.nan  # réplicat dégénéré sans prior : exclu des percentiles (compté ci-dessous)
    ok = ~np.isnan(samples).any(axis=1)
    if ok.sum() < max(2, n_boot // 2):
        raise DegenerateRatingError(f"{n_boot - int(ok.sum())}/{n_boot} réplicats bootstrap dégénérés : IC non fiable")
    lo = np.percentile(samples[ok], 100 * alpha / 2, axis=0)
    hi = np.percentile(samples[ok], 100 * (1 - alpha / 2), axis=0)
    return RatingResult(
        elo=point.elo,
        ci_low=lo,
        ci_high=hi,
        anchor=anchor,
        n_boot=n_boot,
        alpha=alpha,
        regularized=prior > 0,
        degenerate=degenerate,
    )
