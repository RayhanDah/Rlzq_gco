"""Elo / Bradley-Terry (`eval.rating`) : conversions, récupération de forces connues, cas dégénérés explicites."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from eval.rating import (
    DegenerateRatingError,
    elo_to_score,
    fit_bradley_terry,
    rate_players,
    regularized_elo,
    score_to_elo,
)

TRUE_ELO = np.array([0.0, 120.0, -60.0, 250.0])  # joueur 0 = agent aléatoire (ancre)


def _synthetic_wins(elo: npt.NDArray[np.float64], n_games: int, seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    n = elo.size
    wins = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = rng.binomial(n_games, elo_to_score(float(elo[i] - elo[j])))
            wins[i, j], wins[j, i] = w, n_games - w
    return wins


def test_conversions_score_elo() -> None:
    assert elo_to_score(0.0) == 0.5
    assert score_to_elo(0.5) == 0.0
    assert elo_to_score(400.0) == pytest.approx(10 / 11)  # 400 Elo = 10 contre 1
    assert elo_to_score(50.0) == pytest.approx(0.5716, abs=5e-4)
    for d in (-300.0, -12.5, 7.0, 180.0):
        assert score_to_elo(elo_to_score(d)) == pytest.approx(d, abs=1e-9)
    assert elo_to_score(-30.0) == pytest.approx(1 - elo_to_score(30.0))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_score_extreme_leve_au_lieu_de_renvoyer_un_infini(bad: float) -> None:
    with pytest.raises(DegenerateRatingError):
        score_to_elo(bad)


def test_elo_regularise_fini_aux_extremes() -> None:
    assert math.isfinite(regularized_elo(40, 0))
    assert regularized_elo(40, 0) > 0 > regularized_elo(0, 40)
    assert regularized_elo(10, 10) == 0.0
    with pytest.raises(ValueError):
        regularized_elo(1, 1, prior=0.0)


def test_recupere_des_forces_connues() -> None:
    wins = _synthetic_wins(TRUE_ELO, n_games=4000, seed=0)
    fit = fit_bradley_terry(wins, anchor=0)
    assert fit.converged and fit.elo[0] == 0.0 and not fit.regularized
    # 4000 parties par paire : SE Elo ≈ 400/ln(10)/sqrt(n p q) ≈ 3 à 7 Elo par paire ; tolérance 15 Elo (> 3 sigma
    # pour le plus incertain, les écarts de 250 Elo ayant p(1-p) petit).
    assert np.abs(fit.elo - TRUE_ELO).max() < 15.0


def test_ancre_invariante_et_ordre_des_joueurs() -> None:
    wins = _synthetic_wins(TRUE_ELO, n_games=500, seed=1)
    a0 = fit_bradley_terry(wins, anchor=0).elo
    a2 = fit_bradley_terry(wins, anchor=2).elo
    assert a2[2] == 0.0
    assert np.allclose(a0 - a0[2], a2, atol=1e-6)  # changer d'ancre = décalage constant
    assert list(np.argsort(a0)) == list(np.argsort(TRUE_ELO))


def test_deux_joueurs_egale_la_formule_directe() -> None:
    wins = np.array([[0.0, 70.0], [30.0, 0.0]])
    fit = fit_bradley_terry(wins, anchor=1)
    assert fit.elo[0] == pytest.approx(score_to_elo(0.7), abs=1e-6)


def test_ic_bootstrap_couvre_la_verite_et_est_reproductible() -> None:
    wins = _synthetic_wins(TRUE_ELO, n_games=300, seed=2)
    r1 = rate_players(wins, anchor=0, n_boot=300, seed=7)
    r2 = rate_players(wins, anchor=0, n_boot=300, seed=7)
    assert np.array_equal(r1.ci_low, r2.ci_low) and np.array_equal(r1.ci_high, r2.ci_high)  # graine fixe
    assert r1.ci_low[0] == r1.ci_high[0] == 0.0  # l'ancre est exacte
    assert (r1.ci_low <= r1.elo).all() and (r1.elo <= r1.ci_high).all()
    # couverture de la vérité par IC 95 % : non garantie par joueur ; graine fixe, vérifiée à la mesure (ce test fige
    # le résultat de la graine 2, il ne prouve pas la couverture nominale)
    assert ((r1.ci_low <= TRUE_ELO) & (TRUE_ELO <= r1.ci_high)).all()
    width = (r1.ci_high - r1.ci_low)[1:]
    assert (width > 0).all() and (width < 120).all()


def test_largeur_de_l_ic_decroit_avec_le_nombre_de_parties() -> None:
    small = rate_players(_synthetic_wins(TRUE_ELO, 100, 3), n_boot=200, seed=0)
    big = rate_players(_synthetic_wins(TRUE_ELO, 1600, 3), n_boot=200, seed=0)
    assert ((big.ci_high - big.ci_low)[1:] < (small.ci_high - small.ci_low)[1:]).all()


def test_joueur_invaincu_leve_par_defaut() -> None:
    wins = _synthetic_wins(TRUE_ELO, 200, 4)
    wins[3, :3] = 200.0  # joueur 3 bat tout le monde 200-0
    wins[:3, 3] = 0.0
    with pytest.raises(DegenerateRatingError, match=r"invaincu|MLE inexistant"):
        fit_bradley_terry(wins)
    with pytest.raises(DegenerateRatingError):
        rate_players(wins, n_boot=20)


def test_joueur_sans_victoire_leve_par_defaut() -> None:
    wins = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 12.0], [30.0, 18.0, 0.0]])  # joueur 0 : 0 victoire
    with pytest.raises(DegenerateRatingError):
        fit_bradley_terry(wins)


def test_prior_rend_l_elo_fini_et_signale_la_regularisation() -> None:
    wins = np.array([[0.0, 50.0, 50.0], [0.0, 0.0, 30.0], [0.0, 20.0, 0.0]])  # joueur 0 invaincu
    fit = fit_bradley_terry(wins, anchor=1, prior=0.5)
    assert fit.regularized and fit.converged and np.isfinite(fit.elo).all()
    assert fit.elo[0] > fit.elo[2] > -1e9 and fit.elo[0] > 0  # classé premier, fini
    res = rate_players(wins, anchor=1, n_boot=100, seed=0, prior=0.5)
    assert res.regularized
    assert np.isfinite(res.ci_low).all() and np.isfinite(res.ci_high).all()
    assert res.degenerate.tolist() == [True, False, False]  # joueur 0 invaincu ; joueur 2 a des victoires (20 sur 1)


def test_comparaisons_non_connexes_levent_meme_avec_prior() -> None:
    wins = np.zeros((4, 4))
    wins[0, 1], wins[1, 0] = 10.0, 5.0
    wins[2, 3], wins[3, 2] = 7.0, 8.0
    with pytest.raises(DegenerateRatingError, match="sans comparaison"):
        fit_bradley_terry(wins, prior=0.5)


def test_entrees_invalides() -> None:
    with pytest.raises(ValueError):
        fit_bradley_terry(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        fit_bradley_terry(np.array([[0.0, -1.0], [2.0, 0.0]]))
    with pytest.raises(ValueError):
        fit_bradley_terry(np.array([[1.0, 2.0], [2.0, 0.0]]))
    with pytest.raises(ValueError):
        fit_bradley_terry(np.array([[0.0, 2.0], [2.0, 0.0]]), anchor=5)
