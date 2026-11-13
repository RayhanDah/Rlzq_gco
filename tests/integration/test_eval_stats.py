"""Propriétés statistiques de l'évaluation par simulation Monte-Carlo (seeds fixes). Sans RocketGPU ni torch.

Ces simulations sont trop longues pour `tests/unit` (limite projet : 60 s pour tout `tests/unit`). Chaque tolérance est
justifiée dans le test : le plus souvent 3 écarts-types binomiaux de l'erreur de Monte-Carlo.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from eval.rating import elo_to_score, rate_players
from eval.sprt import Decision, SPRTConfig, sprt_scan

N_RUNS = 4000
SE_FACTOR = 3.0
P1 = elo_to_score(50.0)


def _se(p: float, n: int) -> float:
    return math.sqrt(p * (1.0 - p) / n)


def _simulate(
    p_win: float, p_draw: float, cfg: SPRTConfig, seed: int, n_runs: int = N_RUNS, chunk: int = 500
) -> list[tuple[Decision, int, int]]:
    """`n_runs` tests sur des épisodes i.i.d. (victoire p_win, nul p_draw, défaite le reste) ; (décision, épisodes,
    décisifs) par run. Les runs sont générés par lots pour borner la mémoire."""
    rng = np.random.default_rng(seed)
    out: list[tuple[Decision, int, int]] = []
    for start in range(0, n_runs, chunk):
        u = rng.random((min(chunk, n_runs - start), cfg.max_episodes))
        outcomes = np.where(u < p_win, 1, np.where(u < p_win + p_draw, 0, -1)).astype(np.int8)
        for row in outcomes:
            r = sprt_scan(row, cfg)
            out.append((r.decision, r.n_episodes, r.n_decisive))
    return out


def _rate(res: list[tuple[Decision, int, int]], d: Decision) -> float:
    return sum(x[0] is d for x in res) / len(res)


# ------------------------------------------------------------------------------------------------ sans nuls


def test_erreur_de_type_i_sous_h0_conforme_a_alpha() -> None:
    res = _simulate(0.5, 0.0, SPRTConfig(), seed=1)
    alpha_hat = _rate(res, Decision.ACCEPT_H1)
    # Wald : α_réel <= α / (1 - β) = 0,0526 (le dépassement des bornes rend le test conservateur) ;
    # borne basse 0,03 : écart au nominal dû au dépassement modeste (0,043-0,050 mesuré dans le doc §13.4).
    assert 0.03 < alpha_hat <= 0.05 / 0.95 + SE_FACTOR * _se(0.05, N_RUNS)
    assert _rate(res, Decision.CAP_REACHED) < 0.05  # plafond 1200 rarement atteint


def test_puissance_sous_h1_conforme_a_1_moins_beta() -> None:
    res = _simulate(P1, 0.0, SPRTConfig(), seed=2)
    power = _rate(res, Decision.ACCEPT_H1)
    # β_réel <= β / (1 - α) = 0,0526 : puissance >= 0,947, moins 3 ET de Monte-Carlo
    assert 0.947 - SE_FACTOR * _se(0.95, N_RUNS) <= power < 0.995


def test_nombre_moyen_d_episodes_ordre_de_grandeur_du_doc() -> None:
    """docs/01_papers.md §1.9 : ≈ 260 épisodes en moyenne à +50 Elo (Wald : 257 sous H0, 259 sous H1)."""
    for seed, p in ((3, 0.5), (4, P1)):
        n = np.array([m for _, m, _ in _simulate(p, 0.0, SPRTConfig(), seed=seed)], dtype=np.float64)
        assert 257 * 0.9 < n.mean() < 259 * 1.1  # ±10 % (SE de la moyenne ≈ 3, dépassement des bornes en sus)
        assert 550 < np.percentile(n, 95) < 700  # doc : p95 ≈ 633 (mesuré 615-639)


def test_le_plafond_de_1200_n_est_pas_le_p95() -> None:
    """Le doc écrit « plafond 1 200 ≈ p95 » : mesuré, le plafond est atteint dans ≈ 0,2 % des runs (≈ p99,8)."""
    res = _simulate(0.5, 0.0, SPRTConfig(), seed=5)
    assert _rate(res, Decision.CAP_REACHED) < 0.01  # contredit « ≈ p95 » (5 %)


# ------------------------------------------------------------------------------------------------ avec nuls


def test_avec_50_pour_cent_de_nuls_erreurs_inchangees_et_n_double() -> None:
    cfg = SPRTConfig(max_episodes=4000)
    n = 1500
    res = _simulate(0.25, 0.5, cfg, seed=6, n_runs=n)  # H0 vraie
    assert _rate(res, Decision.ACCEPT_H1) <= 0.05 / 0.95 + SE_FACTOR * _se(0.05, n)
    res1 = _simulate(0.5 * P1, 0.5, cfg, seed=7, n_runs=n)  # H1 vraie
    assert 2 * 257 * 0.9 < np.mean([m for _, m, _ in res1]) < 2 * 259 * 1.1  # deux fois plus d'épisodes


def _simulate_sparse(
    p_dec_win: float, p_dec: float, cap_decisive: int, seed: int, n_runs: int
) -> list[tuple[Decision, int, int]]:
    """Même loi que `_simulate` pour une part de nuls très élevée, sans générer les ~18 000 nuls : des épisodes i.i.d.
    (décisif avec probabilité `p_dec`) ont une suite de décisifs i.i.d. ; le test ne dépend que de cette suite (appel
    avec `check_every = 1`), et le k-ième décisif tombe à l'épisode k + NegBin(k, p_dec) (nuls avant le k-ième décisif).
    L'équivalence est vérifiée en direct par `test_simulation_creuse_equivalente_a_la_simulation_directe`."""
    rng = np.random.default_rng(seed)
    cfg = SPRTConfig(max_episodes=10**9, max_decisive=cap_decisive)
    out = []
    for _ in range(n_runs):
        dec = np.where(rng.random(cap_decisive) < p_dec_win, 1, -1).astype(np.int8)
        r = sprt_scan(dec, cfg)
        k = r.n_decisive
        out.append((r.decision, k + int(rng.negative_binomial(k, p_dec)), k))
    return out


def test_simulation_creuse_equivalente_a_la_simulation_directe() -> None:
    p_dec = 0.2  # 80 % de nuls : assez pour que la simulation directe reste courte
    cfg = SPRTConfig(max_episodes=20_000, max_decisive=1200)
    direct = _simulate(p_dec * P1, 1 - p_dec, cfg, seed=20, n_runs=200, chunk=100)
    sparse = _simulate_sparse(P1, p_dec, 1200, seed=21, n_runs=2000)
    md, ms = np.mean([e for _, e, _ in direct]), np.mean([e for _, e, _ in sparse])
    # 200 runs : SE de la moyenne directe ≈ 190 / sqrt(200) / 0,2 ≈ 70 épisodes ; tolérance 10 % (≈ 130 sur 1300)
    assert abs(md - ms) < 0.10 * ms
    assert abs(_rate(direct, Decision.ACCEPT_H1) - _rate(sparse, Decision.ACCEPT_H1)) < 0.06


def test_a_98_pour_cent_de_nuls_alpha_et_puissance_sur_les_decisifs() -> None:
    """Part de nuls de la mesure « aléatoire contre aléatoire » (98,6 %) : les erreurs restent celles du test sur
    les décisifs (modèle de Davidson), le nombre d'épisodes explose (~ 260 / 0,014 ≈ 18 000) et c'est le plafond en
    décisifs qui borne le test."""
    p_dec = 0.014
    n = 2000
    h0 = _simulate_sparse(0.5, p_dec, 1200, seed=8, n_runs=n)
    alpha_hat = _rate(h0, Decision.ACCEPT_H1)
    assert alpha_hat <= 0.05 / 0.95 + SE_FACTOR * _se(0.05, n)
    h1 = _simulate_sparse(P1, p_dec, 1200, seed=9, n_runs=n)
    power = _rate(h1, Decision.ACCEPT_H1)
    assert power >= 0.947 - SE_FACTOR * _se(0.95, n)
    # décisifs consommés ≈ ceux du cas sans nuls (≈ 260), épisodes ≈ décisifs / 0,014
    mean_dec = float(np.mean([d for _, _, d in h1]))
    mean_ep = float(np.mean([e for _, e, _ in h1]))
    assert 257 * 0.9 < mean_dec < 259 * 1.1
    assert 0.9 * mean_dec / p_dec < mean_ep < 1.1 * mean_dec / p_dec
    assert _rate(h0, Decision.CAP_REACHED) < 0.05


# ------------------------------------------------------------------------------------------------ avantage de côté


def _pair_runs(delta: float, side: float, n_runs: int, seed: int) -> list[tuple[Decision, int]]:
    """SPRT rétrospectif évalué aux frontières de paires. Chaque paire : A bleu gagne (décisif) avec σ(Δ + side), A
    orange avec σ(Δ - side) (`side` = avantage Elo du bleu). Sans nuls."""
    rng = np.random.default_rng(seed)
    cfg = SPRTConfig()
    pb, po = elo_to_score(delta + side), elo_to_score(delta - side)
    p = np.array([pb, po])
    n_pairs = cfg.max_episodes // 2
    out = []
    for _ in range(n_runs):
        wins = rng.random((n_pairs, 2)) < p
        seq: npt.NDArray[np.int8] = np.where(wins, 1, -1).astype(np.int8).reshape(-1)  # ordre (paire, côté)
        r = sprt_scan(seq, cfg, check_every=2)
        out.append((r.decision, r.n_episodes))
    return out


def test_sprt_retrospectif_avec_fort_avantage_de_cote() -> None:
    """Avantage de côté de 200 Elo (le bleu gagne 76 % des parties décisives à force égale). La somme des deux épisodes
    d'une paire a la bonne espérance sous H0 (σ(s) + σ(-s) = 1) et une variance PLUS PETITE que le modèle i.i.d.
    (conservateur). Sous Δ = +50 en revanche l'espérance d'une paire vaut σ(250) + σ(-150) = 1,107 < 2 σ(50) = 1,143 :
    l'effet est sous-estimé (≈ +37 Elo effectifs) et la puissance baisse. Mesuré ci-dessous, pas supposé."""
    n = 1500
    h0 = _pair_runs(0.0, 200.0, n, seed=10)
    alpha_hat = sum(d is Decision.ACCEPT_H1 for d, _ in h0) / n
    assert alpha_hat <= 0.05 / 0.95 + SE_FACTOR * _se(0.05, n)  # erreur de type I toujours bornée
    assert all(m % 2 == 0 or m == 1200 for _, m in h0)  # décisions aux frontières de paires
    h1 = _pair_runs(50.0, 200.0, n, seed=11)
    power = sum(d is Decision.ACCEPT_H1 for d, _ in h1) / n
    ref = _pair_runs(50.0, 0.0, n, seed=12)  # même Δ sans avantage de côté
    power_ref = sum(d is Decision.ACCEPT_H1 for d, _ in ref) / n
    assert power_ref >= 0.947 - SE_FACTOR * _se(0.95, n)
    assert power < power_ref  # la perte de puissance existe (documentée dans sprt.py) ...
    assert power > 0.75  # ... mais reste modeste : mesuré 0,824 (contre 0,952 sans avantage de côté), α̂ = 0,011


# ------------------------------------------------------------------------------------------------ rating


def _synthetic_wins(elo: npt.NDArray[np.float64], n_games: int, seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    n = elo.size
    wins = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = rng.binomial(n_games, elo_to_score(float(elo[i] - elo[j])))
            wins[i, j], wins[j, i] = w, n_games - w
    return wins


TRUE_ELO = np.array([0.0, 120.0, -60.0, 250.0])


def test_ic_bootstrap_couvre_la_verite() -> None:
    wins = _synthetic_wins(TRUE_ELO, n_games=300, seed=2)
    r = rate_players(wins, anchor=0, n_boot=300, seed=7)
    assert (r.ci_low <= r.elo).all() and (r.elo <= r.ci_high).all()
    # couverture non garantie par joueur ; graine fixe, vérifiée à la mesure (ce test fige la graine 2)
    assert ((r.ci_low <= TRUE_ELO) & (TRUE_ELO <= r.ci_high)).all()
    width = (r.ci_high - r.ci_low)[1:]
    assert (width > 0).all() and (width < 120).all()


def test_largeur_de_l_ic_decroit_avec_le_nombre_de_parties() -> None:
    small = rate_players(_synthetic_wins(TRUE_ELO, 100, 3), n_boot=200, seed=0)
    big = rate_players(_synthetic_wins(TRUE_ELO, 1600, 3), n_boot=200, seed=0)
    assert ((big.ci_high - big.ci_low)[1:] < (small.ci_high - small.ci_low)[1:]).all()


def test_prior_avec_bootstrap_fini_et_degenere_signale() -> None:
    wins = np.array([[0.0, 50.0, 50.0], [0.0, 0.0, 30.0], [0.0, 20.0, 0.0]])  # joueur 0 invaincu
    res = rate_players(wins, anchor=1, n_boot=100, seed=0, prior=0.5)
    assert res.regularized and res.degenerate.tolist() == [True, False, False]
    assert np.isfinite(res.ci_low).all() and np.isfinite(res.ci_high).all()
    assert res.ci_low[0] > 0  # l'invaincu reste classé au-dessus de l'ancre dans tout l'IC
