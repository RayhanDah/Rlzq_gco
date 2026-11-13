"""SPRT (`eval.sprt`) : bornes de Wald, modèle de nuls, erreurs nominales et nombre d'épisodes par simulation."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from eval.rating import elo_to_score
from eval.sprt import Decision, SPRTConfig, sprt_scan, sprt_status

N_RUNS = 4000
SE_FACTOR = 3.0  # tolérance = 3 écarts-types binomiaux (erreur de Monte-Carlo, graine fixe)


def _simulate(
    p_win: float, p_draw: float, cfg: SPRTConfig, seed: int, check_every: int = 1
) -> list[tuple[Decision, int]]:
    """N_RUNS tests sur des résultats i.i.d. : victoire p_win, nul p_draw, défaite le reste."""
    rng = np.random.default_rng(seed)
    u = rng.random((N_RUNS, cfg.max_episodes))
    outcomes: npt.NDArray[np.int8] = np.where(u < p_win, 1, np.where(u < p_win + p_draw, 0, -1)).astype(np.int8)
    out = []
    for row in outcomes:
        r = sprt_scan(row, cfg, check_every=check_every)
        out.append((r.decision, r.n_episodes))
    return out


def _decisive_p(elo: float) -> float:
    return elo_to_score(elo)


def test_bornes_de_wald_et_hypotheses() -> None:
    cfg = SPRTConfig()  # H0 : 0 Elo, H1 : +50 Elo, alpha = beta = 0,05, plafond 1200
    assert cfg.upper == pytest.approx(math.log(19), abs=1e-12)
    assert cfg.upper == pytest.approx(2.944, abs=5e-4)  # docs/01_papers.md §1.9
    assert cfg.lower == pytest.approx(-2.944, abs=5e-4)
    assert cfg.p0 == 0.5
    assert cfg.p1 == pytest.approx(0.5716, abs=5e-4)  # 0,572 : paper_logo_2202_and_refs.md §13.4
    asym = SPRTConfig(alpha=0.01, beta=0.10)
    assert asym.upper == pytest.approx(math.log(0.9 / 0.01))
    assert asym.lower == pytest.approx(math.log(0.10 / 0.99))


def test_hypotheses_invalides() -> None:
    with pytest.raises(ValueError):
        SPRTConfig(elo0=10.0, elo1=10.0)
    with pytest.raises(ValueError):
        SPRTConfig(alpha=0.0)
    with pytest.raises(ValueError):
        SPRTConfig(max_episodes=0)


def test_les_nuls_ne_changent_pas_le_log_rapport() -> None:
    cfg = SPRTConfig()
    base = sprt_status(30, 0, 20, cfg)
    with_draws = sprt_status(30, 400, 20, cfg)
    assert with_draws.llr == base.llr  # modèle de Davidson : les nuls n'informent pas sur Δ
    assert with_draws.n_episodes == 450


def test_decisions_continuer_h1_h0_plafond() -> None:
    cfg = SPRTConfig(max_episodes=100)
    assert sprt_status(5, 0, 5, cfg).decision is Decision.CONTINUE
    assert sprt_status(60, 0, 10, cfg).decision is Decision.ACCEPT_H1
    assert sprt_status(10, 0, 60, cfg).decision is Decision.ACCEPT_H0
    capped = sprt_status(20, 60, 20, cfg)  # llr = 0, 100 épisodes
    assert capped.decision is Decision.CAP_REACHED
    assert sprt_status(20, 60, 19, cfg).decision is Decision.CONTINUE  # 99 épisodes
    with pytest.raises(ValueError):
        sprt_status(-1, 0, 0, cfg)


def test_scan_s_arrete_au_premier_franchissement_et_aux_frontieres_de_blocs() -> None:
    cfg = SPRTConfig()
    wins = np.ones(200, dtype=np.int8)
    r1 = sprt_scan(wins, cfg, check_every=1)
    assert r1.decision is Decision.ACCEPT_H1
    assert r1.wins == math.ceil(cfg.upper / cfg.win_increment)  # premier franchissement exact
    r2 = sprt_scan(wins, cfg, check_every=2)
    assert r2.n_episodes % 2 == 0 and r2.n_episodes >= r1.n_episodes and r2.n_episodes - r1.n_episodes <= 1
    short = sprt_scan(np.array([1, -1, 1, -1], dtype=np.int8), cfg, check_every=2)
    assert short.decision is Decision.CONTINUE and short.n_episodes == 4
    with pytest.raises(ValueError):
        sprt_scan(np.array([2]), cfg)
    assert sprt_scan(np.array([], dtype=np.int8), cfg).decision is Decision.CONTINUE


def test_plafond_hors_frontiere_de_bloc_evalue_au_plafond() -> None:
    cfg = SPRTConfig(max_episodes=7)
    r = sprt_scan(np.array([1, -1] * 10, dtype=np.int8), cfg, check_every=2)
    assert r.decision is Decision.CAP_REACHED and r.n_episodes == 7


def test_erreur_de_type_i_sous_h0_conforme_a_alpha() -> None:
    cfg = SPRTConfig()
    res = _simulate(0.5, 0.0, cfg, seed=1)
    alpha_hat = sum(d is Decision.ACCEPT_H1 for d, _ in res) / N_RUNS
    se = math.sqrt(0.05 * 0.95 / N_RUNS)
    # Wald : α_réel <= α / (1 - β) = 0,0526 (le dépassement des bornes rend le test conservateur) ;
    # borne basse 0,03 : l'écart au nominal dû au dépassement reste modeste (mesuré 0,043-0,050 dans le doc §13.4).
    assert 0.03 < alpha_hat <= 0.05 / 0.95 + SE_FACTOR * se
    assert sum(d is Decision.CAP_REACHED for d, _ in res) / N_RUNS < 0.05  # plafond 1200 rarement atteint


def test_puissance_sous_h1_conforme_a_1_moins_beta() -> None:
    cfg = SPRTConfig()
    res = _simulate(_decisive_p(50.0), 0.0, cfg, seed=2)
    power = sum(d is Decision.ACCEPT_H1 for d, _ in res) / N_RUNS
    se = math.sqrt(0.95 * 0.05 / N_RUNS)
    assert power >= 0.95 - SE_FACTOR * se - 0.01 and power < 0.995  # 0,01 : β réel <= β/(1-α) = 0,0526


def test_nombre_moyen_d_episodes_ordre_de_grandeur_du_doc() -> None:
    """docs/01_papers.md §1.9 : ≈ 260 épisodes en moyenne à +50 Elo (Wald : 257 sous H0, 259 sous H1)."""
    cfg = SPRTConfig()
    for seed, p in ((3, 0.5), (4, _decisive_p(50.0))):
        n = np.array([m for _, m in _simulate(p, 0.0, cfg, seed=seed)], dtype=np.float64)
        assert 257 * 0.9 < n.mean() < 259 * 1.1  # ±10 % (SE de la moyenne ≈ 3, dépassement des bornes en sus)


def test_avec_nuls_les_erreurs_sont_inchangees_et_n_est_divise_par_la_part_decisive() -> None:
    cfg = SPRTConfig(max_episodes=4000)
    res = _simulate(0.5 * 0.5, 0.5, cfg, seed=5)  # 50 % de nuls, H0 vraie
    alpha_hat = sum(d is Decision.ACCEPT_H1 for d, _ in res) / N_RUNS
    assert alpha_hat <= 0.05 / 0.95 + SE_FACTOR * math.sqrt(0.05 * 0.95 / N_RUNS)
    res1 = _simulate(0.5 * _decisive_p(50.0) * 1.0, 0.5, cfg, seed=6)  # H1 vraie, 50 % de nuls
    n = np.mean([m for _, m in res1])
    assert 2 * 257 * 0.9 < n < 2 * 259 * 1.1  # deux fois plus d'épisodes à 50 % de nuls


def test_plafond_reduit_donne_cap_reached_et_n_borne() -> None:
    cfg = SPRTConfig(max_episodes=100)
    res = _simulate(0.5, 0.0, cfg, seed=7)
    assert all(m <= 100 for _, m in res)
    assert any(d is Decision.CAP_REACHED for d, _ in res)
