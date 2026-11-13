"""SPRT (`eval.sprt`), intervalle de Wilson et critère sur les décisifs : contrôles déterministes et rapides.

Les simulations Monte-Carlo (erreurs nominales, nombre moyen d'épisodes, nuls, côtés) sont dans
`tests/integration/test_eval_stats.py`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.rating import wilson_interval
from eval.sprt import (
    Decision,
    DecisiveVerdict,
    SPRTConfig,
    decisive_win_test,
    sprt_scan,
    sprt_status,
)


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
    with pytest.raises(ValueError):
        SPRTConfig(max_decisive=0)


def test_les_nuls_ne_changent_pas_le_log_rapport() -> None:
    cfg = SPRTConfig()
    base = sprt_status(30, 0, 20, cfg)
    with_draws = sprt_status(30, 400, 20, cfg)
    assert with_draws.llr == base.llr  # modèle de Davidson : les nuls n'informent pas sur Δ
    assert with_draws.n_episodes == 450 and with_draws.n_decisive == 50


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


def test_plafond_en_parties_decisives_et_les_deux_compteurs() -> None:
    cfg = SPRTConfig(max_episodes=10_000, max_decisive=40)
    r = sprt_status(20, 900, 19, cfg)  # 39 décisifs, 939 épisodes : on continue
    assert r.decision is Decision.CONTINUE and (r.n_episodes, r.n_decisive) == (939, 39)
    r = sprt_status(20, 900, 20, cfg)  # 40 décisifs : plafond décisif atteint, loin du plafond d'épisodes
    assert r.decision is Decision.CAP_REACHED and (r.n_episodes, r.n_decisive) == (940, 40)
    d = r.as_dict()
    assert d["n_episodes"] == 940 and d["n_decisive"] == 40
    # une borne de Wald franchie avant le plafond l'emporte
    assert sprt_status(40, 0, 0, SPRTConfig(max_decisive=40)).decision is Decision.ACCEPT_H1


def test_scan_avec_plafond_decisif_s_arrete_au_bon_episode() -> None:
    cfg = SPRTConfig(max_episodes=10_000, max_decisive=6)
    seq = np.array([0, 0, 1, 0, -1, 0, 0, 1, -1, 0, 1, 0, -1, 0], dtype=np.int8)  # 6e décisif à l'épisode 13
    r = sprt_scan(seq, cfg)
    assert r.decision is Decision.CAP_REACHED and r.n_decisive == 6 and r.n_episodes == 13
    r2 = sprt_scan(seq, cfg, check_every=2)  # évalué aux frontières de paires : 14 épisodes, 6 décisifs
    assert r2.n_episodes == 14 and r2.n_decisive == 6


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


# ------------------------------------------------------------------------------------------------ Wilson


def test_wilson_valeur_de_reference_et_bords() -> None:
    low, high = wilson_interval(8, 10)  # référence calculée à la main : (0,4902 ; 0,9434)
    assert low == pytest.approx(0.4902, abs=5e-4) and high == pytest.approx(0.9434, abs=5e-4)
    lo0, hi0 = wilson_interval(0, 20)
    # défini à 0 succès (Wald donnerait un intervalle nul) ; abs=1e-12 : arrondi flottant ~1e-17
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0.0 < hi0 < 0.2
    lo1, hi1 = wilson_interval(20, 20)
    assert hi1 == pytest.approx(1.0, abs=1e-12) and lo1 == pytest.approx(1.0 - hi0)  # symétrie p <-> 1 - p
    low_one, _ = wilson_interval(8, 10, two_sided=False)
    assert low_one > low  # unilatéral : z = 1,645 contre 1,96
    narrow = wilson_interval(800, 1000)
    assert narrow[1] - narrow[0] < high - low
    for bad in ((1, 0), (-1, 5), (6, 5)):
        with pytest.raises(ValueError):
            wilson_interval(*bad)
    with pytest.raises(ValueError):
        wilson_interval(1, 2, alpha=1.0)


# ------------------------------------------------------------------------------------------------ critère décisif


def test_critere_decisif_trois_issues() -> None:
    passed = decisive_win_test(40, 10)  # 80 % sur 50 : borne basse de Wilson > 0,5
    assert passed.verdict is DecisiveVerdict.PASSED and passed.wilson_low is not None and passed.wilson_low > 0.5
    failed = decisive_win_test(30, 25)  # 55 % sur 55 : borne basse < 0,5
    assert failed.verdict is DecisiveVerdict.FAILED and failed.wilson_low is not None and failed.wilson_low < 0.5
    assert decisive_win_test(5, 25).verdict is DecisiveVerdict.FAILED  # franchement moins bon : échoué, pas indéterminé


def test_critere_decisif_jamais_reussi_faute_de_donnees() -> None:
    for w, lo in ((0, 0), (5, 0), (19, 0), (3, 1)):
        r = decisive_win_test(w, lo)  # < 20 décisifs
        assert r.verdict is DecisiveVerdict.INDETERMINATE
    # 5 sur 5 : la borne de Wilson dépasserait 0,5 mais le seuil min_decisive l'interdit
    five = decisive_win_test(5, 0)
    assert five.wilson_low is not None and five.wilson_low > 0.5
    # mêmes comptes, seuil abaissé : la décision change, ce qui prouve que le seuil est ce qui protège
    assert decisive_win_test(5, 0, min_decisive=5).verdict is DecisiveVerdict.PASSED
    assert decisive_win_test(20, 0).verdict is DecisiveVerdict.PASSED  # 20 décisifs : seuil atteint
    none = decisive_win_test(0, 0)
    assert none.win_rate is None and none.wilson_low is None and none.decisive == 0
    assert decisive_win_test(25, 0, alpha=0.05).as_dict()["verdict"] == "passed"


def test_critere_decisif_validation() -> None:
    with pytest.raises(ValueError):
        decisive_win_test(-1, 3)
    with pytest.raises(ValueError):
        decisive_win_test(3, 3, min_decisive=0)
