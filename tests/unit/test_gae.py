"""GAE avec terminal (but), troncature (bootstrap sur V(s′) de l'obs finale) et fin de lot (ADR-003 §6).

Référence calculée à la main, pas à pas, dans les commentaires. Tolérance 1e-6 : les entrées sont des décimaux courts
calculés en float32 (erreur d'arrondi relative ≈ 6e-8 par opération, quelques opérations par terme).
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from training.gae import bootstrap_values, gae  # noqa: E402

GAMMA, LAM = 0.9, 0.8  # γλ = 0,72
ATOL = 1e-6


def _case() -> dict[str, Any]:
    # T = 4 steps, N = 2 agents ; V[t, a].
    values = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    last_value = torch.tensor([0.9, 1.0])  # V(last_obs), bootstrap à t = T−1
    rew = torch.tensor([[0.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 0.0]])
    term = torch.tensor([[False, False], [False, True], [True, False], [False, False]])
    # agent 1 en t = 1 : but ET timeout le même step ⇒ le but prime (pas de ligne finale) ;
    # agent 1 en t = 3 : troncature au dernier step du lot ⇒ V(final_obs) remplace V(last_obs).
    trunc = torch.tensor([[False, False], [True, True], [False, False], [False, True]])
    final_t = torch.tensor([1, 3])
    final_agent = torch.tensor([0, 1])
    final_value = torch.tensor([0.5, -0.25])  # V(s′) des obs finales stockées
    return {
        "values": values,
        "last_value": last_value,
        "rew": rew,
        "term": term,
        "trunc": trunc,
        "final_t": final_t,
        "final_agent": final_agent,
        "final_value": final_value,
    }


def test_bootstrap_values_hand_reference() -> None:
    # Arrange
    c = _case()
    # Act
    v_next = bootstrap_values(c["values"], c["last_value"], c["final_value"], c["final_t"], c["final_agent"], c["term"])
    # Assert
    expected = torch.tensor(
        [
            [0.3, 0.4],  # V(obs[1])
            [0.5, 0.0],  # a0 tronqué : V(final) = 0,5 (pas V(obs[2]), premier état du nouvel épisode) ; a1 but : 0
            [0.0, 0.8],  # a0 but : 0 ; a1 : V(obs[3])
            [0.9, -0.25],  # a0 : V(last_obs) ; a1 tronqué en fin de lot : V(final) = −0,25
        ]
    )
    torch.testing.assert_close(v_next, expected, atol=ATOL, rtol=0)


def test_gae_hand_reference_terminal_truncation_and_batch_end() -> None:
    # Arrange
    c = _case()
    v_next = bootstrap_values(c["values"], c["last_value"], c["final_value"], c["final_t"], c["final_agent"], c["term"])
    # Act
    adv, ret = gae(c["rew"], c["values"], v_next, c["term"], c["trunc"], GAMMA, LAM)
    # Assert — calcul à la main, de t = 3 à t = 0.
    # Agent 0 :
    #   t3 : δ = 0 + 0,9·0,9 − 0,7 = 0,11 ; fin de lot, A = 0,11
    #   t2 : but : δ = 1 + 0 − 0,5 = 0,5 ; trace coupée, A = 0,5
    #   t1 : tronqué : δ = 0 + 0,9·0,5 − 0,3 = 0,15 ; trace coupée, A = 0,15
    #   t0 : δ = 0 + 0,9·0,3 − 0,1 = 0,17 ; A = 0,17 + 0,72·0,15 = 0,278
    # Agent 1 :
    #   t3 : tronqué en fin de lot : δ = 0 + 0,9·(−0,25) − 0,8 = −1,025 ; A = −1,025
    #   t2 : δ = 0 + 0,9·0,8 − 0,6 = 0,12 ; A = 0,12 + 0,72·(−1,025) = −0,618
    #   t1 : but (et timeout) : δ = −1 + 0 − 0,4 = −1,4 ; trace coupée, A = −1,4
    #   t0 : δ = 0 + 0,9·0,4 − 0,2 = 0,16 ; A = 0,16 + 0,72·(−1,4) = −0,848
    expected_adv = torch.tensor([[0.278, -0.848], [0.15, -1.4], [0.5, -0.618], [0.11, -1.025]])
    torch.testing.assert_close(adv, expected_adv, atol=ATOL, rtol=0)
    torch.testing.assert_close(ret, expected_adv + c["values"], atol=ATOL, rtol=0)


def test_truncation_bootstraps_on_final_obs_not_on_current_value() -> None:
    """Régression du défaut du prototype `bench_e2e.gae` (bootstrap sur V(s_t)) : seule V(final_obs) compte."""
    # Arrange : un step, un agent tronqué, V(s_t) = 0,3, V(s′) = 0,5.
    values, rew = torch.tensor([[0.3]]), torch.tensor([[0.0]])
    term, trunc = torch.tensor([[False]]), torch.tensor([[True]])
    # Act
    v_next = bootstrap_values(
        values, torch.tensor([7.0]), torch.tensor([0.5]), torch.tensor([0]), torch.tensor([0]), term
    )
    adv, _ = gae(rew, values, v_next, term, trunc, GAMMA, LAM)
    # Assert : A = γ·V(s′) − V(s_t) = 0,45 − 0,3 ; le prototype aurait donné 0,9·0,3 − 0,3 = −0,03.
    torch.testing.assert_close(adv, torch.tensor([[0.15]]), atol=ATOL, rtol=0)


def test_gae_without_cuts_is_discounted_sum_when_lambda_one() -> None:
    # Arrange : λ = 1, aucun terminal ⇒ A_t + V_t = Σ γ^k r_{t+k} + γ^(T−t) V(last).
    t_len, gamma = 5, 0.95
    rew = torch.arange(t_len, dtype=torch.float32).unsqueeze(1) * 0.1
    values = torch.full((t_len, 1), 0.3)
    none = torch.zeros(t_len, 1, dtype=torch.bool)
    last = torch.tensor([2.0])
    empty = torch.zeros(0, dtype=torch.long)
    v_next = bootstrap_values(values, last, torch.zeros(0), empty, empty, none)
    # Act
    _, ret = gae(rew, values, v_next, none, none, gamma, 1.0)
    # Assert
    for t in range(t_len):
        mc = sum(gamma**k * float(rew[t + k, 0]) for k in range(t_len - t)) + gamma ** (t_len - t) * 2.0
        assert abs(float(ret[t, 0]) - mc) < 1e-5  # somme de ≤ 6 termes float32 d'ordre 1


def test_input_validation() -> None:
    values = torch.zeros(2, 3)
    none = torch.zeros(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError):
        bootstrap_values(values, torch.zeros(2), torch.zeros(0), torch.zeros(0, dtype=torch.long),
                         torch.zeros(0, dtype=torch.long), none)  # fmt: skip
    with pytest.raises(ValueError):
        bootstrap_values(values, torch.zeros(3), torch.zeros(1), torch.zeros(2, dtype=torch.long),
                         torch.zeros(1, dtype=torch.long), none)  # fmt: skip
    with pytest.raises(ValueError):
        gae(torch.zeros(2, 3), values, torch.zeros(3, 2), none, none, 0.9, 0.9)
    with pytest.raises(ValueError):
        gae(torch.zeros(2, 3), values, values, none, none, 1.5, 0.9)
