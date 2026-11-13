"""Gardes (T5) : une règle par test ; fonctions pures (`evaluate` ne modifie ni l'état ni les stats)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from training.guards import (  # noqa: E402
    Action,
    GuardConfig,
    GuardObservation,
    GuardState,
    evaluate,
    grads_finite,
    params_finite,
)

CFG = GuardConfig()
OK = {
    "policy_loss": 0.1,
    "value_loss": 0.2,
    "entropy": 3.0,
    "approx_kl": 0.01,
    "grad_norm": 0.3,
    "explained_variance": 0.5,
}


def _run(stats: dict[str, float], state: GuardState | None = None, **kw: object):  # type: ignore[no-untyped-def]
    return evaluate(CFG, state or GuardState(), GuardObservation(stats, **kw))  # type: ignore[arg-type]


def test_defaults_are_plan_md_values() -> None:
    assert (CFG.kl_threshold, CFG.kl_patience, CFG.entropy_floor_frac, CFG.ev_patience) == (0.05, 5, 0.10, 50)
    assert CFG.entropy_floor == pytest.approx(0.1 * math.log(90))


def test_continue_on_healthy_iteration() -> None:
    state, dec = _run(OK)
    assert dec.action is Action.CONTINUE and dec.healthy and dec.rule is None and state == GuardState()


@pytest.mark.parametrize("key", ["policy_loss", "value_loss", "grad_norm", "entropy"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_stat_restores(key: str, bad: float) -> None:
    _, dec = _run({**OK, key: bad})
    assert dec.action is Action.RESTORE and dec.rule == "nonfinite" and key in dec.reason and not dec.healthy


def test_nonfinite_params_or_grads_restore() -> None:
    assert _run(OK, params_finite=False)[1].action is Action.RESTORE
    assert _run(OK, grads_finite=False)[1].reason.endswith("grads")


def test_kl_needs_five_consecutive_iterations() -> None:
    state = GuardState()
    high = {**OK, "approx_kl": 0.06}
    for i in range(4):
        state, dec = _run(high, state)
        assert dec.action is Action.CONTINUE and not dec.healthy and state.kl_streak == i + 1
    state, dec = _run(high, state)
    assert dec.action is Action.RESTORE and dec.rule == "kl"
    assert state.kl_streak == 0 and state.restores == 1  # séries remises à zéro après restauration


def test_kl_streak_resets_when_below_threshold() -> None:
    state = GuardState(kl_streak=4)
    state, dec = _run({**OK, "approx_kl": 0.05}, state)  # seuil strict : 0,05 n'est pas > 0,05
    assert state.kl_streak == 0 and dec.action is Action.CONTINUE and dec.healthy


def test_entropy_collapse_before_s1_stops_and_is_off_after() -> None:
    low = {**OK, "entropy": 0.05 * math.log(90)}
    assert _run(low)[1].action is Action.STOP and _run(low)[1].rule == "entropy"
    assert _run(low, s1_reached=True)[1].action is Action.CONTINUE
    assert _run({**OK, "entropy": 0.11 * math.log(90)})[1].action is Action.CONTINUE


def test_negative_explained_variance_for_50_iterations_after_s1_stops() -> None:
    state = GuardState()
    neg = {**OK, "explained_variance": -0.1}
    for _ in range(49):
        state, dec = _run(neg, state, s1_reached=True)
        assert dec.action is Action.CONTINUE
    _, dec = _run(neg, state, s1_reached=True)
    assert dec.action is Action.STOP and dec.rule == "explained_variance"
    assert _run(neg, GuardState(ev_streak=49))[1].action is Action.CONTINUE  # avant S1 : règle inactive


def test_ev_streak_resets_on_positive() -> None:
    state, _ = _run({**OK, "explained_variance": 0.0}, GuardState(ev_streak=10), s1_reached=True)
    assert state.ev_streak == 0  # 0 n'est pas < 0


def test_consecutive_restores_cap_prevents_endless_loop_and_healthy_iteration_resets_streak() -> None:
    nan = {**OK, "policy_loss": float("nan")}
    state = GuardState()
    for _ in range(CFG.max_consecutive_restores):
        state, dec = _run(nan, state)
        assert dec.action is Action.RESTORE
    _, dec = _run(nan, state)
    assert dec.action is Action.STOP and dec.rule == "max_consecutive_restores"
    state, _ = _run(OK, state)
    assert state.restores == 0 and state.restores_total == CFG.max_consecutive_restores  # le cumul ne se remet pas à 0


def test_max_restores_counts_cumulative_restores_across_healthy_iterations() -> None:
    # Revue T6 (A1.4) : avant, la remise à zéro par une itération saine rendait la limite sans effet sur un run qui
    # diverge de temps en temps.
    nan = {**OK, "policy_loss": float("nan")}
    cfg = replace(CFG, max_restores=2, max_consecutive_restores=5)
    state = GuardState()
    for _ in range(2):
        state, dec = evaluate(cfg, state, GuardObservation(nan))
        assert dec.action is Action.RESTORE
        state, dec = evaluate(cfg, state, GuardObservation(OK))  # itération saine entre deux
        assert dec.action is Action.CONTINUE and state.restores == 0
    _, dec = evaluate(cfg, state, GuardObservation(nan))
    assert dec.action is Action.STOP and dec.rule == "max_restores" and "2" in dec.reason


def test_restore_limits_are_validated() -> None:
    with pytest.raises(ValueError):
        GuardConfig(max_consecutive_restores=-1)
    assert GuardConfig().max_restores == 3 and GuardConfig().max_consecutive_restores == 2


def test_old_guard_state_dict_without_restores_total_still_loads() -> None:
    # Un checkpoint antérieur à A1 n'a pas `restores_total` : GuardState(**dict) reste valide.
    assert GuardState(**{"kl_streak": 1, "entropy_streak": 0, "ev_streak": 0, "restores": 1}).restores_total == 0


def test_evaluate_is_pure() -> None:
    state, stats = GuardState(kl_streak=2), dict(OK)
    out1 = evaluate(CFG, state, GuardObservation(stats))
    out2 = evaluate(CFG, state, GuardObservation(stats))
    assert out1 == out2 and state == GuardState(kl_streak=2) and stats == OK


def test_thresholds_are_configurable() -> None:
    cfg = replace(CFG, kl_threshold=0.01, kl_patience=1)
    _, dec = evaluate(cfg, GuardState(), GuardObservation({**OK, "approx_kl": 0.02}))
    assert dec.action is Action.RESTORE
    with pytest.raises(ValueError):
        GuardConfig(kl_patience=0)


def test_params_and_grads_finite_helpers() -> None:
    p = torch.nn.Parameter(torch.ones(3))
    q = torch.nn.Parameter(torch.ones(2))
    assert params_finite([p, q]) and grads_finite([p, q])  # pas de gradient : vrai
    q.grad = torch.tensor([1.0, float("inf")])
    assert not grads_finite([p, q]) and params_finite([p, q])
    with torch.no_grad():
        p[1] = float("nan")
    assert not params_finite([p, q])
