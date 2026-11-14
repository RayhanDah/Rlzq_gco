"""Évaluation par matchs (`docs/01_papers.md` §1.9) : jeu apparié, Elo Bradley-Terry, SPRT.

`eval.rating` et `eval.sprt` ne dépendent que de numpy. Les symboles de `eval.match` (qui importe torch, l'env et les
modèles) sont résolus à la demande (`from eval import play_match` fonctionne, mais `import eval` reste léger).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from eval.rating import (
    DegenerateRatingError,
    RatingResult,
    elo_to_score,
    fit_bradley_terry,
    garwood_interval,
    poisson_cdf,
    rate_players,
    score_to_elo,
    wilson_interval,
)
from eval.sprt import (
    Decision,
    DecisiveTestResult,
    DecisiveVerdict,
    SPRTConfig,
    SPRTResult,
    decisive_win_test,
    sprt_scan,
    sprt_status,
)

if TYPE_CHECKING:
    from eval.match import (
        MatchResult,
        MatchSummary,
        NetPolicy,
        Policy,
        PolicyInfo,
        RandomPolicy,
        load_checkpoint_policy,
        load_policy,
        play_match,
        sprt_on_match,
        summarize,
    )

_MATCH_NAMES = (
    "MatchResult",
    "MatchSummary",
    "NetPolicy",
    "Policy",
    "PolicyInfo",
    "RandomPolicy",
    "load_checkpoint_policy",
    "load_policy",
    "play_match",
    "sprt_on_match",
    "summarize",
)


def __getattr__(name: str) -> Any:
    if name in _MATCH_NAMES:
        return getattr(importlib.import_module("eval.match"), name)
    raise AttributeError(f"module 'eval' has no attribute {name!r}")


__all__ = [
    "Decision",
    "DecisiveTestResult",
    "DecisiveVerdict",
    "DegenerateRatingError",
    "MatchResult",
    "MatchSummary",
    "NetPolicy",
    "Policy",
    "PolicyInfo",
    "RandomPolicy",
    "RatingResult",
    "SPRTConfig",
    "SPRTResult",
    "decisive_win_test",
    "elo_to_score",
    "fit_bradley_terry",
    "garwood_interval",
    "load_checkpoint_policy",
    "load_policy",
    "play_match",
    "poisson_cdf",
    "rate_players",
    "score_to_elo",
    "sprt_on_match",
    "sprt_scan",
    "sprt_status",
    "summarize",
    "wilson_interval",
]
