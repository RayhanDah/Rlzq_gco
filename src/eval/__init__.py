"""Évaluation par matchs (`docs/01_papers.md` §1.9) : jeu apparié, Elo Bradley-Terry, SPRT."""

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
from eval.rating import (
    DegenerateRatingError,
    RatingResult,
    elo_to_score,
    fit_bradley_terry,
    rate_players,
    score_to_elo,
)
from eval.sprt import Decision, SPRTConfig, SPRTResult, sprt_scan, sprt_status

__all__ = [
    "Decision",
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
    "elo_to_score",
    "fit_bradley_terry",
    "load_checkpoint_policy",
    "load_policy",
    "play_match",
    "rate_players",
    "score_to_elo",
    "sprt_on_match",
    "sprt_scan",
    "sprt_status",
    "summarize",
]
