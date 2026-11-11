"""Récompense GCO (Goal/Concede Only), `docs/GCO_SPEC.md` §2–§3, ADR-002. Indépendante du simulateur."""

from rewards.config import (
    HASH_SCHEMA_VERSION,
    PHYSICS_TICK_RATE_HZ,
    SPEC_VERSION,
    RewardConfig,
    RewardVariant,
    VariantRules,
    rules_for,
)
from rewards.gco import (
    GoalNet,
    RewardFunction,
    StepOutcome,
    Team,
    episode_end,
    gco_team_rewards,
    make_reward,
    scorer_only_rewards,
)

__all__ = [
    "HASH_SCHEMA_VERSION",
    "PHYSICS_TICK_RATE_HZ",
    "SPEC_VERSION",
    "GoalNet",
    "RewardConfig",
    "RewardFunction",
    "RewardVariant",
    "StepOutcome",
    "Team",
    "VariantRules",
    "episode_end",
    "gco_team_rewards",
    "make_reward",
    "rules_for",
    "scorer_only_rewards",
]
