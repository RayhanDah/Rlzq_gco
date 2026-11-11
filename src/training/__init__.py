"""Entraînement PPO v0 : tampon de rollout, GAE, mise à jour PPO et recouvrement collecte/apprentissage (ADR-003 §6).

Les checkpoints, gardes et journaux (T5) vivent dans des modules séparés, non importés ici.
"""

from training.gae import bootstrap_values, gae
from training.ppo import (
    IterationResult,
    Learner,
    LearnerError,
    OptimizerLike,
    OverlapTrainer,
    PPOConfig,
    PPOStats,
    make_optimizer,
    resolve_device,
)
from training.rollout import Collector, RolloutBatch, RolloutBuffer, VecEnv

__all__ = [
    "Collector",
    "IterationResult",
    "Learner",
    "LearnerError",
    "OptimizerLike",
    "OverlapTrainer",
    "PPOConfig",
    "PPOStats",
    "RolloutBatch",
    "RolloutBuffer",
    "VecEnv",
    "bootstrap_values",
    "gae",
    "make_optimizer",
    "resolve_device",
]
