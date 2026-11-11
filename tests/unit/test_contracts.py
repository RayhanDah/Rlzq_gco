"""Cohérence des constantes dupliquées entre modules écrits en parallèle (ADR-003 §3, §4, §8)."""

from __future__ import annotations

import actions
import models
from obs import OBS_DIM, obs_dim


def test_policy_head_matches_action_table() -> None:
    # Revue T4 (2026-11-11) : `models.policy` duplique N_ACTIONS pour rester autonome ; `actions.table` fait foi.
    assert models.N_ACTIONS == actions.N_ACTIONS == actions.ACTION_TABLE.shape[0]


def test_policy_accepts_obs_dimension() -> None:
    policy = models.PolicyNet(OBS_DIM)
    value = models.ValueNet(OBS_DIM)

    first_policy_layer = next(policy.parameters())
    first_value_layer = next(value.parameters())

    assert OBS_DIM == obs_dim(2) == 109
    assert first_policy_layer.shape[1] == OBS_DIM
    assert first_value_layer.shape[1] == OBS_DIM
