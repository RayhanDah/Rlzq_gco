"""Observations (ADR-003 §3) : `BatchState` (§2), schéma, pads, builder."""

from obs.builder import build_obs
from obs.pads import (
    PAD_MIRROR_PERM,
    PAD_POSITIONS_ROCKETGPU,
    PAD_POSITIONS_ROCKETSIM,
    ROCKETGPU_TO_ROCKETSIM,
    ROCKETSIM_TO_ROCKETGPU,
)
from obs.schema import OBS_DIM, OBS_SCHEMA_VERSION, obs_dim
from obs.state import (
    BOOST_MAX,
    N_PADS,
    QUAT_NORM_TOL,
    BatchState,
    default_car_team,
)

__all__ = [
    "BOOST_MAX",
    "N_PADS",
    "OBS_DIM",
    "OBS_SCHEMA_VERSION",
    "PAD_MIRROR_PERM",
    "PAD_POSITIONS_ROCKETGPU",
    "PAD_POSITIONS_ROCKETSIM",
    "QUAT_NORM_TOL",
    "ROCKETGPU_TO_ROCKETSIM",
    "ROCKETSIM_TO_ROCKETGPU",
    "BatchState",
    "build_obs",
    "default_car_team",
    "obs_dim",
]
