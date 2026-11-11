"""Environnement batché (ADR-003 §5) : `BatchEnv`, interface `Simulator`, états initiaux (spec §2.3)."""

from env.batch_env import BatchEnv, ObsBuilder
from env.initial_states import InitialDraw, InitialStates, InitialStateSampler, sample_generic_states
from env.sim_adapter import RocketGPUSimulator, Simulator, SimulatorUnavailableError

__all__ = [
    "BatchEnv",
    "InitialDraw",
    "InitialStateSampler",
    "InitialStates",
    "ObsBuilder",
    "RocketGPUSimulator",
    "Simulator",
    "SimulatorUnavailableError",
    "sample_generic_states",
]
