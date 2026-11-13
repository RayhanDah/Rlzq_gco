"""Évaluation par matchs sur RocketGPU CPU (build local, non versionné). Sauté si RocketGPU ou les meshes sont absents.

Environnement : `ROCKETGPU_PY_DIR`, `ROCKETGPU_COLLISION_MESH_DIR` ; optionnel `ROCKETGPU_DLL_DIR`.
"""

from __future__ import annotations

import numpy as np
import pytest

from env import RocketGPUSimulator, SimulatorUnavailableError
from eval.match import DRAW, LOSS, WIN, MatchResult, RandomPolicy, pair_seeds, play_match, summarize
from rewards import RewardConfig

try:
    _PROBE = RocketGPUSimulator(2)
    _SKIP: str | None = None
except (SimulatorUnavailableError, OSError) as exc:
    _SKIP = f"RocketGPU indisponible : {exc}"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

N_ENVS = 32
TICK_SKIP = 8
CFG = RewardConfig()  # spec : 20 s sans touche, 120 s de durée max


def test_kickoff_determine_par_arene_et_graine_pas_par_la_graine_seule() -> None:
    """Condition du jeu apparié, MESURÉE : même (arène, graine) => mêmes positions ; même graine dans deux arènes
    différentes => positions pas toujours identiques (d'où une paire jouée dans UNE arène, deux épisodes de suite)."""
    sim = RocketGPUSimulator(4, 2, TICK_SKIP)
    ids = np.arange(4, dtype=np.int64)
    sim.reset_to_kickoff(ids, np.full(4, 5, dtype=np.uint64))
    first = sim.state().car_pos.copy()
    sim.reset_to_kickoff(ids, np.full(4, 5, dtype=np.uint64))
    assert np.array_equal(first, sim.state().car_pos)  # reproductible par arène
    differs = False
    for seed in range(1, 40):
        sim.reset_to_kickoff(ids, np.full(4, seed, dtype=np.uint64))
        pos = sim.state().car_pos
        differs |= not all(np.array_equal(pos[0], pos[a]) for a in range(1, 4))
    assert differs  # mesuré : la graine seule ne fixe pas le kickoff entre arènes
    spots = set()
    for seed in pair_seeds(0, 60):
        sim.reset_to_kickoff(np.array([0], dtype=np.int64), np.array([seed], dtype=np.uint64))
        spots.add(tuple(np.round(sim.state().car_pos[0, 0, :2]).astype(int)))
    assert len(spots) == 5  # les graines couvrent les 5 kickoffs officiels


def _run() -> MatchResult:
    sim = RocketGPUSimulator(N_ENVS, 2, TICK_SKIP)
    return play_match(RandomPolicy(1), RandomPolicy(2), sim, 64, seed=3, reward_config=CFG)


def test_agent_aleatoire_contre_agent_aleatoire_sur_kickoffs_reels() -> None:
    res = _run()
    assert res.n_pairs == 32 and res.outcome.shape == (32, 2)
    assert np.isin(res.outcome, (WIN, DRAW, LOSS)).all()
    assert (res.ticks > 0).all() and (res.ticks % TICK_SKIP == 0).all()
    assert (res.ticks <= CFG.max_episode_ticks).all()
    # un nul est une troncature (no-touch >= 20 s ou durée max) ; un but met fin à l'épisode avant
    assert (res.ticks[res.outcome == DRAW] >= CFG.no_touch_ticks).all()
    s = summarize(res, n_boot=200, seed=0)
    assert s.minutes_played > 0 and s.mean_episode_s <= CFG.max_episode_s
    assert res.env_steps > 0 and res.wall_s > 0


def test_determinisme_a_seed_fixe_avec_rocketgpu() -> None:
    assert _run().equals(_run())
