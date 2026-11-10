"""Tests du prototype bout-en-bout (fonctions pures : actions, quaternions, obs, GAE). Sans simulateur."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench_e2e", ROOT / "scripts" / "bench_e2e.py")
assert spec and spec.loader
e2e = importlib.util.module_from_spec(spec)
sys.modules["bench_e2e"] = e2e
spec.loader.exec_module(e2e)


def test_action_table_has_90_distinct_valid_rows() -> None:
    t = e2e.action_table()
    assert t.shape == (90, 8)
    assert len({tuple(r) for r in t}) == 90
    assert np.all(np.abs(t[:, :5]) <= 1) and set(np.unique(t[:, 5:])) <= {0.0, 1.0}


def test_quat_axes_identity_and_yaw_90() -> None:
    fwd, up = e2e.quat_axes(np.array([[1.0, 0, 0, 0], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]], dtype=np.float32))
    np.testing.assert_allclose(fwd[0], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(up[0], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(fwd[1], [0, 1, 0], atol=1e-6)  # lacet de +90° autour de z


def _fake_state(n_env: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    car_dt = np.dtype(
        [("pos", "<f4", 3), ("rot", "<f4", 4), ("vel", "<f4", 3), ("ang_vel", "<f4", 3), ("boost", "<f4"),
         ("is_on_ground", "u1"), ("has_flipped", "u1"), ("has_double_jumped", "u1"), ("is_demoed", "u1")]
    )  # fmt: skip
    cars = np.zeros(2 * n_env, dtype=car_dt)
    for f in ("pos", "vel", "ang_vel"):
        cars[f] = rng.normal(size=cars[f].shape)
    cars["rot"][:, 0] = 1.0
    balls = np.zeros(n_env, dtype=np.dtype([("pos", "<f4", 3), ("vel", "<f4", 3), ("ang_vel", "<f4", 3)]))
    balls["pos"] = rng.normal(size=(n_env, 3))
    pads = np.zeros(34 * n_env, dtype=np.dtype([("is_active", "u1")]))
    return cars, balls, pads


def test_build_obs_shape_and_mirror_of_ball() -> None:
    rng = np.random.default_rng(0)
    cars, balls, pads = _fake_state(3, rng)
    obs = e2e.build_obs(cars, balls, pads, np.zeros((6, 8), dtype=np.float32))
    assert obs.shape == (6, e2e.OBS_DIM) and obs.dtype == np.float32 and np.isfinite(obs).all()
    # Balle vue par bleu (agent 0) et orange (agent 1) : x et y opposés, z identique.
    np.testing.assert_allclose(obs[1, :2], -obs[0, :2], rtol=1e-6)
    assert obs[1, 2] == obs[0, 2]


def test_gae_terminal_no_bootstrap_and_truncation() -> None:
    rew = np.array([[0.0], [1.0]], dtype=np.float32)
    val = np.array([[0.5], [0.2]], dtype=np.float32)
    term = np.array([[0.0], [1.0]], dtype=np.float32)
    adv, ret = e2e.gae(rew, val, np.array([9.0], dtype=np.float32), term, np.zeros_like(term), 0.9, 1.0)
    assert adv[1, 0] == pytest.approx(1.0 - 0.2)  # but : pas de bootstrap sur V(s')
    assert adv[0, 0] == pytest.approx(0.9 * 0.2 - 0.5 + 0.9 * adv[1, 0])
    np.testing.assert_allclose(ret, adv + val)
