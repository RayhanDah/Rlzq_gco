"""Tests de `src/actions` : table (ADR-003 §4) et écriture des contrôles (tampon de 24 octets de RocketGPU)."""

from __future__ import annotations

import numpy as np
import pytest

from actions import (
    ACTION_SCHEMA_VERSION,
    ACTION_TABLE,
    ACTION_TABLE_SHA256,
    CONTROL_FIELDS,
    N_ACTIONS,
    table_sha256,
    write_controls,
)

# Disposition indépendante de rocketgpu.h:116-125 : 5 f32 (0, 4, 8, 12, 16) puis 3 u8 (20, 21, 22), taille 24.
RG_DTYPE = np.dtype(
    {
        "names": list(CONTROL_FIELDS),
        "formats": ["<f4"] * 5 + ["u1"] * 3,
        "offsets": [0, 4, 8, 12, 16, 20, 21, 22],
        "itemsize": 24,
    }
)


def test_table_shape_and_version() -> None:
    assert ACTION_SCHEMA_VERSION == 1 and N_ACTIONS == 90
    assert ACTION_TABLE.shape == (90, 8) and ACTION_TABLE.dtype == np.float32
    assert not ACTION_TABLE.flags.writeable


def test_rows_unique() -> None:
    assert len({tuple(r) for r in ACTION_TABLE.tolist()}) == 90


def test_hash_frozen() -> None:
    assert ACTION_TABLE_SHA256 == "86baa15c48c42c497f3ea0fe62efeb49e4a8241cb3191957822e453cd2d0b655"
    assert table_sha256() == ACTION_TABLE_SHA256


def test_no_boost_without_full_throttle() -> None:
    """Écart de physique A1 : RocketGPU freine au sol si boost sans gaz."""
    t = ACTION_TABLE
    boost, throttle = t[:, 6] == 1, t[:, 0]
    assert boost.any()
    assert (throttle[boost] == 1).all()


def test_domains_and_structure() -> None:
    t = ACTION_TABLE
    assert np.isin(t[:, :5], (-1, 0, 1)).all()
    assert np.isin(t[:, 5:], (0, 1)).all()
    ground, air = t[:24], t[24:]
    assert (ground[:, [2, 4, 5]] == 0).all() and (ground[:, 1] == ground[:, 3]).all()
    assert (air[:, 1] == air[:, 3]).all()
    assert (air[:, 0] == air[:, 6]).all()  # throttle = boost en l'air
    assert (air[:, 5] * np.abs(air[:, 3]) == 0).all()  # jump => yaw = 0
    assert not ((air[:, 2] == 0) & (air[:, 4] == 0) & (air[:, 5] == 0)).any()
    moved = (air[:, 2] != 0) | (air[:, 3] != 0) | (air[:, 4] != 0)
    np.testing.assert_array_equal(air[:, 7], (air[:, 5] == 1) & moved)
    assert ground.shape[0] == 24 and air.shape[0] == 66


def test_write_controls_roundtrip() -> None:
    rng = np.random.default_rng(0)
    cap = 40
    ctrl = np.zeros(cap, dtype=RG_DTYPE)
    ctrl_sentinel = ctrl.copy()
    idx = rng.integers(0, N_ACTIONS, size=32).astype(np.int64)
    write_controls(ctrl, idx)
    raw = ctrl.view(np.uint8).reshape(cap, 24)
    f = raw[:32, :20].copy().view("<f4").reshape(32, 5)
    np.testing.assert_array_equal(f, ACTION_TABLE[idx, :5])
    np.testing.assert_array_equal(raw[:32, 20:23], ACTION_TABLE[idx, 5:].astype(np.uint8))
    assert (raw[:32, 23] == 0).all()  # octet de remplissage intact
    np.testing.assert_array_equal(ctrl[32:], ctrl_sentinel[32:])  # slots au-delà non touchés
    for name, col in zip(CONTROL_FIELDS, ACTION_TABLE[idx].T, strict=True):
        np.testing.assert_array_equal(ctrl[name][:32], col.astype(ctrl[name].dtype))


def test_write_controls_every_action() -> None:
    ctrl = np.zeros(N_ACTIONS, dtype=RG_DTYPE)
    write_controls(ctrl, np.arange(N_ACTIONS, dtype=np.int64))
    for j, name in enumerate(CONTROL_FIELDS):
        np.testing.assert_array_equal(ctrl[name].astype(np.float32), ACTION_TABLE[:, j])


def test_write_controls_rejects_bad_input() -> None:
    ctrl = np.zeros(4, dtype=RG_DTYPE)
    with pytest.raises(ValueError):
        write_controls(ctrl, np.array([0, N_ACTIONS], dtype=np.int64))
    with pytest.raises(ValueError):
        write_controls(ctrl, np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError):
        write_controls(ctrl, np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError):
        write_controls(ctrl, np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError):
        write_controls(np.zeros(4, dtype=np.dtype([("throttle", "<f4")])), np.zeros(1, dtype=np.int64))
