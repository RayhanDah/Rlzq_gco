"""Tests de `src/obs` : builder, miroir (spec §7.5), pads, fuite. Propriétés à seed fixe, sans RocketGPU ni réseau."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from actions import ACTION_TABLE, N_ACTIONS
from obs import (
    OBS_DIM,
    PAD_MIRROR_PERM,
    PAD_POSITIONS_ROCKETGPU,
    PAD_POSITIONS_ROCKETSIM,
    ROCKETGPU_TO_ROCKETSIM,
    ROCKETSIM_TO_ROCKETGPU,
    BatchState,
    build_obs,
    obs_dim,
)
from obs import schema as S

M = np.array([-1.0, -1.0, 1.0], dtype=np.float32)


def random_state(rng: np.random.Generator, n_envs: int, n_cars: int) -> BatchState:
    e, c = n_envs, n_cars

    def vec(shape: tuple[int, ...], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return rng.uniform(lo, hi, size=(*shape, 3)).astype(np.float32)

    pos_lo = np.array([-4000, -5000, 17], dtype=np.float32)
    pos_hi = np.array([4000, 5000, 1900], dtype=np.float32)
    q = rng.normal(size=(e, c, 4))
    q = (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)
    state = BatchState(
        ball_pos=vec((e,), pos_lo, pos_hi),
        ball_vel=vec((e,), -6000 * np.ones(3), 6000 * np.ones(3)),
        ball_ang_vel=vec((e,), -6 * np.ones(3), 6 * np.ones(3)),
        car_pos=vec((e, c), pos_lo, pos_hi),
        car_vel=vec((e, c), -2300 * np.ones(3), 2300 * np.ones(3)),
        car_ang_vel=vec((e, c), -5.5 * np.ones(3), 5.5 * np.ones(3)),
        car_quat=q,
        car_boost=rng.uniform(0, 100, size=(e, c)).astype(np.float32),
        car_on_ground=rng.integers(0, 2, size=(e, c), dtype=np.uint8),
        car_has_jumped=rng.integers(0, 2, size=(e, c), dtype=np.uint8),
        car_has_double_jumped=rng.integers(0, 2, size=(e, c), dtype=np.uint8),
        car_has_flipped=rng.integers(0, 2, size=(e, c), dtype=np.uint8),
        car_is_demoed=rng.integers(0, 2, size=(e, c), dtype=np.uint8),
        car_air_time_since_jump=rng.uniform(0, 3, size=(e, c)).astype(np.float32),
        pad_active=rng.integers(0, 2, size=(e, 34), dtype=np.uint8),
        pad_cooldown=rng.uniform(0, 10, size=(e, 34)).astype(np.float32),
        car_team=BatchState.zeros(1, c).car_team,
    )
    state.validate()
    return state


def mirror_state(s: BatchState) -> BatchState:
    """Rotation de 180° autour de z + équipes échangées (slots décalés de C/2). Écrit indépendamment du builder :
    q' = q_z(pi) ⊗ q = (-z, -y, x, w)."""
    c = s.n_cars
    sh = c // 2

    def swap(a: np.ndarray) -> np.ndarray:
        return np.roll(a, sh, axis=1)

    q = s.car_quat
    q2 = np.stack([-q[..., 3], -q[..., 2], q[..., 1], q[..., 0]], axis=-1)
    return dataclasses.replace(
        s,
        ball_pos=s.ball_pos * M,
        ball_vel=s.ball_vel * M,
        ball_ang_vel=s.ball_ang_vel * M,
        car_pos=swap(s.car_pos * M),
        car_vel=swap(s.car_vel * M),
        car_ang_vel=swap(s.car_ang_vel * M),
        car_quat=swap(q2),
        car_boost=swap(s.car_boost),
        car_on_ground=swap(s.car_on_ground),
        car_has_jumped=swap(s.car_has_jumped),
        car_has_double_jumped=swap(s.car_has_double_jumped),
        car_has_flipped=swap(s.car_has_flipped),
        car_is_demoed=swap(s.car_is_demoed),
        car_air_time_since_jump=swap(s.car_air_time_since_jump),
        pad_active=s.pad_active[:, PAD_MIRROR_PERM],
        pad_cooldown=s.pad_cooldown[:, PAD_MIRROR_PERM],
    )


def prev_of(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(-1, N_ACTIONS, size=n).astype(np.int64)


def test_dimensions_and_dtype() -> None:
    assert OBS_DIM == 109 == obs_dim(2) == 51 + 29 * 2
    assert obs_dim(4) == 167
    assert S.BALL == slice(0, 9) and S.PREV_ACTION == slice(9, 17) and S.PADS == slice(17, 51)
    assert S.SELF == slice(51, 80) and S.others_slice(2) == slice(80, 109)
    rng = np.random.default_rng(0)
    obs = build_obs(random_state(rng, 5, 2), prev_of(rng, 10))
    assert obs.shape == (10, 109) and obs.dtype == np.float32


@pytest.mark.parametrize("n_cars", [2, 4])
def test_mirror_orange_equals_blue_of_mirrored_state(n_cars: int) -> None:
    rng = np.random.default_rng(1)
    e = 6
    for _ in range(5):
        s = random_state(rng, e, n_cars)
        prev = prev_of(rng, e * n_cars)
        m = mirror_state(s)
        prev_m = np.roll(prev.reshape(e, n_cars), n_cars // 2, axis=1).reshape(-1)
        o = build_obs(s, prev).reshape(e, n_cars, -1)
        om = build_obs(m, prev_m).reshape(e, n_cars, -1)
        # chaque agent de s égale l'agent de l'état miroité (équipe échangée)
        np.testing.assert_allclose(o, np.roll(om, n_cars // 2, axis=1), rtol=0, atol=2e-6)
        # et en particulier orange(s) == bleu(m(s))
        np.testing.assert_allclose(o[:, n_cars - 1], om[:, n_cars // 2 - 1], rtol=0, atol=2e-6)


def test_mirror_is_nontrivial() -> None:
    """Garde-fou : sans miroir, l'obs orange différerait ; ici elle diffère de celle d'un bleu non miroité."""
    rng = np.random.default_rng(2)
    s = random_state(rng, 4, 2)
    swapped = dataclasses.replace(s, car_team=s.car_team[::-1].copy())
    a = build_obs(s, np.full(8, -1, dtype=np.int64))
    b = build_obs(swapped, np.full(8, -1, dtype=np.int64))
    assert not np.allclose(a, b)


def test_finite_and_bounded() -> None:
    rng = np.random.default_rng(3)
    s = random_state(rng, 64, 2)
    obs = build_obs(s, prev_of(rng, 128))
    assert np.isfinite(obs).all()
    assert np.abs(obs).max() <= 7.0  # borne analytique : vitesse relative locale <= (6000+2300)·sqrt(3)/2300 = 6,25
    assert np.abs(obs[:, S.PADS]).max() <= 1.0 and obs[:, S.PADS].min() > 0.0
    car = obs[:, S.SELF]
    assert np.abs(car[:, S.CAR_FWD]).max() <= 1.0 + 1e-6
    assert np.abs(car[:, S.CAR_UP]).max() <= 1.0 + 1e-6
    assert np.abs(car[:, S.CAR_POS]).max() <= 1.0 + 1e-6
    assert car[:, S.CAR_BOOST].min() >= 0 and car[:, S.CAR_BOOST].max() <= 1.0


def test_reference_values_identity_blue() -> None:
    """État au repos : valeurs attendues calculées à la main."""
    s = BatchState.zeros(1, 2)
    s.ball_pos[0] = (4096.0, 0.0, 2044.0)
    s.car_pos[0, 0] = (0.0, -5120.0, 0.0)
    s.car_boost[0, 0] = 50.0
    s.car_on_ground[0, 0] = 1
    obs = build_obs(s, np.array([-1, 5], dtype=np.int64))
    np.testing.assert_allclose(obs[0, 0:3], (1.0, 0.0, 1.0))
    np.testing.assert_allclose(obs[0, S.PREV_ACTION], 0.0)
    np.testing.assert_allclose(obs[1, S.PREV_ACTION], ACTION_TABLE[5])
    np.testing.assert_allclose(obs[0, S.SELF][S.CAR_POS], (0.0, -1.0, 0.0))
    np.testing.assert_allclose(obs[0, S.SELF][S.CAR_FWD], (1.0, 0.0, 0.0))
    np.testing.assert_allclose(obs[0, S.SELF][S.CAR_UP], (0.0, 0.0, 1.0))
    assert obs[0, S.SELF][S.CAR_BOOST] == pytest.approx(0.5)
    assert obs[0, S.SELF][S.CAR_ON_GROUND] == 1.0 and obs[0, S.SELF][S.CAR_HAS_FLIP_OR_JUMP] == 1.0
    # balle - voiture = (4096, 5120, 2044), repère identité, échelle scalaire 5120
    np.testing.assert_allclose(obs[0, S.SELF][S.CAR_BALL_POS_LOCAL], (0.8, 1.0, 2044 / 5120), rtol=1e-6)
    # pads actifs : 1 ; cooldown 3 s inactif -> 1/4
    s.pad_active[0, 7] = 0
    s.pad_cooldown[0, 7] = 3.0
    obs = build_obs(s, np.array([-1, -1], dtype=np.int64))
    assert obs[0, S.PADS][7] == pytest.approx(0.25) and obs[0, S.PADS][8] == 1.0
    # orange : pad 7 apparaît à PAD_MIRROR_PERM[7]
    assert obs[1, S.PADS][PAD_MIRROR_PERM[7]] == pytest.approx(0.25)


def test_has_flip_or_jump_window() -> None:
    s = BatchState.zeros(1, 2)
    s.car_air_time_since_jump[0] = (1.0, 1.3)
    obs = build_obs(s, np.array([-1, -1], dtype=np.int64))
    assert obs[0, S.SELF][S.CAR_HAS_FLIP_OR_JUMP] == 1.0 and obs[1, S.SELF][S.CAR_HAS_FLIP_OR_JUMP] == 0.0
    s.car_has_flipped[0, 0] = 1
    assert build_obs(s, np.array([-1, -1], dtype=np.int64))[0, S.SELF][S.CAR_HAS_FLIP_OR_JUMP] == 0.0


def test_negative_cooldown_tolerated() -> None:
    s = BatchState.zeros(1, 2)
    s.pad_cooldown[0, 3] = -1.0 / 120.0
    obs = build_obs(s, np.array([-1, -1], dtype=np.int64))
    assert obs[0, S.PADS][3] == 1.0  # pad_active = 0 et cooldown clampé à 0 -> 1/(1+0)


def test_out_buffer_and_no_aliasing() -> None:
    rng = np.random.default_rng(4)
    s = random_state(rng, 8, 2)
    prev = prev_of(rng, 16)
    buf = np.full((16, OBS_DIM), np.nan, dtype=np.float32)
    ret = build_obs(s, prev, out=buf)
    assert ret is buf and np.isfinite(buf).all()
    np.testing.assert_array_equal(buf, build_obs(s, prev))
    fresh = build_obs(s, prev)
    assert not np.shares_memory(fresh[0], fresh[1])
    assert not np.shares_memory(fresh, s.ball_pos)
    before = fresh.copy()
    fresh[0] = 123.0
    np.testing.assert_array_equal(fresh[1:], before[1:])
    with pytest.raises(ValueError):
        build_obs(s, prev, out=np.zeros((16, OBS_DIM), dtype=np.float64))
    with pytest.raises(ValueError):
        build_obs(s, prev, out=np.zeros((16, OBS_DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        build_obs(s, prev[:-1])


def test_no_leak_from_other_arena() -> None:
    rng = np.random.default_rng(5)
    s = random_state(rng, 4, 2)
    prev = prev_of(rng, 8)
    base = build_obs(s, prev)
    s2 = dataclasses.replace(
        s,
        ball_pos=s.ball_pos.copy(),
        car_pos=s.car_pos.copy(),
        pad_active=s.pad_active.copy(),
        car_boost=s.car_boost.copy(),
    )
    s2.ball_pos[2] += 100.0
    s2.car_pos[2] += 50.0
    s2.pad_active[2] ^= 1
    s2.car_boost[2] = 0.0
    other = build_obs(s2, prev)
    keep = [0, 1, 2, 3, 6, 7]  # agents des arènes 0, 1, 3
    np.testing.assert_array_equal(base[keep], other[keep])
    assert not np.array_equal(base[4:6], other[4:6])


def test_no_excluded_field_in_state() -> None:
    """Temps restant, horloges de troncature et score ne sont pas dans `BatchState` : l'obs n'y a pas accès."""
    names = {f.name for f in dataclasses.fields(BatchState)}
    assert not any(k in n for n in names for k in ("remaining", "tick", "score", "episode", "touch"))


def test_prev_action_index_validation() -> None:
    s = BatchState.zeros(1, 2)
    with pytest.raises(ValueError):
        build_obs(s, np.array([0, N_ACTIONS], dtype=np.int64))


# --- pads -------------------------------------------------------------------------------------------------------------


def test_pad_permutations_are_bijections() -> None:
    for p in (PAD_MIRROR_PERM, ROCKETGPU_TO_ROCKETSIM, ROCKETSIM_TO_ROCKETGPU):
        assert sorted(p.tolist()) == list(range(34))
    np.testing.assert_array_equal(ROCKETSIM_TO_ROCKETGPU[ROCKETGPU_TO_ROCKETSIM], np.arange(34))
    np.testing.assert_array_equal(ROCKETGPU_TO_ROCKETSIM[ROCKETSIM_TO_ROCKETGPU], np.arange(34))


def test_mirror_perm_is_involution_and_positions_consistent() -> None:
    np.testing.assert_array_equal(PAD_MIRROR_PERM[PAD_MIRROR_PERM], np.arange(34))
    loc = PAD_POSITIONS_ROCKETGPU
    np.testing.assert_allclose(loc[PAD_MIRROR_PERM], loc * M, atol=1e-3)
    assert not np.any(PAD_MIRROR_PERM == np.arange(34))  # aucun pad n'est son propre miroir


def test_known_pad_indices() -> None:
    # docs/reports/physics_parity.md A3 : gros pad (3072, 4096) = indice 31 (RocketGPU), 30 (RocketSim).
    assert tuple(PAD_POSITIONS_ROCKETGPU[31, :2]) == (3072.0, 4096.0)
    assert tuple(PAD_POSITIONS_ROCKETSIM[30, :2]) == (3072.0, 4096.0)
    assert ROCKETGPU_TO_ROCKETSIM[31] == 30
    big_rs = {3, 4, 15, 18, 29, 30}
    assert {int(ROCKETGPU_TO_ROCKETSIM[i]) for i in range(28, 34)} == big_rs


def test_rocketsim_rocketgpu_positions_agree() -> None:
    a = PAD_POSITIONS_ROCKETGPU[ROCKETSIM_TO_ROCKETGPU]  # positions RG, indexées en ordre RS
    assert np.abs(a - PAD_POSITIONS_ROCKETSIM).max() <= 2.0 + 1e-6  # (±1788, ±2302) vs (±1788, ±2300)
    assert (PAD_POSITIONS_ROCKETGPU[:, 2] == np.where(np.arange(34) >= 28, 73.0, 70.0)).all()
