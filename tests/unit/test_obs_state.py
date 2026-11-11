"""Tests de `obs.state.BatchState` (ADR-003 §2) : construction, une violation par règle, vues sans copie."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from obs import BOOST_MAX, N_PADS, QUAT_NORM_TOL, BatchState, default_car_team
from rewards import Team

EXPECTED_FIELDS = [
    "ball_pos",
    "ball_vel",
    "ball_ang_vel",
    "car_pos",
    "car_vel",
    "car_ang_vel",
    "car_quat",
    "car_boost",
    "car_on_ground",
    "car_has_jumped",
    "car_has_double_jumped",
    "car_has_flipped",
    "car_is_demoed",
    "car_air_time_since_jump",
    "pad_active",
    "pad_cooldown",
    "car_team",
]


def _fields(state: BatchState) -> dict[str, Any]:
    return {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}


def _replace(state: BatchState, **changes: Any) -> BatchState:
    return BatchState(**{**_fields(state), **changes})


def _copy_field(state: BatchState, name: str) -> Any:
    return getattr(state, name).copy()


@pytest.mark.parametrize(("n_envs", "n_cars"), [(1, 2), (5, 2), (3, 4)])
def test_construction_valide_1v1_et_2v2(n_envs: int, n_cars: int) -> None:
    state = BatchState.zeros(n_envs, n_cars)  # Arrange + Act
    state.validate()  # Assert : ne lève pas
    assert (state.n_envs, state.n_cars) == (n_envs, n_cars)
    assert state.car_pos.shape == (n_envs, n_cars, 3)
    assert state.car_quat.shape == (n_envs, n_cars, 4)
    assert state.pad_active.shape == (n_envs, N_PADS)


def test_champs_et_dtypes_conformes_adr() -> None:
    state = BatchState.zeros(2, 2)
    assert [f.name for f in dataclasses.fields(state)] == EXPECTED_FIELDS
    for name, arr in _fields(state).items():
        if name == "car_team":
            assert arr.dtype == np.int64
        elif name in {"pad_active", "car_on_ground", "car_has_jumped", "car_has_double_jumped", "car_has_flipped"}:
            assert arr.dtype == np.uint8
        elif name == "car_is_demoed":
            assert arr.dtype == np.uint8
        else:
            assert arr.dtype == np.float32


def test_equipes_par_slot() -> None:
    assert default_car_team(2).tolist() == [Team.BLUE, Team.ORANGE]
    assert default_car_team(4).tolist() == [0, 0, 1, 1]
    for bad in (0, 1, 3):
        with pytest.raises(ValueError, match="pair"):
            default_car_team(bad)


def test_quaternion_unitaire_dans_la_tolerance_accepte() -> None:
    state = BatchState.zeros(1, 2)
    state.car_quat[0, 0] = np.float32([1.0 + 0.5 * QUAT_NORM_TOL, 0, 0, 0])
    state.validate()


def test_validate_ne_modifie_rien() -> None:
    state = BatchState.zeros(2, 2)
    before = {k: v.copy() for k, v in _fields(state).items()}
    state.validate()
    for k, v in _fields(state).items():
        assert np.array_equal(v, before[k])


# --- une violation par règle ---


def test_refuse_dtype_incorrect() -> None:
    state = BatchState.zeros(2, 2)
    bad = _replace(state, ball_pos=state.ball_pos.astype(np.float64))
    with pytest.raises(ValueError, match="ball_pos: dtype"):
        bad.validate()


def test_refuse_dtype_uint8_remplace_par_bool() -> None:
    state = BatchState.zeros(2, 2)
    bad = _replace(state, car_on_ground=state.car_on_ground.astype(np.bool_))
    with pytest.raises(ValueError, match="car_on_ground: dtype"):
        bad.validate()


def test_refuse_shape_incorrecte() -> None:
    state = BatchState.zeros(2, 2)
    bad = _replace(state, car_pos=np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"car_pos: shape"):
        bad.validate()


def test_refuse_incoherence_n_envs() -> None:
    state = BatchState.zeros(3, 2)
    bad = _replace(state, car_boost=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="car_boost: shape"):
        bad.validate()


def test_refuse_incoherence_n_cars() -> None:
    state = BatchState.zeros(2, 2)
    bad = _replace(state, car_vel=np.zeros((2, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="car_vel: shape"):
        bad.validate()


def test_refuse_nombre_de_pads_incorrect() -> None:
    state = BatchState.zeros(2, 2)
    bad = _replace(state, pad_cooldown=np.zeros((2, N_PADS - 1), dtype=np.float32))
    with pytest.raises(ValueError, match="pad_cooldown: shape"):
        bad.validate()


def test_refuse_n_cars_impair() -> None:
    state = BatchState.zeros(2, 2)
    team = np.array([0, 0, 1], dtype=np.int64)
    kw = {
        "car_pos": (3, 3),
        "car_vel": (3, 3),
        "car_ang_vel": (3, 3),
        "car_quat": (3, 4),
        "car_boost": (3,),
        "car_air_time_since_jump": (3,),
    }
    changes: dict[str, Any] = {k: np.zeros((2, *s), dtype=np.float32) for k, s in kw.items()}
    for k in ("car_on_ground", "car_has_jumped", "car_has_double_jumped", "car_has_flipped", "car_is_demoed"):
        changes[k] = np.zeros((2, 3), dtype=np.uint8)
    changes["car_quat"][..., 0] = 1.0
    bad = _replace(state, car_team=team, **changes)
    with pytest.raises(ValueError, match="n_cars doit être pair"):
        bad.validate()


@pytest.mark.parametrize("name", ["ball_pos", "car_vel", "car_air_time_since_jump", "pad_cooldown", "car_quat"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_refuse_valeurs_non_finies(name: str, value: float) -> None:
    state = BatchState.zeros(2, 2)
    arr = _copy_field(state, name)
    arr.reshape(-1)[0] = value
    with pytest.raises(ValueError, match=f"{name}: valeurs non finies"):
        _replace(state, **{name: arr}).validate()


@pytest.mark.parametrize("scale", [1.0 + 2 * QUAT_NORM_TOL, 0.0, 1.0 - 2 * QUAT_NORM_TOL])
def test_refuse_quaternion_non_unitaire(scale: float) -> None:
    state = BatchState.zeros(2, 2)
    quat = _copy_field(state, "car_quat")
    quat[1, 1] = np.float32([scale, 0, 0, 0])
    with pytest.raises(ValueError, match="car_quat: norme"):
        _replace(state, car_quat=quat).validate()


@pytest.mark.parametrize("name", ["car_on_ground", "car_has_jumped", "car_has_double_jumped", "car_has_flipped"])
def test_refuse_booleen_hors_0_1(name: str) -> None:
    state = BatchState.zeros(2, 2)
    arr = _copy_field(state, name)
    arr[0, 0] = 2
    with pytest.raises(ValueError, match=f"{name}: valeurs hors"):
        _replace(state, **{name: arr}).validate()


def test_refuse_is_demoed_et_pad_active_hors_0_1() -> None:
    state = BatchState.zeros(2, 2)
    demoed = _copy_field(state, "car_is_demoed")
    demoed[0, 0] = 255
    with pytest.raises(ValueError, match="car_is_demoed: valeurs hors"):
        _replace(state, car_is_demoed=demoed).validate()
    pads = _copy_field(state, "pad_active")
    pads[1, 3] = 7
    with pytest.raises(ValueError, match="pad_active: valeurs hors"):
        _replace(state, pad_active=pads).validate()


@pytest.mark.parametrize("boost", [-0.5, BOOST_MAX + 0.5])
def test_refuse_boost_hors_plage(boost: float) -> None:
    state = BatchState.zeros(2, 2)
    arr = _copy_field(state, "car_boost")
    arr[0, 1] = boost
    with pytest.raises(ValueError, match="car_boost: hors"):
        _replace(state, car_boost=arr).validate()


def test_refuse_air_time_et_cooldown_negatifs() -> None:
    state = BatchState.zeros(2, 2)
    air = _copy_field(state, "car_air_time_since_jump")
    air[0, 0] = -0.1
    with pytest.raises(ValueError, match="car_air_time_since_jump: valeur négative"):
        _replace(state, car_air_time_since_jump=air).validate()
    cd = _copy_field(state, "pad_cooldown")
    cd[0, 0] = -1.0
    with pytest.raises(ValueError, match="pad_cooldown: valeur négative"):
        _replace(state, pad_cooldown=cd).validate()


@pytest.mark.parametrize("team", [[1, 0], [0, 0], [1, 1], [0, 2]])
def test_refuse_equipes_incorrectes_1v1(team: list[int]) -> None:
    state = BatchState.zeros(2, 2)
    with pytest.raises(ValueError, match="car_team"):
        _replace(state, car_team=np.array(team, dtype=np.int64)).validate()


def test_refuse_equipes_incorrectes_2v2() -> None:
    state = BatchState.zeros(2, 4)
    with pytest.raises(ValueError, match="car_team"):
        _replace(state, car_team=np.array([0, 1, 0, 1], dtype=np.int64)).validate()


def test_refuse_champ_non_ndarray() -> None:
    state = BatchState.zeros(2, 2)
    with pytest.raises(ValueError, match="ndarray attendu"):
        _replace(state, ball_vel=[[0.0, 0.0, 0.0]] * 2).validate()


def test_refuse_n_envs_nul() -> None:
    state = BatchState.zeros(1, 2)
    bad = BatchState.zeros(0, 2)
    with pytest.raises(ValueError, match="n_envs doit être >= 1"):
        bad.validate()
    state.validate()


def test_plusieurs_violations_listees() -> None:
    state = BatchState.zeros(2, 2)
    boost = _copy_field(state, "car_boost")
    boost[0, 0] = 200.0
    cd = _copy_field(state, "pad_cooldown")
    cd[0, 0] = -1.0
    with pytest.raises(ValueError) as exc:
        _replace(state, car_boost=boost, pad_cooldown=cd).validate()
    assert "car_boost" in str(exc.value)
    assert "pad_cooldown" in str(exc.value)


# --- vues sans copie ---


def test_vues_strided_sans_copie_conservees() -> None:
    # Arrange : mémoire « sim » structurée (E, C), champs lus comme vues strided.
    n_envs, n_cars = 3, 2
    sim_dtype = np.dtype([("pos", np.float32, 3), ("quat", np.float32, 4), ("boost", np.float32), ("ground", np.uint8)])
    sim = np.zeros((n_envs, n_cars), dtype=sim_dtype)
    sim["quat"][..., 0] = 1.0
    base = BatchState.zeros(n_envs, n_cars)
    # Act
    state = _replace(
        base,
        car_pos=sim["pos"],
        car_quat=sim["quat"],
        car_boost=sim["boost"],
        car_on_ground=sim["ground"],
    )
    state.validate()
    # Assert : mêmes tampons, non contigus, et les écritures du sim sont visibles.
    assert not state.car_pos.flags["C_CONTIGUOUS"]
    assert np.shares_memory(state.car_pos, sim)
    assert np.shares_memory(state.car_boost, sim)
    sim["pos"][2, 1] = (1.0, 2.0, 3.0)
    sim["boost"][0, 0] = 55.0
    assert state.car_pos[2, 1].tolist() == [1.0, 2.0, 3.0]
    assert state.car_boost[0, 0] == 55.0
    sim["boost"][0, 0] = 500.0  # le sim corrompt l'état : validate() le voit sur la même mémoire
    with pytest.raises(ValueError, match="car_boost: hors"):
        state.validate()


def test_construction_ne_copie_pas() -> None:
    state = BatchState.zeros(2, 2)
    ball = np.zeros((2, 3), dtype=np.float32)
    rebuilt = _replace(state, ball_pos=ball)
    assert rebuilt.ball_pos is ball
    assert rebuilt.car_pos is state.car_pos


def test_batchstate_gele() -> None:
    state = BatchState.zeros(1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.ball_pos = np.zeros((1, 3), dtype=np.float32)  # type: ignore[misc]
