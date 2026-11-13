"""Parité entraînement / RLBot v5 (ADR-004, artefact du gate G3).

Principe : RocketGPU (CPU) joue des épisodes en boucle fermée ; à chaque tick l'état du simulateur est exporté en
`GamePacket` RLBot synthétique par `export_packet` (écrit ici, à partir de la sémantique documentée du stub
`rlbot_flatbuffers` 0.19.0, sans utiliser l'adaptateur), `PacketAdapter.step` le reconvertit et choisit l'action, dont
les `ControllerState` pilotent le simulateur. À chaque décision (tous les 8 ticks) on compare l'observation de
l'adaptateur à `build_obs(état du simulateur, action précédente)`, pour le bleu et pour l'orange.

Ce que cela prouve : l'adaptateur est cohérent avec NOTRE lecture des conventions RLBot (unités, Euler, ordre des pads,
timers, tick-skip). Ce que cela ne prouve pas : que le jeu réel produit ces champs avec ces conventions (Phase 5, jeu
lancé par RLBot). Sauté si RLBot ou RocketGPU est absent (`ROCKETGPU_PY_DIR`, `ROCKETGPU_COLLISION_MESH_DIR`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("rlbot")
from rlbot import flat

from actions import ACTION_TABLE, N_ACTIONS
from env import RocketGPUSimulator, SimulatorUnavailableError
from env.initial_states import sample_generic_states
from obs import schema as S
from obs.builder import build_obs
from obs.pads import PAD_POSITIONS_ROCKETSIM, ROCKETGPU_TO_ROCKETSIM
from obs.state import N_PADS
from rlbot_bot import PacketAdapter, action_from_controls

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

try:
    _PROBE = RocketGPUSimulator(2, 2, 1)
    _SKIP: str | None = None
except (SimulatorUnavailableError, OSError) as exc:
    _SKIP = f"RocketGPU indisponible : {exc}"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

N_ENVS = 8
TICK_SKIP = 8

# --- Export RocketGPU -> paquet RLBot (sémantique du stub, indépendant de l'adaptateur) -------------------------------


def quat_to_euler(q: F64) -> tuple[float, float, float]:
    """Quaternion (w, x, y, z) -> (pitch, yaw, roll) RLBot : pitch = asin(avant_z), yaw = atan2(avant_y, avant_x),
    roll = atan2(-axe_y_z, haut_z) (stub `Rotator` : pitch + nez en haut, yaw + vers +y, roll + penché à droite)."""
    w, x, y, z = q / np.linalg.norm(q)
    fwd = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    side_z = 2 * (y * z + w * x)  # composante z de l'axe y du corps
    up_z = 1 - 2 * (x * x + y * y)
    return float(np.arcsin(np.clip(fwd[2], -1, 1))), float(np.arctan2(fwd[1], fwd[0])), float(np.arctan2(-side_z, up_z))


def _vec(v: Any) -> flat.Vector3:
    return flat.Vector3(float(v[0]), float(v[1]), float(v[2]))


_PAD_ORDER = np.lexsort((PAD_POSITIONS_ROCKETSIM[:, 0], PAD_POSITIONS_ROCKETSIM[:, 1]))  # y puis x (stub FieldInfo)
"""Ordre du jeu : positions de RocketSim (±2300, comme le jeu) triées par y puis x ; entrée = indice RocketSim."""
_RS_TO_RG = np.argsort(ROCKETGPU_TO_ROCKETSIM)


def export_field_info() -> flat.FieldInfo:
    pads = [
        flat.BoostPad(_vec(PAD_POSITIONS_ROCKETSIM[rs]), is_full_boost=bool(PAD_POSITIONS_ROCKETSIM[rs][2] == 73.0))
        for rs in _PAD_ORDER
    ]
    return flat.FieldInfo(boost_pads=pads)


def export_packet(sim: RocketGPUSimulator, arena: int, frame: int, phase: Any = None) -> flat.GamePacket:
    raw = sim.raw_arrays
    cars = raw["car_states"].reshape(sim.n_envs, sim.n_cars)[arena]
    ball = raw["ball_states"][arena]
    pads = raw["boost_pad_states"].reshape(sim.n_envs, N_PADS)[arena]
    players = []
    for k in range(sim.n_cars):
        c = cars[k]
        pitch, yaw, roll = quat_to_euler(np.asarray(c["rot"], dtype=np.float64))
        phys = flat.Physics(_vec(c["pos"]), flat.Rotator(pitch, yaw, roll), _vec(c["vel"]), _vec(c["ang_vel"]))
        players.append(
            flat.PlayerInfo(
                physics=phys,
                air_state=flat.AirState(int(c["air_state"])),
                dodge_timeout=float(c["dodge_timeout"]),
                demolished_timeout=float(c["demo_respawn_timer"]) if c["is_demoed"] else -1.0,
                team=0 if k < sim.n_cars // 2 else 1,
                boost=float(c["boost"]),
                has_jumped=bool(c["has_jumped"]),
                has_double_jumped=bool(c["has_double_jumped"]),
                has_dodged=bool(c["has_flipped"]),
            )
        )
    pad_states: list[flat.BoostPadState] = []
    for rs in _PAD_ORDER:
        p = pads[_RS_TO_RG[rs]]
        full = bool(p["is_big"])
        timer = 0.0 if p["is_active"] else (10.0 if full else 4.0) - float(p["cooldown"])
        pad_states.append(flat.BoostPadState(is_active=bool(p["is_active"]), timer=timer))
    ball_info = flat.BallInfo(
        physics=flat.Physics(_vec(ball["pos"]), flat.Rotator(), _vec(ball["vel"]), _vec(ball["ang_vel"]))
    )
    info = flat.MatchInfo(frame_num=frame, match_phase=phase if phase is not None else flat.MatchPhase.Active)
    return flat.GamePacket(players=players, boost_pads=pad_states, balls=[ball_info], match_info=info)


# --- Boucle fermée ---------------------------------------------------------------------------------------------------


@dataclass
class ParityStats:
    n_compared: int = 0
    max_abs: dict[str, float] = field(default_factory=dict)
    mismatch_rows: dict[str, int] = field(default_factory=dict)
    on_ground_diff_rows: int = 0
    jump_flag_rows: int = 0
    inactive_pads: int = 0
    flipped_rows: int = 0
    double_jumped_rows: int = 0
    dodge_window_rows: int = 0
    demoed_rows: int = 0
    airborne_rows: int = 0
    action_roundtrip_failures: int = 0
    boost_without_gas: int = 0
    teams_seen: set[int] = field(default_factory=set)


_BLOCKS = {
    "balle": S.BALL,
    "action_prec": S.PREV_ACTION,
    "pads": S.PADS,
    "soi_pos": slice(S.SELF.start + S.CAR_POS.start, S.SELF.start + S.CAR_POS.stop),
    "soi_avant": slice(S.SELF.start + S.CAR_FWD.start, S.SELF.start + S.CAR_FWD.stop),
    "soi_haut": slice(S.SELF.start + S.CAR_UP.start, S.SELF.start + S.CAR_UP.stop),
    "soi_vel": slice(S.SELF.start + S.CAR_VEL.start, S.SELF.start + S.CAR_VEL.stop),
    "soi_angvel_local": slice(S.SELF.start + S.CAR_ANG_VEL_LOCAL.start, S.SELF.start + S.CAR_ANG_VEL_LOCAL.stop),
    "soi_balle_loc": slice(S.SELF.start + S.CAR_BALL_POS_LOCAL.start, S.SELF.start + S.CAR_BALL_VEL_LOCAL.stop),
    "soi_boost": slice(S.SELF.start + S.CAR_BOOST, S.SELF.start + S.CAR_BOOST + 1),
    "soi_on_ground": slice(S.SELF.start + S.CAR_ON_GROUND, S.SELF.start + S.CAR_ON_GROUND + 1),
    "soi_flip": slice(S.SELF.start + S.CAR_HAS_FLIP_OR_JUMP, S.SELF.start + S.CAR_HAS_FLIP_OR_JUMP + 1),
    "soi_demoed": slice(S.SELF.start + S.CAR_IS_DEMOED, S.SELF.start + S.CAR_IS_DEMOED + 1),
    "soi_has_jumped": slice(S.SELF.start + S.CAR_HAS_JUMPED, S.SELF.start + S.CAR_HAS_JUMPED + 1),
    "autre": slice(S.OTHERS_START, S.OBS_DIM),  # inclut `on_ground` de l'autre voiture
}


def run_closed_loop(scenario: str, n_decisions: int, seed: int, euler_noise: float = 0.0) -> ParityStats:
    """`n_decisions` décisions par voiture dans `N_ENVS` arènes ; `euler_noise` : quantification des angles (rad)."""
    rng = np.random.default_rng(seed)
    sim = RocketGPUSimulator(N_ENVS, 2, 1)  # 1 tick par appel : un paquet par tick, comme RLBot
    field_info = export_field_info()
    if scenario == "generic":
        sim.set_states(np.arange(N_ENVS, dtype=np.int64), sample_generic_states(rng, N_ENVS, 2))
        sim.step(np.zeros(N_ENVS * 2, dtype=np.int64))  # un tick d'amorçage : les champs « reportés » sont recalculés
    adapters = [[PacketAdapter(field_info, k, TICK_SKIP, validate=True) for k in range(2)] for _ in range(N_ENVS)]
    prev_train = np.full(N_ENVS * 2, -1, dtype=np.int64)
    held = np.zeros(N_ENVS * 2, dtype=np.int64)
    stats = ParityStats()
    for tick in range(n_decisions * TICK_SKIP):
        decided: dict[int, np.ndarray[Any, Any]] = {}
        picked: dict[int, int] = {}
        ref = build_obs(sim.state(), prev_train) if tick % TICK_SKIP == 0 else None
        for e in range(N_ENVS):
            packet = export_packet(sim, e, tick)
            if euler_noise > 0.0:
                _quantize_euler(packet, euler_noise)
            for k in range(2):
                slot = e * 2 + k

                def policy(
                    obs: F32, slot: int = slot, decided: dict[int, Any] = decided, picked: dict[int, int] = picked
                ) -> int:
                    decided[slot] = obs.copy()
                    picked[slot] = int(rng.integers(0, N_ACTIONS))
                    return picked[slot]

                ctrl = adapters[e][k].step(packet, policy)
                try:
                    held[slot] = action_from_controls(ctrl)
                except ValueError:
                    stats.action_roundtrip_failures += 1
                row = ACTION_TABLE[held[slot]]
                if row[6] == 1 and row[0] != 1:
                    stats.boost_without_gas += 1
        if tick % TICK_SKIP == 0:
            assert ref is not None and set(decided) == set(range(N_ENVS * 2))
            _compare(stats, sim, ref, decided)
            for slot, a in picked.items():
                prev_train[slot] = a
        sim.step(held)
    return stats


def _quantize_euler(packet: flat.GamePacket, step: float) -> None:
    for p in packet.players:
        r = p.physics.rotation
        r.pitch, r.yaw, r.roll = (round(v / step) * step for v in (r.pitch, r.yaw, r.roll))


def _compare(stats: ParityStats, sim: RocketGPUSimulator, ref: F32, decided: dict[int, Any]) -> None:
    got = np.stack([decided[i] for i in range(ref.shape[0])])
    cars = sim.raw_arrays["car_states"]
    diff = np.abs(got - ref)
    # `on_ground` (soi et autre) comparé comme les autres colonnes : `air_state == 0` des deux côtés (ADR-004 §3.1).
    own_bad = diff[:, S.SELF.start + S.CAR_ON_GROUND] > 0
    other_bad = diff[:, S.OTHERS_START + S.CAR_ON_GROUND] > 0
    stats.on_ground_diff_rows += int((own_bad | other_bad).sum())
    for name, sl in _BLOCKS.items():
        d = diff[:, sl]
        stats.max_abs[name] = max(stats.max_abs.get(name, 0.0), float(d.max()))
        stats.mismatch_rows[name] = stats.mismatch_rows.get(name, 0) + int((d.max(axis=1) > 1e-3).sum())
    stats.n_compared += ref.shape[0]
    stats.jump_flag_rows += int(((cars["air_state"] == 1) & (cars["is_on_ground"] != 0)).sum())
    pads = sim.raw_arrays["boost_pad_states"]
    stats.inactive_pads += int((pads["is_active"] == 0).sum())
    stats.flipped_rows += int((cars["has_flipped"] != 0).sum())
    stats.double_jumped_rows += int((cars["has_double_jumped"] != 0).sum())
    stats.dodge_window_rows += int((cars["dodge_timeout"] >= 0).sum())
    stats.demoed_rows += int((cars["is_demoed"] != 0).sum())
    stats.airborne_rows += int((cars["is_on_ground"] == 0).sum())


@pytest.fixture(scope="module", params=["kickoff", "generic"])
def stats(request: pytest.FixtureRequest) -> ParityStats:
    return run_closed_loop(request.param, n_decisions=150, seed=7)


def test_obs_adaptateur_egale_obs_entrainement(stats: ParityStats) -> None:
    assert stats.n_compared > 0
    for name in _BLOCKS:
        assert stats.max_abs[name] <= TOL[name], f"{name}: écart {stats.max_abs[name]:.3g} > {TOL[name]:.3g}"


def test_couverture_non_vide(stats: ParityStats) -> None:
    """Test non vacueux : pads inactifs, vol, double saut, flip et fenêtre de dodge sont couverts."""
    assert stats.inactive_pads > 0 and stats.airborne_rows > 0
    assert stats.flipped_rows > 0 and stats.double_jumped_rows > 0 and stats.dodge_window_rows > 0


def test_aller_retour_actions_et_boost_plein_gaz(stats: ParityStats) -> None:
    assert stats.action_roundtrip_failures == 0
    assert stats.boost_without_gas == 0


def test_on_ground_strict(stats: ParityStats) -> None:
    """`on_ground` est identique des deux côtés, soi et autre voiture : aucun écart toléré (ADR-004 §3.1)."""
    assert stats.on_ground_diff_rows == 0
    assert stats.jump_flag_rows > 0  # non vacueux : des lignes avec `is_on_ground = 1` et AirState Jumping existent


def test_quantification_des_angles_16_bits() -> None:
    """Sensibilité : angles arrondis au pas 2π/65536 (quantification rapportée par une source tierce, non vérifiée)."""
    st = run_closed_loop("generic", n_decisions=40, seed=11, euler_noise=2 * np.pi / 65536)
    for name in ("soi_avant", "soi_haut", "soi_angvel_local", "soi_balle_loc"):
        assert st.max_abs[name] <= QUANT_TOL, f"{name}: {st.max_abs[name]:.3g}"


def test_temoin_negatif_roll_inverse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le test détecte une convention fausse : roll de signe opposé dans l'adaptateur => écart bien au-delà de TOL."""
    import rlbot_bot.adapter as ad

    real = ad.euler_to_quat
    monkeypatch.setattr(ad, "euler_to_quat", lambda p, y, r: real(p, y, -r))
    st = run_closed_loop("generic", n_decisions=20, seed=5)
    assert st.max_abs["soi_haut"] > 100 * TOL["soi_haut"]


TOL: dict[str, float] = {name: 1e-6 for name in _BLOCKS}
"""Blocs copiés tels quels (pos, vel, boost, pads...) : mesuré 0 ; marge 1e-6 pour l'arrondi f32."""
for _name in ("soi_avant", "soi_haut", "soi_angvel_local", "soi_balle_loc", "autre"):
    TOL[_name] = 1e-5
"""Blocs dépendant de l'aller-retour quaternion -> Euler -> quaternion : mesuré <= 7,2e-7 (150 décisions x 8 arènes x 2
voitures, 2 scénarios) ; tolérance 1e-5 = règle `.claude/rules/rlbot.md` (parité < 1e-5), marge ~14x."""
QUANT_TOL = 3e-4
"""Mesuré <= 1,2e-4 avec angles quantifiés à 2π/65536 (erreur d'angle <= 4,8e-5 rad, amplifiée au plus ~2,5x)."""
