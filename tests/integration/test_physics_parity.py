"""Parité physique RocketGPU @f0508a1 (candidat) vs RocketSim 2.2.1 (référence), épisodes 1v1 complets.

Mêmes états initiaux (écrits explicitement dans les deux simulateurs) et mêmes contrôles (seed fixe), tick-skip 8.
Conventions vérifiées par introspection (voir docs/reports/physics_parity.md) :
- RocketGPU : quaternion (w, x, y, z), `rot` = rotation dont les colonnes sont (forward, right, up) au sens de
  RocketSim (`RotMat.forward/right/up`) ; voiture 0 = bleue, voiture 1 = orange ; boost en 0..100 dans les deux.
- Environnement : `ROCKETGPU_PY_DIR` (dossier du module `rocketgpu_py`), `ROCKETGPU_DLL_DIR` (DLL MinGW, défaut
  C:\\ProgramData\\mingw64\\mingw64\\bin), `ROCKETGPU_COLLISION_MESH_DIR` (dossier contenant `soccar/`).
  Sans eux, le test est sauté.

Deux classes de scénarios, deux régimes de test :
- DÉTERMINISTES (sans contact ni boost-pad ambigu, ou contact simple) : seuils serrés, fixés APRÈS mesure (voir `TOL_*`
  et leur justification ci-dessous) ;
- LONGS / CHAOTIQUES (contrôles aléatoires, contacts multiples) : la divergence est mesurée et rapportée, jamais
  exigée nulle. Seule une garde grossière détecte une casse (état non fini, voiture hors de l'arène, écart de
  plusieurs ordres de grandeur au-dessus de la mesure).

Rapport : `python tests/integration/test_physics_parity.py --report` imprime le tableau Markdown des écarts.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261111
TICK_SKIP = 8
HORIZONS = (16, 64, 120, 240, 600, 2400)  # ticks (multiples de TICK_SKIP) ; 120 ticks = 1 s
DLL_DIR = os.environ.get("ROCKETGPU_DLL_DIR", r"C:\ProgramData\mingw64\mingw64\bin")

# Ordre des champs de contrôle : throttle, steer, pitch, yaw, roll, jump, boost, handbrake
THROTTLE, STEER, PITCH, YAW, ROLL, JUMP, BOOST, HANDBRAKE = range(8)
CAR_Z = 17.0  # hauteur de repos de l'Octane (M1c : 17,01 dans les deux simulateurs)
# RocketSim 2.2.1 laisse flotter une balle posée en l'air avec une vitesse EXACTEMENT nulle (elle reste endormie) ;
# RocketGPU la fait tomber (mesuré, voir le rapport). Les scénarios « chute libre » lancent donc la balle à 1 mm/s,
# vitesse qui réveille la balle des deux côtés. Le cas vitesse nulle est couvert par un test de caractérisation.
MIDAIR_VZ = 0.001
BALL_Z = 93.15  # hauteur de repos de la balle au kickoff RocketSim (RocketGPU : 92,75, voir le rapport)


# --------------------------------------------------------------------------------------------- chargement


def _mesh_dir() -> Path | None:
    for cand in (os.environ.get("ROCKETGPU_COLLISION_MESH_DIR"), str(ROOT / "vendor" / "collision_meshes")):
        if cand and (Path(cand) / "soccar").is_dir():
            return Path(cand)
    return None


def _load() -> tuple[Any, Any, Path]:
    """Retourne (rocketgpu_py, RocketSim, dossier des meshes) ou lève ImportError/RuntimeError."""
    mesh = _mesh_dir()
    if mesh is None:
        raise RuntimeError("meshes soccar absents (ROCKETGPU_COLLISION_MESH_DIR)")
    py_dir = os.environ.get("ROCKETGPU_PY_DIR", "")
    if not py_dir or not Path(py_dir).is_dir():
        raise RuntimeError("ROCKETGPU_PY_DIR absent ou invalide")
    os.environ["ROCKETGPU_COLLISION_MESH_DIR"] = str(mesh)
    if os.name == "nt" and Path(DLL_DIR).is_dir():
        os.add_dll_directory(DLL_DIR)
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    import rocketgpu_py as rg
    import RocketSim as rs

    if not getattr(rs, "_parity_inited", False):
        rs.init(str(mesh))  # lève « Already inited » au second appel dans un même processus
        rs._parity_inited = True
    return rg, rs, mesh


# --------------------------------------------------------------------------------------------- géométrie


def rot_from_ypr(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Matrice 3x3 dont les colonnes sont (forward, right, up) au sens RocketSim ; yaw autour de +z."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) d'une rotation propre (algorithme de Shepperd)."""
    t = np.trace(r)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2 * np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2 * np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def angle_between(ra: np.ndarray, rb: np.ndarray) -> float:
    """Angle (rad) de la rotation relative ra^T rb."""
    c = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


# --------------------------------------------------------------------------------------------- scénarios


@dataclass
class CarInit:
    pos: tuple[float, float, float]
    yaw: float = 0.0
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    boost: float = 33.3
    rot: np.ndarray | None = None  # remplace `yaw` si fourni


@dataclass
class BallInit:
    pos: tuple[float, float, float] = (0.0, 0.0, BALL_Z)
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Scenario:
    name: str
    cars: tuple[CarInit, CarInit]
    ball: BallInit
    controls: np.ndarray  # [n_steps, 2, 8]
    kind: str  # "deterministic" | "contact" | "chaotic"
    doc: str = ""
    tags: set[str] = field(default_factory=set)

    @property
    def n_steps(self) -> int:
        return self.controls.shape[0]


FAR_IDLE = CarInit(pos=(3000.0, 4000.0, CAR_Z), yaw=np.pi, boost=0.0)  # voiture inerte hors du chemin
BLUE_KICKOFF = dict(pos=(-2048.0, -2560.0, CAR_Z), yaw=np.pi / 4)  # spot diagonal de kickoff, face à +x +y
ORANGE_KICKOFF = dict(pos=(2048.0, 2560.0, CAR_Z), yaw=np.pi / 4 + np.pi)


def const_controls(n_steps: int, **kw: float) -> np.ndarray:
    c = np.zeros((n_steps, 2, 8))
    names = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
    for k, v in kw.items():
        c[:, 0, names.index(k)] = v
    return c


def random_controls(rng: np.random.Generator, n_steps: int, biased: bool) -> np.ndarray:
    """Contrôles tenus TICK_SKIP ticks. `biased=False` : distribution de scripts/benchmark_sim.py (axes U(-1,1), boutons
    à 0,5). `biased=True` : conduite plausible (gaz, boost fréquent, sauts rares)."""
    c = np.zeros((n_steps, 2, 8))
    c[..., :5] = rng.uniform(-1.0, 1.0, (n_steps, 2, 5))
    if biased:
        c[..., THROTTLE] = np.where(rng.random((n_steps, 2)) < 0.85, 1.0, c[..., THROTTLE])
        c[..., JUMP] = rng.random((n_steps, 2)) < 0.08
        c[..., BOOST] = rng.random((n_steps, 2)) < 0.5
        c[..., HANDBRAKE] = rng.random((n_steps, 2)) < 0.1
    else:
        c[..., 5:] = rng.random((n_steps, 2, 3)) < 0.5
    return c


def build_scenarios() -> dict[str, Scenario]:
    S: dict[str, Scenario] = {}

    def add(s: Scenario) -> None:
        S[s.name] = s

    # --- Immobile : référence de bruit (zéro contrôle, 5 s). Doit être quasi nul.
    add(
        Scenario(
            "idle_kickoff",
            (CarInit(**BLUE_KICKOFF), CarInit(**ORANGE_KICKOFF)),  # type: ignore[arg-type]
            BallInit(),
            const_controls(75),
            "deterministic",
            "Kickoff diagonal, aucun contrôle : tassement de la suspension et de la balle.",
        )
    )
    # --- Kickoff + contrôles constants (voiture 0 seule, voiture 1 inerte à son spot ; contact balle après ~2,5 s)
    for name, kw, doc in (
        ("kickoff_throttle", dict(throttle=1.0), "plein gaz"),
        ("kickoff_throttle_steer", dict(throttle=1.0, steer=1.0), "plein gaz, braquage max à droite"),
        ("kickoff_throttle_boost", dict(throttle=1.0, boost=1.0), "plein gaz + boost"),
        ("kickoff_reverse", dict(throttle=-1.0), "marche arrière pleine"),
        (
            "kickoff_boost_steer_drift",
            dict(throttle=1.0, boost=1.0, steer=0.5, handbrake=1.0),
            "boost + braquage + frein à main",
        ),
    ):
        add(
            Scenario(
                name,
                (CarInit(**BLUE_KICKOFF), CarInit(**ORANGE_KICKOFF)),  # type: ignore[arg-type]
                BallInit(),
                const_controls(75, **kw),
                "deterministic",
                f"Kickoff diagonal, voiture 0 : {doc} ; voiture 1 inerte. Contact balle possible après ~300 ticks.",
            )
        )
    # --- Sauts (voiture à l'arrêt au centre, balle loin)
    ctr = dict(pos=(-1500.0, -3000.0, CAR_Z), yaw=np.pi / 2)

    def jump_scn(name: str, plan: dict[int, dict[str, float]], doc: str, vel_y: float = 0.0, steps: int = 75) -> None:
        c = np.zeros((steps, 2, 8))
        for step, kw in plan.items():
            for k, v in kw.items():
                c[step, 0, ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake").index(k)] = v
        if vel_y:
            c[:, 0, THROTTLE] = 1.0
        add(
            Scenario(
                name,
                (CarInit(vel=(0.0, vel_y, 0.0), boost=0.0, **ctr), FAR_IDLE),  # type: ignore[arg-type]
                BallInit(pos=(3500.0, 3000.0, 500.0), vel=(0.0, 0.0, -MIDAIR_VZ)),
                c,
                "deterministic",
                doc,
                {"jump"},
            )
        )

    jump_scn("jump_tap", {0: dict(jump=1.0)}, "saut simple, bouton tenu 1 step (8 ticks)")
    jump_scn("jump_hold", {s: dict(jump=1.0) for s in range(4)}, "saut tenu 4 steps (> jump_max_time 0,2 s)")
    jump_scn(
        "double_jump",
        {0: dict(jump=1.0), 6: dict(jump=1.0)},
        "saut puis second saut sans direction (double saut)",
    )
    jump_scn(
        "flip_forward",
        {0: dict(jump=1.0), 2: dict(jump=1.0, pitch=-1.0)},
        "saut puis flip avant (pitch -1)",
    )
    jump_scn(
        "flip_diag",
        {0: dict(jump=1.0), 2: dict(jump=1.0, pitch=-1.0, yaw=1.0)},
        "saut puis flip diagonal (pitch -1, yaw 1)",
    )
    jump_scn(
        "flip_forward_moving",
        {0: dict(jump=1.0), 2: dict(jump=1.0, pitch=-1.0)},
        "flip avant à 1000 uu/s environ (voiture lancée, gaz tenu)",
        vel_y=1000.0,
    )
    # --- Balle : chute libre, rebonds sol et murs, sans voiture proche
    add(
        Scenario(
            "ball_drop",
            (CarInit(**BLUE_KICKOFF), FAR_IDLE),  # type: ignore[arg-type]
            BallInit(pos=(1000.0, -500.0, 1500.0), vel=(0.0, 0.0, -MIDAIR_VZ)),
            const_controls(150),
            "deterministic",
            "Chute libre de 1500 uu sans vitesse horizontale : rebonds au sol (restitution 0,6).",
            {"ball"},
        )
    )
    add(
        Scenario(
            "ball_drop_spin",
            (CarInit(**BLUE_KICKOFF), FAR_IDLE),  # type: ignore[arg-type]
            BallInit(pos=(500.0, -500.0, 800.0), vel=(300.0, 200.0, 0.0), ang_vel=(3.0, -2.0, 1.0)),
            const_controls(150),
            "deterministic",
            "Chute avec vitesse horizontale et rotation : friction du rebond.",
            {"ball"},
        )
    )
    add(
        Scenario(
            "ball_wall_ceiling",
            (CarInit(**BLUE_KICKOFF), FAR_IDLE),  # type: ignore[arg-type]
            BallInit(pos=(2500.0, -1000.0, 500.0), vel=(2000.0, 300.0, 1500.0)),
            const_controls(150),
            "deterministic",
            "Balle lancée vers le mur latéral x=4096 puis le plafond z=2044 : rebonds mur/plafond/sol.",
            {"ball"},
        )
    )

    # --- Frappe de balle par une voiture lancée (la balle est au repos au centre)
    def hit_scn(name: str, x_off: float, speed: float, boost: float, doc: str) -> None:
        add(
            Scenario(
                name,
                (
                    CarInit(pos=(x_off, -2500.0, CAR_Z), yaw=np.pi / 2, vel=(0.0, speed, 0.0), boost=boost),
                    FAR_IDLE,
                ),
                BallInit(),
                const_controls(150, throttle=1.0, boost=1.0 if boost > 0 else 0.0),
                "contact",
                doc,
                {"hit"},
            )
        )

    hit_scn("hit_center", 0.0, 1400.0, 0.0, "frappe frontale centrée à 1400 uu/s, gaz plein")
    hit_scn("hit_offset", 60.0, 1400.0, 0.0, "frappe décalée de 60 uu en x à 1400 uu/s")
    hit_scn("hit_fast_offset", -100.0, 1900.0, 100.0, "frappe décalée de -100 uu à 1900 uu/s avec boost")
    # --- Contrôles aléatoires, kickoff diagonal, les deux voitures pilotées. Variantes « _tb » : mêmes tirages mais gaz
    # forcé à 1 quand le boost est pressé, pour écarter l'écart « boost sans gaz » mesuré à part (boost_override).
    rng = np.random.default_rng(SEED)
    for horizon_s, steps in ((5, 75), (20, 300)):
        for biased in (False, True):
            for i in range(RANDOM_SEEDS):
                base = random_controls(rng, steps, biased)
                tb = base.copy()
                tb[..., THROTTLE] = np.where(tb[..., BOOST] > 0, 1.0, tb[..., THROTTLE])
                for suffix, ctl in (("", base), ("_tb", tb)):
                    tag = ("biased" if biased else "uniform") + suffix
                    add(
                        Scenario(
                            f"random_{tag}_{horizon_s}s_{i:02d}",
                            (CarInit(**BLUE_KICKOFF), CarInit(**ORANGE_KICKOFF)),  # type: ignore[arg-type]
                            BallInit(),
                            ctl,
                            "chaotic",
                            f"Contrôles aléatoires ({tag}), {horizon_s} s, graine {SEED}, tirage n°{i}.",
                            {"random", tag, f"{horizon_s}s"},
                        )
                    )
    # --- Boost au sol sans gaz (écart mesuré, voir le rapport) : voiture à l'arrêt, boost pressé
    for name, thr, doc in (
        ("boost_ground_throttle0", 0.0, "boost sans gaz"),
        ("boost_ground_throttle_neg", -1.0, "boost avec marche arrière pleine"),
        ("boost_ground_throttle1", 1.0, "boost avec plein gaz (contrôle : doit concorder)"),
    ):
        add(
            Scenario(
                name,
                (CarInit(pos=(-1500.0, -3000.0, CAR_Z), yaw=np.pi / 2, boost=100.0), FAR_IDLE),
                BallInit(pos=(3500.0, 3000.0, 500.0), vel=(0.0, 0.0, -MIDAIR_VZ)),
                const_controls(15, throttle=thr, boost=1.0),
                "boost_override",
                f"Voiture à l'arrêt au sol, {doc}.",
            )
        )
    return S


RANDOM_SEEDS = 8

# --------------------------------------------------------------------------------------------- simulateurs


class RocketSimWorld:
    def __init__(self, rs: Any, sc: Scenario) -> None:
        self.rs = rs
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.cars = [self.arena.add_car(rs.Team.BLUE), self.arena.add_car(rs.Team.ORANGE)]
        self.arena.reset_kickoff(0)
        for car, init in zip(self.cars, sc.cars, strict=True):
            st = car.get_state()
            r = init.rot if init.rot is not None else rot_from_ypr(init.yaw)
            st.pos = rs.Vec(*init.pos)
            st.vel = rs.Vec(*init.vel)
            st.ang_vel = rs.Vec(*init.ang_vel)
            st.rot_mat = rs.RotMat(forward=rs.Vec(*r[:, 0]), right=rs.Vec(*r[:, 1]), up=rs.Vec(*r[:, 2]))
            st.boost = init.boost
            car.set_state(st)
        bs = self.arena.ball.get_state()
        bs.pos, bs.vel, bs.ang_vel = rs.Vec(*sc.ball.pos), rs.Vec(*sc.ball.vel), rs.Vec(*sc.ball.ang_vel)
        self.arena.ball.set_state(bs)

    def step(self, ctrl: np.ndarray, ticks: int = TICK_SKIP) -> None:
        for car, c in zip(self.cars, ctrl, strict=True):
            cc = self.rs.CarControls()
            cc.throttle, cc.steer, cc.pitch, cc.yaw, cc.roll = (float(x) for x in c[:5])
            cc.jump, cc.boost, cc.handbrake = bool(c[5]), bool(c[6]), bool(c[7])
            car.set_controls(cc)
        self.arena.step(ticks)

    def snapshot(self) -> dict[str, np.ndarray]:
        cs = [c.get_state() for c in self.cars]
        b = self.arena.ball.get_state()
        v = lambda x: np.array([x.x, x.y, x.z], dtype=np.float64)  # noqa: E731
        return {
            "car_pos": np.stack([v(s.pos) for s in cs]),
            "car_vel": np.stack([v(s.vel) for s in cs]),
            "car_angvel": np.stack([v(s.ang_vel) for s in cs]),
            "car_rot": np.stack(
                [np.stack([v(s.rot_mat.forward), v(s.rot_mat.right), v(s.rot_mat.up)], axis=1) for s in cs]
            ),
            "car_boost": np.array([s.boost for s in cs], dtype=np.float64),
            "car_ground": np.array([s.is_on_ground for s in cs]),
            "ball_pos": v(b.pos),
            "ball_vel": v(b.vel),
            "ball_angvel": v(b.ang_vel),
        }


def _raw_copy(view: np.ndarray) -> np.ndarray:
    """Copie inscriptible octet à octet d'un tableau de structures (les dtypes RocketGPU ont des trous)."""
    return np.frombuffer(bytearray(view.tobytes()), dtype=view.dtype)


class RocketGPUWorld:
    """État initial écrit dans les tableaux de structures bruts (`upload_state_array`, mode PublicState).

    Pourquoi pas `upload_car_states([CarState])` : le binding Python de `CarState` n'expose pas `basis_*`; l'état
    téléchargé d'un kickoff a `basis_authoritative=1`, et l'import recalcule alors `rot` depuis `basis_*`
    (`CanonicalizeImportedCarOrientation`, math.hpp:1306) : l'orientation envoyée via `.rot` est ignorée (mesuré, voir
    le rapport). On écrit donc `basis_forward/right/up` (colonnes de la matrice, même convention que RocketSim) comme
    `SeedCarBasisFromRotMat` (rocketsim_compat.cpp:25). Même raison pour `ball_states.sleeping` : la balle du
    kickoff est endormie et `BallState` n'expose pas ce champ.
    """

    def __init__(self, rg: Any, sc: Scenario) -> None:
        import rocketgpu_py_cuda as views

        self.rg, self.views = rg, views
        cfg = rg.SimConfig()
        cfg.env_count = 1
        cfg.max_cars_per_env = 2
        self.sim = rg.BatchSim(cfg)
        self.sim.set_env_car_count([2])
        self.sim.set_teams([rg.Team.Blue, rg.Team.Orange])
        self.sim.reset_to_kickoff(0)
        cars = _raw_copy(views.host_struct_array(self.sim, "car_states"))  # copie octet à octet, inscriptible
        for i, init in enumerate(sc.cars):
            r = init.rot if init.rot is not None else rot_from_ypr(init.yaw)
            cars["pos"][i], cars["vel"][i], cars["ang_vel"][i] = init.pos, init.vel, init.ang_vel
            cars["basis_forward"][i], cars["basis_right"][i], cars["basis_up"][i] = r[:, 0], r[:, 1], r[:, 2]
            cars["basis_authoritative"][i] = 1
            cars["pad_frame_pos_valid"][i] = 0  # sinon balayage des pads entre l'ancienne et la nouvelle position
            cars["boost_pad_touching"][i] = 0
            cars["boost"][i] = init.boost
        self.sim.upload_state_array("car_states", cars.ctypes.data, cars.shape[0])
        ball = _raw_copy(views.host_struct_array(self.sim, "ball_states"))
        ball["pos"][0], ball["vel"][0], ball["ang_vel"][0] = sc.ball.pos, sc.ball.vel, sc.ball.ang_vel
        ball["basis_forward"][0], ball["basis_right"][0], ball["basis_up"][0] = (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ball["basis_authoritative"][0] = 1
        ball["sleeping"][0] = 0
        ball["deactivation_time"][0] = 0.0
        self.sim.upload_state_array("ball_states", ball.ctypes.data, ball.shape[0])
        self.cars = views.host_struct_array(self.sim, "car_states")  # vues vivantes (CPU)
        self.ball = views.host_struct_array(self.sim, "ball_states")

    def step(self, ctrl: np.ndarray, ticks: int = TICK_SKIP) -> None:
        out = []
        for c in ctrl:
            cc = self.rg.CarControls()
            cc.throttle, cc.steer, cc.pitch, cc.yaw, cc.roll = (float(x) for x in c[:5])
            cc.jump, cc.boost, cc.handbrake = bool(c[5]), bool(c[6]), bool(c[7])
            out.append(cc)
        self.sim.upload_controls(out)
        self.sim.step(ticks)

    def snapshot(self) -> dict[str, np.ndarray]:
        c, b = self.cars, self.ball
        f8 = lambda x: np.array(x, dtype=np.float64)  # noqa: E731
        rot = np.stack(
            [np.stack([c["basis_forward"][i], c["basis_right"][i], c["basis_up"][i]], axis=1) for i in range(2)]
        )
        return {
            "car_pos": f8(c["pos"]),
            "car_vel": f8(c["vel"]),
            "car_angvel": f8(c["ang_vel"]),
            "car_rot": f8(rot),
            "car_boost": f8(c["boost"]),
            "car_ground": np.array(c["is_on_ground"], dtype=bool),
            "ball_pos": f8(b["pos"][0]),
            "ball_vel": f8(b["vel"][0]),
            "ball_angvel": f8(b["ang_vel"][0]),
        }


METRICS = ("pos", "vel", "angvel", "rot", "boost")  # voiture ; balle : pos, vel, angvel


WARMUP_TICKS = 1


def run_scenario(rg: Any, rs: Any, sc: Scenario, warmup: int = WARMUP_TICKS) -> dict[str, np.ndarray]:
    """Retourne les séries d'écarts [n_steps+1] : `car0_pos`, `car1_vel`, `ball_pos`, ... (indice 0 = état initial).

    `warmup` : ticks sans contrôle exécutés dans les deux simulateurs avant la mesure, pour que les deux voitures soient
    posées (RocketSim indique `wheels_with_contact` = 0000 juste après un kickoff). Mesuré : l'effet sur les écarts est
    faible (comparer `--report` et `--report --cold` dans docs/reports/physics_parity.md) ; le décalage d'un tick de
    la réponse au gaz subsiste même après 30 ticks d'échauffement (voir `lag_probe`).
    """
    a, b = RocketSimWorld(rs, sc), RocketGPUWorld(rg, sc)
    if warmup:
        idle = np.zeros((2, 8))
        a.step(idle, warmup)
        b.step(idle, warmup)
    snaps_a, snaps_b = [a.snapshot()], [b.snapshot()]
    for k in range(sc.n_steps):
        a.step(sc.controls[k])
        b.step(sc.controls[k])
        snaps_a.append(a.snapshot())
        snaps_b.append(b.snapshot())
    out: dict[str, list[float]] = {}
    for sa, sb in zip(snaps_a, snaps_b, strict=True):
        for i in range(2):
            out.setdefault(f"car{i}_pos", []).append(float(np.linalg.norm(sa["car_pos"][i] - sb["car_pos"][i])))
            out.setdefault(f"car{i}_vel", []).append(float(np.linalg.norm(sa["car_vel"][i] - sb["car_vel"][i])))
            out.setdefault(f"car{i}_angvel", []).append(
                float(np.linalg.norm(sa["car_angvel"][i] - sb["car_angvel"][i]))
            )
            out.setdefault(f"car{i}_rot", []).append(angle_between(sa["car_rot"][i], sb["car_rot"][i]))
            out.setdefault(f"car{i}_boost", []).append(float(abs(sa["car_boost"][i] - sb["car_boost"][i])))
            out.setdefault(f"car{i}_ground_mismatch", []).append(float(sa["car_ground"][i] != sb["car_ground"][i]))
        out.setdefault("ball_pos", []).append(float(np.linalg.norm(sa["ball_pos"] - sb["ball_pos"])))
        out.setdefault("ball_vel", []).append(float(np.linalg.norm(sa["ball_vel"] - sb["ball_vel"])))
        out.setdefault("ball_angvel", []).append(float(np.linalg.norm(sa["ball_angvel"] - sb["ball_angvel"])))
        out.setdefault("min_z", []).append(float(min(sb["car_pos"][:, 2].min(), sb["ball_pos"][2])))
        out.setdefault("finite", []).append(
            float(all(np.isfinite(v).all() for v in sb.values() if v.dtype.kind == "f"))
        )
    return {k: np.array(v) for k, v in out.items()}


def at(series: np.ndarray, ticks: int) -> float | None:
    i = ticks // TICK_SKIP
    return float(series[i]) if i < len(series) else None


def lag_probe(rg: Any, rs: Any, sc: Scenario, horizons: tuple[int, ...] = (16, 64, 120)) -> dict[int, dict[str, float]]:
    """Pas d'un tick : écart de position/vitesse (voiture 0) entre RocketSim à t et RocketGPU à t - lag (0 ou 1).
    Les contrôles de `sc` sont tenus TICK_SKIP ticks comme ailleurs ; 30 ticks d'échauffement sans contrôle avant."""
    a, b = RocketSimWorld(rs, sc), RocketGPUWorld(rg, sc)
    idle = np.zeros((2, 8))
    a.step(idle, 30)
    b.step(idle, 30)
    total = max(horizons)
    pa, va, pb, vb = [], [], [], []
    for t in range(total):
        c = sc.controls[min(t // TICK_SKIP, sc.n_steps - 1)]
        a.step(c, 1)
        b.step(c, 1)
        sa, sb = a.snapshot(), b.snapshot()
        pa.append(sa["car_pos"][0])
        va.append(sa["car_vel"][0])
        pb.append(sb["car_pos"][0])
        vb.append(sb["car_vel"][0])
    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        i = h - 1  # état après h ticks
        out[h] = {
            "pos_lag0": float(np.linalg.norm(pa[i] - pb[i])),
            "vel_lag0": float(np.linalg.norm(va[i] - vb[i])),
            "pos_lag1": float(np.linalg.norm(pa[i] - pb[i - 1])),  # RocketGPU un tick plus tôt
            "vel_lag1": float(np.linalg.norm(va[i] - vb[i - 1])),
        }
    return out


# --------------------------------------------------------------------------------------------- tests

try:
    _RG, _RS, _MESH = _load()
    _SKIP: str | None = None
except (ImportError, RuntimeError, OSError) as exc:  # RocketGPU, RocketSim ou meshes absents : test sauté
    _RG = _RS = _MESH = None
    _SKIP = f"RocketGPU/RocketSim indisponible : {exc}"

pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
SCENARIOS = build_scenarios()
_CACHE: dict[str, dict[str, np.ndarray]] = {}


def result(name: str, warmup: int = WARMUP_TICKS) -> dict[str, np.ndarray]:
    key = f"{name}|{warmup}"
    if key not in _CACHE:
        _CACHE[key] = run_scenario(_RG, _RS, SCENARIOS[name], warmup)
    return _CACHE[key]


# Seuils des scénarios déterministes. FIXÉS APRÈS MESURE (docs/reports/physics_parity.md, section « Seuils »).
# Règle : seuil = environ 2 à 2,5 fois le pire écart mesuré sur la famille à l'horizon donné, arrondi. Il détecte une
# régression de l'ordre de 2x le niveau actuel sans être fragile au bruit flottant ; il N'AFFIRME PAS la parité avec le
# jeu, seulement la cohérence mesurée des deux simulateurs, et n'a pas été ajusté pour faire passer un scénario.
# Pire écart mesuré (warmup 1 tick, commit f4f1a45 + ce fichier, 2026-11-11) -> seuil :
#   kickoff @16  : pos 1,73 -> 4 ; vel 11,2 -> 25 ; angvel 0,032 -> 0,1 ; rot 0,0081 -> 0,03 ; boost 0,282 -> 0,6
#   kickoff @120 : pos 5,26 -> 12 ; vel 3,8 -> 10 ; angvel 0,0143 -> 0,05 ; rot 0,0024 -> 0,01 ; boost 0,311 -> 0,7
#   jump @16     : pos 0,92 -> 2 ; vel 7,9 -> 16 ; angvel 0,0225 -> 0,06 ; rot 0,0037 -> 0,01 ; boost 0 -> 0,01
#   jump @120    : pos 7,7 -> 16 ; vel 8,3 -> 17 ; angvel 0,099 -> 0,2 ; rot 0,060 -> 0,12 ; boost 0 -> 0,01
#   ball @120    : ball_pos 0,63 -> 1,5 ; ball_vel 1,99 -> 4 ; ball_angvel 0,002 -> 0,01
#   ball @240    : ball_pos 3,0 -> 6 ; ball_vel 2,64 -> 5,5 ; ball_angvel 0,0078 -> 0,02
#   hit @120     : car0_pos 0,245 (hit_fast_offset ; 0,045 ailleurs) -> 0,6 ; ball_pos 0 -> 0,1 (avant contact)
#   hit @240     : car0_pos 0,84 -> 2 ; car0_vel 1,57 -> 3,2 ; ball_pos 4,75 -> 10 ; ball_vel 5,43 -> 11 ;
#                  ball_angvel 0,041 -> 0,08
# Les horizons 600 et 2400 ne sont PAS testés (contacts répétés, chaos) : ils sont rapportés seulement.
TOL: dict[tuple[str, int], dict[str, float]] = {
    ("kickoff", 16): dict(car0_pos=4, car0_vel=25, car0_angvel=0.1, car0_rot=0.03, car0_boost=0.6),
    ("kickoff", 120): dict(car0_pos=12, car0_vel=10, car0_angvel=0.05, car0_rot=0.01, car0_boost=0.7),
    ("jump", 16): dict(car0_pos=2, car0_vel=16, car0_angvel=0.06, car0_rot=0.01, car0_boost=0.01),
    ("jump", 120): dict(car0_pos=16, car0_vel=17, car0_angvel=0.2, car0_rot=0.12, car0_boost=0.01),
    ("ball", 120): dict(ball_pos=1.5, ball_vel=4, ball_angvel=0.01),
    ("ball", 240): dict(ball_pos=6, ball_vel=5.5, ball_angvel=0.02),
    ("hit", 120): dict(car0_pos=0.6, ball_pos=0.1),
    ("hit", 240): dict(car0_pos=2, car0_vel=3.2, ball_pos=10, ball_vel=11, ball_angvel=0.08),
}


def _family(name: str) -> str:
    if name.startswith("kickoff_") or name == "idle_kickoff":
        return "kickoff"
    if name.startswith(("jump_", "double_", "flip_")):
        return "jump"
    if name.startswith("ball_"):
        return "ball"
    if name.startswith("boost_ground"):
        return "boost_override"
    if name.startswith("hit_"):
        return "hit"
    return name


DET = [n for n, s in SCENARIOS.items() if s.kind == "deterministic"]
BOOST_OVERRIDE = [n for n, s in SCENARIOS.items() if s.kind == "boost_override"]
CONTACT = [n for n, s in SCENARIOS.items() if s.kind == "contact"]
CHAOTIC = [n for n, s in SCENARIOS.items() if s.kind == "chaotic"]


def test_initial_states_identical() -> None:
    """Contrôle de la harnais : à l'instant 0 les deux simulateurs ont le même état (conventions quaternion, ordre
    des voitures, unités). Un écart ici signale une erreur de conversion, pas de physique."""
    for name in ("idle_kickoff", "flip_diag", "ball_drop_spin", "hit_offset"):
        r = result(name, 0)  # sans échauffement : l'indice 0 est l'état écrit
        for key in (
            "car0_pos",
            "car1_pos",
            "car0_vel",
            "car0_angvel",
            "ball_pos",
            "ball_vel",
            "ball_angvel",
            "car0_boost",
        ):
            assert r[key][0] < 1e-3, (name, key, r[key][0])
        assert r["car0_rot"][0] < 1e-3 and r["car1_rot"][0] < 1e-3, (name, r["car0_rot"][0])


@pytest.mark.parametrize("name", DET)
def test_deterministic_scenarios(name: str) -> None:
    r = result(name)
    assert all(r["finite"] == 1.0), "état non fini dans RocketGPU"
    assert r["min_z"].min() > -10.0, "objet sous le sol dans RocketGPU"
    checked = 0
    for (family, horizon), bounds in TOL.items():
        if family != _family(name):
            continue
        for key, bound in bounds.items():
            val = at(r[key], horizon)
            assert val is not None, (name, key, horizon)
            assert val <= bound, f"{name} {key} @{horizon} ticks : {val:.4g} > seuil {bound}"
            checked += 1
    assert checked > 0, f"aucun seuil pour la famille de {name}"


@pytest.mark.parametrize("name", CONTACT)
def test_contact_scenarios(name: str) -> None:
    """Frappe de balle par une voiture lancée : seuils serrés jusqu'à 240 ticks (premier contact inclus)."""
    r = result(name)
    assert all(r["finite"] == 1.0)
    assert r["min_z"].min() > -10.0
    for horizon in (120, 240):
        for key, bound in TOL[("hit", horizon)].items():
            val = at(r[key], horizon)
            assert val is not None and val <= bound, f"{name} {key} @{horizon} : {val} > {bound}"


@pytest.mark.parametrize("name", CHAOTIC)
def test_chaotic_scenarios_sane(name: str) -> None:
    """Pas d'exigence d'égalité (divergence chaotique, rapportée dans docs/reports/physics_parity.md) : états finis,
    rien sous le sol, balle dans l'arène."""
    r = result(name)
    assert all(r["finite"] == 1.0)
    assert r["min_z"].min() > -10.0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="ÉCART MESURÉ : RocketGPU applique le frein au sol quand |gaz| <= 0,01 même en boostant "
    "(raycast_wheels_and_contact_prep.hpp:2713-2722) ; RocketSim 2.2.1 accélère quelle que soit la valeur du gaz. "
    "Retirer ce xfail quand l'écart est corrigé ou toléré par décision (ADR).",
)
@pytest.mark.parametrize("name", ["boost_ground_throttle0", "boost_ground_throttle_neg"])
def test_boost_override_on_ground(name: str) -> None:
    """Voiture à l'arrêt, boost pressé : l'écart de vitesse après 120 ticks doit rester <= 50 uu/s (critère d'exigence,
    pas un seuil issu de la mesure ; écart mesuré : voir le rapport)."""
    r = result(name)
    vel_err = at(r["car0_vel"], 120)
    assert vel_err is not None and vel_err <= 50.0, vel_err


def test_boost_override_control_case_matches() -> None:
    """Contrôle : avec plein gaz, boost au sol concorde (les seuils `kickoff` s'appliquent)."""
    r = result("boost_ground_throttle1")
    assert at(r["car0_vel"], 120) is not None and at(r["car0_vel"], 120) <= 25.0  # type: ignore[operator]
    assert at(r["car0_boost"], 120) is not None and at(r["car0_boost"], 120) <= 0.7  # type: ignore[operator]


def test_zero_velocity_midair_ball_known_difference() -> None:
    """Caractérisation (pas une exigence de parité) : une balle posée en l'air à vitesse EXACTEMENT nulle reste
    suspendue dans RocketSim 2.2.1 et tombe dans RocketGPU. Si RocketSim change, ce test le signalera."""
    sc = Scenario(
        "ball_zero_vel",
        (CarInit(**BLUE_KICKOFF), FAR_IDLE),  # type: ignore[arg-type]
        BallInit(pos=(1000.0, -500.0, 1500.0)),
        const_controls(15),
        "deterministic",
    )
    a, b = RocketSimWorld(_RS, sc), RocketGPUWorld(_RG, sc)
    for k in range(15):
        a.step(sc.controls[k])
        b.step(sc.controls[k])
    assert a.snapshot()["ball_pos"][2] == pytest.approx(1500.0, abs=0.01)
    assert b.snapshot()["ball_pos"][2] < 1400.0


# --------------------------------------------------------------------------------------------- rapport


def _fmt(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}g}"


RANDOM_TAGS = tuple(f"{t}_{h}" for h in ("5s", "20s") for t in ("uniform", "uniform_tb", "biased", "biased_tb"))


def report(warmup: int = WARMUP_TICKS) -> None:
    cols = ("car0_pos", "car0_vel", "car0_angvel", "car0_rot", "car0_boost", "ball_pos", "ball_vel", "ball_angvel")
    print("| scénario | horizon (ticks) | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    groups: dict[str, list[str]] = {}
    for n, s in SCENARIOS.items():
        groups.setdefault("random" if s.kind == "chaotic" else n, []).append(n)
    for g, names in groups.items():
        if g != "random":
            r = result(names[0], warmup)
            for h in HORIZONS:
                if at(r["car0_pos"], h) is None:
                    continue
                print(f"| {g} | {h} | " + " | ".join(_fmt(at(r[c], h)) for c in cols) + " |")
        else:
            for tag in RANDOM_TAGS:
                sel = [n for n in names if n.startswith(f"random_{tag}_") and n[len(f"random_{tag}_") :].isdigit()]
                for h in HORIZONS:
                    vals = {c: [at(result(n, warmup)[c], h) for n in sel] for c in cols}
                    if vals["car0_pos"][0] is None:
                        continue
                    cells = [
                        f"{_fmt(float(np.median(v)))} / {_fmt(float(np.max(v)))}"  # type: ignore[arg-type]
                        for v in vals.values()
                    ]
                    print(f"| random_{tag} (médiane / max, n={len(sel)}) | {h} | " + " | ".join(cells) + " |")


LAG_SCENARIOS = (
    "kickoff_throttle",
    "kickoff_throttle_steer",
    "kickoff_throttle_boost",
    "kickoff_reverse",
    "jump_tap",
    "flip_forward",
)


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    p.add_argument("--cold", action="store_true", help="sans tick d'échauffement (warmup=0)")
    p.add_argument("--lag", action="store_true", help="sonde de décalage d'un tick (voiture 0)")
    p.add_argument("--series", help="nom de scénario : imprime la série complète d'écarts")
    args = p.parse_args(argv)
    if _SKIP:
        raise SystemExit(_SKIP)
    if args.report:
        report(0 if args.cold else WARMUP_TICKS)
    if args.lag:
        print("| scénario | horizon | pos lag0 | pos lag1 | vel lag0 | vel lag1 |")
        print("|---|---|---|---|---|---|")
        for n in LAG_SCENARIOS:
            for h, v in lag_probe(_RG, _RS, SCENARIOS[n]).items():
                print(
                    f"| {n} | {h} | {v['pos_lag0']:.3g} | {v['pos_lag1']:.3g} "
                    f"| {v['vel_lag0']:.3g} | {v['vel_lag1']:.3g} |"
                )
    if args.series:
        r = result(args.series, 0 if args.cold else WARMUP_TICKS)
        keys = [k for k in r if k not in ("finite", "min_z")]
        print("tick " + " ".join(keys))
        for i in range(len(r["ball_pos"])):
            print(i * TICK_SKIP, *(f"{r[k][i]:.4g}" for k in keys))


Callable  # noqa: B018 (import conservé pour les annotations futures)

if __name__ == "__main__":
    _main()
