"""Façonnage de la variante `dense-diag` : HORS SPEC, diagnostic du pipeline, jamais promue (ADR-011 D1).

    r_t(i) = g_t(i)                                                   # §2.1 exact (gco_team_rewards)
           + 0,001  * max(0, v_voiture . u(voiture -> balle)) / 2300
           + 0,01   * max(0, v_balle . u(balle -> fond du but adverse de i)) / 6000
           + 0,0002 * 1[voiture pas au sol]

Guide RLGym officiel, poids divisés par 10 pour garder le but à +-1 (ADR-011 D1). Normalisation et cibles :
`rlgym` v1.2.0, `rlgym/utils/common_values.py:12-18` (ORANGE_GOAL_BACK, BLUE_GOAL_BACK, BALL_MAX_SPEED, CAR_MAX_SPEED).

Repère et unités : ceux de `obs.state.BatchState` (monde du sim, uu et uu/s, z vers le haut, filet bleu en y < 0). Le
bleu (équipe 0) attaque donc y > 0, cible (0, +6000, 321,3875) ; l'orange (équipe 1) cible (0, -6000, 321,3875).
État lu en FIN de pas de décision (après les 8 ticks), pas avant ni pendant.

Points à ne pas oublier :
- Les termes de façonnage s'appliquent à CHAQUE pas, y compris le pas d'un but et les pas de troncature (la récompense
  est par pas, ADR-011) : un pas de troncature n'est pas nul, et un pas de but vaut +-1 plus le
  façonnage du pas. Le bootstrap sur V(s') des troncatures est inchangé.
- Le zéro-somme exact de §2.1 n'est PLUS vrai pour cette variante (les termes de façonnage ne se compensent pas entre
  équipes). C'est une des raisons pour lesquelles elle est hors spec.
- Une voiture démolie est traitée comme les autres (état du sim tel quel) ; aucune exception.
- Entrées numpy, vectorisé sur (E, A) ; aucune dépendance au simulateur ni à `obs` (évite un import circulaire).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

W_SPEED_TO_BALL = 0.001
W_BALL_TO_GOAL = 0.01
W_IN_AIR = 0.0002
CAR_MAX_SPEED = 2300.0
"""rlgym v1.2.0, common_values.py : CAR_MAX_SPEED."""
BALL_MAX_SPEED = 6000.0
"""rlgym v1.2.0, common_values.py : BALL_MAX_SPEED."""
ORANGE_GOAL_BACK = (0.0, 6000.0, 321.3875)
BLUE_GOAL_BACK = (0.0, -6000.0, 321.3875)
"""rlgym v1.2.0, common_values.py:12-18. Cible de l'équipe bleue = ORANGE_GOAL_BACK, et inversement."""

_EPS = 1e-6


class KinematicState(Protocol):
    """Sous-ensemble de `obs.state.BatchState` lu par le façonnage (typage structurel, pas d'import)."""

    @property
    def ball_pos(self) -> FloatArray: ...
    @property
    def ball_vel(self) -> FloatArray: ...
    @property
    def car_pos(self) -> FloatArray: ...
    @property
    def car_vel(self) -> FloatArray: ...
    @property
    def car_on_ground(self) -> npt.NDArray[np.uint8]: ...
    @property
    def car_team(self) -> npt.NDArray[np.int64]: ...


def _unit(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Vecteur unitaire sur le dernier axe ; le vecteur nul reste nul (pas de NaN)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > _EPS, v / np.maximum(n, _EPS), np.float32(0.0)).astype(np.float32, copy=False)


def dense_shaping(state: KinematicState) -> FloatArray:
    """Façonnage (E, A) float32 de ADR-011 D1, état en fin de pas. Toujours >= 0, <= 0,0112 par pas."""
    ball_pos = np.asarray(state.ball_pos, dtype=np.float32)[:, None, :]  # (E, 1, 3)
    ball_vel = np.asarray(state.ball_vel, dtype=np.float32)[:, None, :]
    car_pos = np.asarray(state.car_pos, dtype=np.float32)  # (E, A, 3)
    car_vel = np.asarray(state.car_vel, dtype=np.float32)
    team = np.asarray(state.car_team)  # (A,)
    goal = np.where(
        (team == 0)[:, None],
        np.asarray(ORANGE_GOAL_BACK, dtype=np.float32),  # le bleu attaque le fond du but orange
        np.asarray(BLUE_GOAL_BACK, dtype=np.float32),
    )  # (A, 3)

    to_ball = _unit(ball_pos - car_pos)
    speed_to_ball = np.maximum(np.float32(0.0), np.sum(car_vel * to_ball, axis=-1)) / np.float32(CAR_MAX_SPEED)
    to_goal = _unit(goal[None, :, :] - ball_pos)
    ball_to_goal = np.maximum(np.float32(0.0), np.sum(ball_vel * to_goal, axis=-1)) / np.float32(BALL_MAX_SPEED)
    in_air = (np.asarray(state.car_on_ground) == 0).astype(np.float32)
    out = (
        np.float32(W_SPEED_TO_BALL) * speed_to_ball
        + np.float32(W_BALL_TO_GOAL) * ball_to_goal
        + np.float32(W_IN_AIR) * in_air
    )
    return np.asarray(out, dtype=np.float32)
