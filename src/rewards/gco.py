"""Récompense Goal/Concede Only (spec §2.1) et fins d'épisode (§2.2), vectorisées sur un lot d'arènes.

Entrées, sans dépendance au simulateur :
- `goal_net` (E,) : filet touché par la balle pendant le step, `GoalNet` (NONE = -1, BLUE = 0, ORANGE = 1) ;
- `agent_team` (A,) ou (E, A) : équipe de chaque agent, `Team` (BLUE = 0, ORANGE = 1), équipes de même taille ;
- `ticks_since_touch`, `episode_ticks` (E,) : compteurs en ticks physiques (120 Hz), pour les troncatures seulement.

Le dernier toucheur n'est jamais une entrée du signal GCO : le signe vient uniquement du filet touché (csc inclus).
Seule l'ablation hors spec « scorer-only » reçoit un buteur désigné, et uniquement pour retirer le +1 aux coéquipiers.

Choix d'ambiguïté (le plus strict, détaillés dans l'ADR-002) :
- A6. But et condition de troncature au même step : `terminated = True`, `truncated = False` (§7.4 : but ⇒ non tronqué).
- A7. Choix d'interprétation : les troncatures se déclenchent dès que le seuil est atteint (`>=`), lecture de
  « aucune touche depuis 20 s » (`>` serait défendable).
- A8. Équipes de tailles inégales refusées (la somme nulle §2.1 n'est définie que pour NvN).
- A9. scorer-only : le −1 reste attribué à toute l'équipe qui encaisse ; le +1 ne va qu'au buteur désigné s'il est dans
  l'équipe qui marque, sinon à personne (csc). La somme nulle n'y tient pas : c'est pourquoi elle est hors spec.
- A10. Passer un buteur à une variante GCO (S/std/ext) est une erreur, pas un argument ignoré.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from rewards.config import RewardConfig, RewardVariant

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class Team(IntEnum):
    BLUE = 0
    ORANGE = 1


class GoalNet(IntEnum):
    """Filet dans lequel la balle est entrée pendant le step. Valeurs alignées sur `Team` : le filet de l'équipe t."""

    NONE = -1
    BLUE = 0
    """But dans le camp bleu : orange marque."""
    ORANGE = 1
    """But dans le camp orange : bleu marque."""


def _as_int_array(name: str, value: npt.ArrayLike) -> IntArray:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} doit être un tableau d'entiers, dtype reçu {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _goal_net(value: npt.ArrayLike) -> IntArray:
    net = _as_int_array("goal_net", value)
    if net.ndim != 1:
        raise ValueError(f"goal_net doit être de forme (E,), reçu {net.shape}")
    if np.any((net < GoalNet.NONE) | (net > GoalNet.ORANGE)):
        raise ValueError("goal_net hors de {-1, 0, 1}")
    return net


def _agent_team(value: npt.ArrayLike, n_envs: int) -> IntArray:
    teams = _as_int_array("agent_team", value)
    if teams.ndim == 1:
        teams = np.broadcast_to(teams, (n_envs, teams.shape[0]))
    if teams.ndim != 2 or teams.shape[0] != n_envs or teams.shape[1] == 0:
        raise ValueError(f"agent_team doit être de forme (A,) ou ({n_envs}, A) avec A > 0, reçu {teams.shape}")
    if np.any((teams != Team.BLUE) & (teams != Team.ORANGE)):
        raise ValueError("agent_team hors de {0, 1}")
    n_blue = np.count_nonzero(teams == Team.BLUE, axis=1)
    if np.any(2 * n_blue != teams.shape[1]):
        raise ValueError("équipes de tailles inégales : la somme nulle §2.1 exige NvN (choix A8)")
    return teams


def _ticks(name: str, value: npt.ArrayLike, n_envs: int) -> IntArray:
    ticks = _as_int_array(name, value)
    if ticks.shape != (n_envs,):
        raise ValueError(f"{name} doit être de forme ({n_envs},), reçu {ticks.shape}")
    if np.any(ticks < 0):
        raise ValueError(f"{name} doit être >= 0")
    return ticks


def gco_team_rewards(goal_net: npt.ArrayLike, agent_team: npt.ArrayLike) -> FloatArray:
    """Spec §2.1 : +1 à chaque agent dont l'équipe marque, −1 à chaque agent dont l'équipe encaisse, 0 sans but.

    Retourne un tableau (E, A) float32. Somme nulle par arène quand les équipes sont de même taille (vérifié)."""
    net = _goal_net(goal_net)
    teams = _agent_team(agent_team, net.shape[0])
    scored = (net != GoalNet.NONE)[:, None]
    own_net = teams == net[:, None]
    signed = np.where(own_net, np.float32(-1.0), np.float32(1.0))
    return np.where(scored, signed, np.float32(0.0)).astype(np.float32, copy=False)


def scorer_only_rewards(goal_net: npt.ArrayLike, agent_team: npt.ArrayLike, scorer: npt.ArrayLike) -> FloatArray:
    """Ablation HORS SPEC (§2.1) : comme `gco_team_rewards`, mais le +1 ne va qu'au buteur désigné (choix A9).

    `scorer` (E,) : indice d'agent crédité du but dans [0, A), ou -1 si aucun (csc, ou pas de but)."""
    net = _goal_net(goal_net)
    teams = _agent_team(agent_team, net.shape[0])
    who = _as_int_array("scorer", scorer)
    n_agents = teams.shape[1]
    if who.shape != net.shape:
        raise ValueError(f"scorer doit être de forme {net.shape}, reçu {who.shape}")
    if np.any((who < -1) | (who >= n_agents)):
        raise ValueError(f"scorer hors de [-1, {n_agents})")
    base = gco_team_rewards(net, teams)
    is_scorer = np.arange(n_agents)[None, :] == who[:, None]
    return np.where((base > 0) & ~is_scorer, np.float32(0.0), base).astype(np.float32, copy=False)


def episode_end(
    goal_net: npt.ArrayLike,
    ticks_since_touch: npt.ArrayLike,
    episode_ticks: npt.ArrayLike,
    config: RewardConfig,
) -> tuple[BoolArray, BoolArray]:
    """Spec §2.2 et §7.4 : `(terminated, truncated)`, chacun (E,) booléen.

    But ⇒ terminé (pas de bootstrap). Sinon, aucune touche depuis `no_touch_timeout_s` ou durée `max_episode_s`
    atteinte ⇒ tronqué (bootstrap sur V(s′)). Jamais les deux à la fois (choix A6)."""
    net = _goal_net(goal_net)
    n_envs = net.shape[0]
    no_touch = _ticks("ticks_since_touch", ticks_since_touch, n_envs) >= config.no_touch_ticks
    too_long = _ticks("episode_ticks", episode_ticks, n_envs) >= config.max_episode_ticks
    terminated = net != GoalNet.NONE
    truncated = ~terminated & (no_touch | too_long)
    return terminated, truncated


@dataclass(frozen=True, slots=True)
class StepOutcome:
    rewards: FloatArray
    """(E, A) float32."""
    terminated: BoolArray
    """(E,) : but marqué, pas de bootstrap."""
    truncated: BoolArray
    """(E,) : no-touch ou durée max, bootstrap sur V(s′)."""


class RewardFunction:
    """Point d'entrée unique, sélectionné par config (§3, §7.7). Sans état : un appel = un step de décision."""

    __slots__ = ("_config", "_hash")

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self._hash = config.config_hash()

    @property
    def config(self) -> RewardConfig:
        return self._config

    @property
    def variant(self) -> RewardVariant:
        return self._config.variant

    @property
    def config_hash(self) -> str:
        """À écrire dans chaque checkpoint (§7.7)."""
        return self._hash

    def rewards(
        self, goal_net: npt.ArrayLike, agent_team: npt.ArrayLike, scorer: npt.ArrayLike | None = None
    ) -> FloatArray:
        """Récompenses (E, A). `scorer` est exigé par scorer-only et refusé par les variantes GCO (choix A10)."""
        if self._config.rules.team_attribution:
            if scorer is not None:
                raise ValueError(f"{self.variant.value} : le signal ne dépend pas du buteur, scorer doit être None")
            return gco_team_rewards(goal_net, agent_team)
        if scorer is None:
            raise ValueError(f"{self.variant.value} : scorer est requis")
        return scorer_only_rewards(goal_net, agent_team, scorer)

    def episode_end(
        self, goal_net: npt.ArrayLike, ticks_since_touch: npt.ArrayLike, episode_ticks: npt.ArrayLike
    ) -> tuple[BoolArray, BoolArray]:
        return episode_end(goal_net, ticks_since_touch, episode_ticks, self._config)

    def step(
        self,
        goal_net: npt.ArrayLike,
        agent_team: npt.ArrayLike,
        ticks_since_touch: npt.ArrayLike,
        episode_ticks: npt.ArrayLike,
        scorer: npt.ArrayLike | None = None,
    ) -> StepOutcome:
        """Récompenses et drapeaux de fin pour un step de décision sur E arènes."""
        rewards = self.rewards(goal_net, agent_team, scorer)
        terminated, truncated = self.episode_end(goal_net, ticks_since_touch, episode_ticks)
        return StepOutcome(rewards=rewards, terminated=terminated, truncated=truncated)

    def rewards_one(
        self, goal_net: GoalNet | int, agent_team: Sequence[Team | int], scorer: int | None = None
    ) -> tuple[float, ...]:
        """Version une arène, pour l'éval et RLBot ; passe par le même chemin vectorisé."""
        batch_scorer = None if scorer is None else [scorer]
        return tuple(float(r) for r in self.rewards([int(goal_net)], [list(agent_team)], batch_scorer)[0])


def make_reward(config: RewardConfig | Mapping[str, Any]) -> RewardFunction:
    """Construit la fonction de récompense depuis une `RewardConfig` ou un dict de config (clés inconnues refusées)."""
    cfg = config if isinstance(config, RewardConfig) else RewardConfig.from_dict(config)
    return RewardFunction(cfg)
