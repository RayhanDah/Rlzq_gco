"""Configuration d'un run d'entraînement : dataclasses typées, validées, chargées depuis YAML.

Toute clé inconnue, tout type faux (un `bool` n'est pas un entier, une chaîne n'est pas un nombre), toute valeur hors
domaine lève `ConfigError` avec le chemin de la clé fautive. Les valeurs par défaut viennent de `docs/01_papers.md` §2
(PPO), de `docs/GCO_SPEC.md` §2.3 (p_kick 0,2 -> 0,8) et de `docs/PLAN.md` (gardes), voir `PPOConfig` et `GuardConfig`.

Deux hashs :
- `RunConfig.trajectory_hash()` : tout ce qui détermine la trajectoire d'entraînement (env, récompense, états initiaux,
  modèle, PPO, collecte, budget, seed, device). C'est lui qui est écrit dans les checkpoints et vérifié à la reprise ;
- `RunConfig.full_hash()` : la config entière (y compris checkpoints, gardes, journalisation, section smoke), écrite
  dans `run_meta.json`. Changer la fréquence des checkpoints ou le seuil d'une garde ne bloque donc pas une reprise.

Le budget est en timesteps agents ; `total_iterations` (dont dépendent les calendriers LR et entropie de `PPOConfig`)
est dérivé : ⌈total_timesteps / (n_envs · n_cars · steps_per_batch)⌉.
"""

from __future__ import annotations

import dataclasses
import math
import re
import types
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from rewards import RewardConfig, RewardVariant, rules_for
from training.checkpoint import hash_config
from training.guards import GuardConfig
from training.ppo import PPOConfig

_DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d+)?|xpu(:\d+)?)$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# Sections qui ne changent pas la trajectoire d'entraînement : exclues du hash vérifié à la reprise.
OPERATIONAL_SECTIONS = ("checkpoint", "kill", "logging", "smoke")


class ConfigError(ValueError):
    """Configuration invalide (clé inconnue, type faux, valeur hors domaine, clé manquante)."""


@dataclass(frozen=True, kw_only=True)
class EnvSection:
    n_envs: int = 4096  # ADR-001 M4
    n_cars: int = 2
    tick_skip: int = 8

    def __post_init__(self) -> None:
        if self.n_envs < 1:
            raise ValueError(f"n_envs={self.n_envs} < 1")
        if self.n_cars < 2 or self.n_cars % 2:
            raise ValueError(f"n_cars={self.n_cars} : pair et >= 2 attendu")
        if self.tick_skip < 1:
            raise ValueError(f"tick_skip={self.tick_skip} < 1")


@dataclass(frozen=True, kw_only=True)
class RewardSection:
    """Variante de récompense (`rewards.RewardVariant`) et limites d'épisode (spec §2.2)."""

    variant: str = "GCO-std"
    no_touch_timeout_s: float = 20.0
    max_episode_s: float = 120.0

    def build(self) -> RewardConfig:
        return RewardConfig(
            variant=RewardVariant(self.variant),
            no_touch_timeout_s=self.no_touch_timeout_s,
            max_episode_s=self.max_episode_s,
        )

    def __post_init__(self) -> None:
        self.build()  # ValueError si variante inconnue ou durées hors tick


@dataclass(frozen=True, kw_only=True)
class InitialStatesSection:
    """Calendrier de `p_kick` (spec §2.3) : linéaire de `p_kick_start` à `p_kick_end` sur `anneal_timesteps`."""

    p_kick_start: float = 0.2
    p_kick_end: float = 0.8
    anneal_timesteps: int

    def __post_init__(self) -> None:
        for name in ("p_kick_start", "p_kick_end"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name}={getattr(self, name)} hors de [0, 1]")
        if self.anneal_timesteps < 0:
            raise ValueError("anneal_timesteps < 0")


@dataclass(frozen=True, kw_only=True)
class ModelSection:
    """Tailles des couches cachées ; None = défauts de `models` (256×3)."""

    policy_hidden: tuple[int, ...] | None = None
    value_hidden: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("policy_hidden", "value_hidden"):
            hidden = getattr(self, name)
            if hidden is not None and (len(hidden) == 0 or min(hidden) < 1):
                raise ValueError(f"{name} : entiers >= 1 attendus")


@dataclass(frozen=True, kw_only=True)
class CollectionSection:
    steps_per_batch: int  # ADR-001 M4 : 7 steps par lot à 4 096 arènes
    overlap: bool = True  # collecte du lot k+1 pendant l'apprentissage du lot k (staleness 1)

    def __post_init__(self) -> None:
        if self.steps_per_batch < 1:
            raise ValueError("steps_per_batch < 1")


@dataclass(frozen=True, kw_only=True)
class BudgetSection:
    total_timesteps: int  # timesteps agents (transitions)

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps < 1")


@dataclass(frozen=True, kw_only=True)
class CheckpointSection:
    every_iterations: int = 50
    keep: int = 3  # `CheckpointManager.keep` ; le dernier checkpoint sain est conservé en plus
    include_sim_state: bool = True  # reprise exacte (état du simulateur) ; sinon « modulo reset des arènes »

    def __post_init__(self) -> None:
        if self.every_iterations < 1 or self.keep < 1:
            raise ValueError("every_iterations >= 1 et keep >= 1")


@dataclass(frozen=True, kw_only=True)
class LoggingSection:
    every_iterations: int = 1
    fsync: bool = False

    def __post_init__(self) -> None:
        if self.every_iterations < 1:
            raise ValueError("every_iterations >= 1")


@dataclass(frozen=True, kw_only=True)
class SmokeSection:
    """Exigences du smoke test (`scripts/train.py --smoke`) : reprise obligatoire au milieu du run, zéro NaN."""

    resume_at_fraction: float = 0.5
    require_resume: bool = True
    max_nan: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.resume_at_fraction < 1.0:
            raise ValueError("resume_at_fraction doit être dans ]0, 1[")
        if self.max_nan < 0:
            raise ValueError("max_nan >= 0")


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    name: str
    seed: int = 0
    device: str = "auto"
    env: EnvSection = field(default_factory=EnvSection)
    reward: RewardSection = field(default_factory=RewardSection)
    initial_states: InitialStatesSection
    model: ModelSection = field(default_factory=ModelSection)
    ppo: PPOConfig
    collection: CollectionSection
    budget: BudgetSection
    checkpoint: CheckpointSection = field(default_factory=CheckpointSection)
    kill: GuardConfig = field(default_factory=GuardConfig)
    logging: LoggingSection = field(default_factory=LoggingSection)
    smoke: SmokeSection | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"name={self.name!r} : lettres, chiffres, '_', '.', '-' uniquement")
        if self.seed < 0:
            raise ValueError("seed >= 0")
        if not _DEVICE_RE.match(self.device):
            raise ValueError(f"device={self.device!r} : auto, cpu, cuda[:n] ou xpu[:n]")
        if self.ppo.total_iterations != self.total_iterations:
            raise ValueError(
                f"ppo.total_iterations={self.ppo.total_iterations} != {self.total_iterations} "
                "(dérivé du budget : utiliser RunConfig.from_dict)"
            )
        if not rules_for(RewardVariant(self.reward.variant)).generic_initial_states:
            st = self.initial_states
            if st.p_kick_start != 1.0 or st.p_kick_end != 1.0:
                raise ValueError(
                    f"variante {self.reward.variant} : états initiaux génériques interdits (spec §3), "
                    "p_kick_start et p_kick_end doivent valoir 1.0"
                )

    # ---------------------------------------------------------------------------------------------- dérivés

    @property
    def n_agents(self) -> int:
        return self.env.n_envs * self.env.n_cars

    @property
    def batch_agent_steps(self) -> int:
        """Transitions agents par lot (et par itération)."""
        return self.n_agents * self.collection.steps_per_batch

    @property
    def total_iterations(self) -> int:
        return derive_total_iterations(self.env, self.collection, self.budget)

    @property
    def anneal_iterations(self) -> int:
        return math.ceil(self.initial_states.anneal_timesteps / self.batch_agent_steps)

    def p_kick_at(self, iteration: int) -> float:
        """`p_kick` à utiliser pour l'itération numéro `iteration` (0 = première) : linéaire puis constant.

        L'annealing couvre `n = anneal_iterations` itérations (indices 0 à n-1) : l'itération 0 utilise `p_kick_start`,
        l'itération n-1 (la dernière de l'annealing) utilise `p_kick_end`, et la valeur reste à la cible ensuite.
        Avant A1 la cible n'était atteinte qu'à l'indice n, soit une itération trop tard (0,767 au lieu de 0,8 à la
        dernière itération de `configs/smoke.yaml`). Cas dégénérés : n = 0 ou 1 (pas de rampe possible) : cible dès
        l'itération 0."""
        st = self.initial_states
        n = self.anneal_iterations
        frac = 1.0 if n <= 1 else min(max(iteration, 0) / (n - 1), 1.0)
        return st.p_kick_start + frac * (st.p_kick_end - st.p_kick_start)

    # ---------------------------------------------------------------------------------------------- hash

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = dataclasses.asdict(self)
        return out

    def trajectory_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        for k in OPERATIONAL_SECTIONS:
            d.pop(k)
        return d

    def trajectory_hash(self) -> str:
        return hash_config(self.trajectory_dict())

    def full_hash(self) -> str:
        return hash_config(self.to_dict())

    # ---------------------------------------------------------------------------------------------- chargement

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        if not isinstance(data, Mapping):
            raise ConfigError("la configuration doit être un mapping")
        raw = dict(data)
        ppo_raw = raw.get("ppo")
        if not isinstance(ppo_raw, Mapping):
            raise ConfigError("ppo : section manquante ou qui n'est pas un mapping")
        if "total_iterations" in ppo_raw:
            raise ConfigError("ppo.total_iterations : clé dérivée du budget, ne pas la fournir")
        env = _build(EnvSection, raw.get("env", {}), "env")
        collection = _build(CollectionSection, raw.get("collection"), "collection")
        budget = _build(BudgetSection, raw.get("budget"), "budget")
        raw["ppo"] = {**ppo_raw, "total_iterations": derive_total_iterations(env, collection, budget)}
        built = _build(cls, raw, "config")
        assert isinstance(built, cls)
        return built

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        try:
            import yaml  # importé ici : `import training` ne doit pas exiger PyYAML (venv XPU, dicts Python)
        except ImportError as exc:
            raise ConfigError("PyYAML absent de cet environnement : `uv pip install pyyaml==6.0.3`") from exc
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} : YAML invalide : {exc}") from exc
        return cls.from_dict(data)


def derive_total_iterations(env: EnvSection, collection: CollectionSection, budget: BudgetSection) -> int:
    return math.ceil(budget.total_timesteps / (env.n_envs * env.n_cars * collection.steps_per_batch))


# -------------------------------------------------------------------------------------------------- construction


def _build(cls: type[Any], data: Any, path: str) -> Any:
    """Construit la dataclass `cls` depuis un mapping : clés inconnues et types faux refusés, avec chemin."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} : mapping attendu, reçu {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{path} : clés inconnues {unknown} (connues : {sorted(known)})")
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, f in known.items():
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], f"{path}.{name}")
        elif f.default is MISSING and f.default_factory is MISSING:
            missing.append(name)
    if missing:
        raise ConfigError(f"{path} : clés obligatoires absentes {missing}")
    try:
        return cls(**kwargs)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{path} : {exc}") from exc


_SCI_NO_DOT_RE = re.compile(r"^[+-]?\d+[eE][+-]?\d+$")


def _sci_hint(value: Any, tp: type) -> str:
    """Explication ajoutée au message quand une chaîne est en réalité une notation scientifique sans point : la
    résolution YAML 1.1 de PyYAML exige un point (`2.0e-4`) et rend sinon `2e-4` comme une chaîne."""
    if not isinstance(value, str) or not _SCI_NO_DOT_RE.match(value.strip()):
        return ""
    head = f" : {value!r} est une notation scientifique sans point, que PyYAML lit comme une chaîne ; "
    if tp is float:
        mantissa, exp = value.strip().lower().split("e")
        return head + f"écrire {mantissa}.0e{exp} (avec un point)"
    number = float(value)
    as_int = f"{int(number)}" if number == int(number) else "l'entier en toutes lettres"
    return head + f"écrire l'entier en toutes lettres ({as_int})"


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        if value is None and type(None) in args:
            return None
        rest = [a for a in args if a is not type(None)]
        if len(rest) != 1:
            raise TypeError(f"{path} : union non gérée {tp}")
        return _coerce(rest[0], value, path)
    if origin is tuple:
        elem = get_args(tp)[0]
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{path} : liste attendue, reçu {type(value).__name__}")
        return tuple(_coerce(elem, v, f"{path}[{i}]") for i, v in enumerate(value))
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} : booléen attendu, reçu {type(value).__name__} ({value!r})")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(
                f"{path} : entier attendu, reçu {type(value).__name__} ({value!r}){_sci_hint(value, int)}"
            )
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(
                f"{path} : nombre attendu, reçu {type(value).__name__} ({value!r}){_sci_hint(value, float)}"
            )
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} : chaîne attendue, reçu {type(value).__name__} ({value!r})")
        return value
    raise TypeError(f"{path} : type de champ non géré {tp}")
