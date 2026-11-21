"""Configuration d'un run d'entraînement : dataclasses typées, validées, chargées depuis YAML.

Toute clé inconnue, tout type faux (un `bool` n'est pas un entier, une chaîne n'est pas un nombre), toute valeur hors
domaine lève `ConfigError` avec le chemin de la clé fautive. Les valeurs par défaut viennent de `docs/01_papers.md` §2
(PPO), de `docs/GCO_SPEC.md` §2.3 (p_kick 0,2 -> 0,8) et de `docs/PLAN.md` (gardes), voir `PPOConfig` et `GuardConfig`.

Deux hashs :
- `RunConfig.trajectory_hash()` : tout ce qui détermine la trajectoire d'entraînement (env, récompense, états initiaux,
  modèle, PPO, collecte, budget, seed, device). C'est lui qui est écrit dans les checkpoints et vérifié à la reprise ;
- `RunConfig.full_hash()` : la config entière (y compris checkpoints, gardes, journalisation, section smoke), écrite
  dans `run_meta.json`. Changer la fréquence des checkpoints ou le seuil d'une garde ne bloque donc pas une reprise.

Le budget est en timesteps agents APPRIS (transitions qui entrent dans la perte, ADR-005 D3) ; `total_iterations`
(dont dépendent les calendriers LR et entropie de `PPOConfig`) est dérivé : ⌈total_timesteps / batch_learned_steps⌉.
Sans pool d'adversaires, appris = collectés (n_envs · n_cars · steps_per_batch par lot). Avec le pool (section `pool`,
ADR-005 D5), les agents gelés des arènes « passé » ne comptent pas. L'annealing de `p_kick` est aussi en timesteps
appris (ADR-005 D4). Section `pool` absente (None) : aucune clé `pool` dans les hashs, qui restent ceux d'avant le pool.
Section `sil` (self-imitation, `training.sil.SILConfig`, ADR-005 D6 intervention (3)) : même règle, absente => hashs
inchangés ; présente, elle entre dans le hash de trajectoire (activée ou non). Activée, elle n'est acceptée que si la
variante l'autorise (spec §3 : interdite en GCO-S) et, sous C1–C4, si son coefficient atteint 0 (C2) avant la fin
du budget.
Section `aux` (tâche auxiliaire « prochain but », `training.auxtask.AuxConfig`, ADR-010, intervention (4)) : même règle
de hash (absente => hashs inchangés) ; activée, refusée si la variante ne l'autorise pas (spec §3 : interdite en GCO-S)
et si la SIL est aussi activée (ADR-010 D11). λ constant par défaut (ADR-010 D6) ; calendrier optionnel
`anneal_start_timesteps` / `anneal_end_timesteps`, qui doit alors finir avant la fin du budget (lecture littérale de
C2).
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
from training.auxtask import AuxConfig
from training.checkpoint import hash_config
from training.guards import GuardConfig
from training.ppo import PPOConfig
from training.selfplay import n_past_arenas
from training.sil import SILConfig

_DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d+)?|xpu(:\d+)?)$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# Sections qui ne changent pas la trajectoire d'entraînement : exclues du hash vérifié à la reprise.
OPERATIONAL_SECTIONS = ("checkpoint", "kill", "logging", "smoke")
P_KICK_SCHEDULE_VERSION = 2
"""Version de la FORMULE de `RunConfig.p_kick_at` (les paramètres, eux, sont dans le hash de trajectoire). 1 : avant
A1 (cible atteinte une itération trop tard) ; 2 : A1 (cible à l'indice n - 1) puis A3 (annealing en timesteps appris,
identique sans pool). Choix (revue A1 vers A4) : la version n'entre PAS dans `trajectory_hash`, ce qui changerait le
hash de toutes les configs existantes ; elle est écrite dans `extra["p_kick_schedule_version"]` des checkpoints et
`training.run` refuse une reprise d'une autre version. Un checkpoint sans cette clé (antérieur à A4) est accepté : sa
version ne peut pas être prouvée. Changer la formule de `p_kick_at` => incrémenter cette constante."""


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
class PoolSection:
    """Pool d'adversaires (ADR-005 D5 ; `docs/01_papers.md` §1.8, OpenAI Five App. N)."""

    past_fraction: float = 0.2  # D5 : 20 % des arènes contre un instantané passé (Five : 20 %)
    snapshot_every: int = 50  # D5 : instantané toutes les 50 itérations (≈ 2,6M appris à 4 096 arènes)
    max_size: int = 100  # D5 : pool plafonné, le plus mauvais q sort
    quality_lr: float = 0.01  # D5, Five : η = 0,01

    def __post_init__(self) -> None:
        if not 0.0 < self.past_fraction < 1.0:
            raise ValueError("past_fraction doit être dans ]0, 1[")
        if self.snapshot_every < 1 or self.max_size < 1:
            raise ValueError("snapshot_every >= 1 et max_size >= 1")
        if not self.quality_lr > 0.0:
            raise ValueError("quality_lr > 0")


@dataclass(frozen=True, kw_only=True)
class BudgetSection:
    total_timesteps: int  # timesteps agents APPRIS (transitions qui entrent dans la perte, ADR-005 D3)

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
    pool: PoolSection | None = None  # None : self-play contre la version courante seule (avant ADR-005)
    sil: SILConfig | None = None  # None : pas de self-imitation (hashs d'avant la SIL)
    aux: AuxConfig | None = None  # None : pas de tâche auxiliaire (hashs d'avant ADR-010)

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"name={self.name!r} : lettres, chiffres, '_', '.', '-' uniquement")
        if self.seed < 0:
            raise ValueError("seed >= 0")
        if not _DEVICE_RE.match(self.device):
            raise ValueError(f"device={self.device!r} : auto, cpu, cuda[:n] ou xpu[:n]")
        if self.pool is not None:
            learned_per_batch(self.env, self.collection, self.pool)  # ValueError si aucune arène « passé »
        if self.ppo.total_iterations != self.total_iterations:
            raise ValueError(
                f"ppo.total_iterations={self.ppo.total_iterations} != {self.total_iterations} "
                "(dérivé du budget : utiliser RunConfig.from_dict)"
            )
        if self.sil is not None and self.sil.enabled:
            rules = rules_for(RewardVariant(self.reward.variant))
            if not rules.self_imitation:
                raise ValueError(f"variante {self.reward.variant} : self-imitation interdite (spec §3)")
            end = self.sil.anneal_end_timesteps
            if rules.learned_signals_require_c1_c4 and end >= self.budget.total_timesteps:
                raise ValueError(
                    f"sil.anneal_end_timesteps={end} >= budget {self.budget.total_timesteps} : "
                    "C2 exige un coefficient nul avant la phase finale du run"
                )
        if self.aux is not None and self.aux.enabled:
            rules = rules_for(RewardVariant(self.reward.variant))
            if not rules.auxiliary_tasks:
                raise ValueError(f"variante {self.reward.variant} : tâche auxiliaire interdite (spec §3)")
            if self.sil is not None and self.sil.enabled:
                raise ValueError("sil et aux activées ensemble : refusé (ADR-010 D11, deux résultats isolés d'abord)")
            aux_end = self.aux.anneal_end_timesteps
            if aux_end is not None and aux_end >= self.budget.total_timesteps:
                raise ValueError(
                    f"aux.anneal_end_timesteps={aux_end} >= budget {self.budget.total_timesteps} : un calendrier de λ "
                    "doit finir avant la fin du run (sinon : omettre le calendrier, λ constant, ADR-010 D6)"
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
        """Transitions agents COLLECTÉES par lot (et par itération), agents gelés compris."""
        return self.n_agents * self.collection.steps_per_batch

    @property
    def n_past_arenas(self) -> int:
        """Arènes contre un instantané passé (0 sans pool)."""
        return 0 if self.pool is None else n_past_arenas(self.env.n_envs, self.pool.past_fraction)

    @property
    def batch_learned_steps(self) -> int:
        """Transitions agents APPRISES par lot : collectées moins celles du camp gelé des arènes « passé »."""
        return learned_per_batch(self.env, self.collection, self.pool)

    @property
    def total_iterations(self) -> int:
        return derive_total_iterations(self.env, self.collection, self.budget, self.pool)

    @property
    def anneal_iterations(self) -> int:
        """Itérations de l'annealing de `p_kick` : `anneal_timesteps` est en timesteps appris (ADR-005 D4)."""
        return math.ceil(self.initial_states.anneal_timesteps / self.batch_learned_steps)

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
        if out["pool"] is None:
            del out["pool"]  # hashs inchangés pour les configs d'avant le pool (checkpoints existants repris)
        if out["sil"] is None:
            del out["sil"]  # idem pour les configs d'avant la SIL
        if out["aux"] is None:
            del out["aux"]  # idem pour les configs d'avant la tâche auxiliaire (ADR-010 D8-2)
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
        pool = None if raw.get("pool") is None else _build(PoolSection, raw["pool"], "pool")
        try:
            total = derive_total_iterations(env, collection, budget, pool)
        except ValueError as exc:
            raise ConfigError(f"pool : {exc}") from exc
        raw["ppo"] = {**ppo_raw, "total_iterations": total}
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


def learned_per_batch(env: EnvSection, collection: CollectionSection, pool: PoolSection | None = None) -> int:
    """Transitions apprises par lot : (agents − agents gelés) · steps_per_batch ; un camp gelé = n_cars/2 agents."""
    n_past = 0 if pool is None else n_past_arenas(env.n_envs, pool.past_fraction)
    if pool is not None and not 1 <= n_past < env.n_envs:
        raise ValueError(
            f"pool.past_fraction={pool.past_fraction} sur {env.n_envs} arènes : {n_past} arène(s) « passé », "
            "il en faut au moins 1 et au plus n_envs - 1"
        )
    return (env.n_envs * env.n_cars - n_past * (env.n_cars // 2)) * collection.steps_per_batch


def derive_total_iterations(
    env: EnvSection, collection: CollectionSection, budget: BudgetSection, pool: PoolSection | None = None
) -> int:
    return math.ceil(budget.total_timesteps / learned_per_batch(env, collection, pool))


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
