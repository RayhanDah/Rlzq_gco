"""Configuration de la récompense GCO : variantes (spec §3), limites d'épisode (§2.2), hash stable.

Référence : `docs/GCO_SPEC.md` v0.2 et `docs/adr/002-reward-gco.md`. Aucune dépendance au simulateur.

Choix d'ambiguïté (le plus strict, détaillés dans l'ADR-002) :
- A1. La récompense §2.1 est identique pour GCO-S, GCO-std et GCO-ext : la variante ne change que les règles déclarées
  (`VariantRules`, transcription du tableau §3), que d'autres modules consultent.
- A2. « scorer-only » hérite des règles de GCO-S (le niveau le plus strict) ; seule l'attribution du +1 change.
- A3. En GCO-ext, les signaux appris restent soumis à C1–C4 : extension stricte, la spec ne lie C1–C4 qu'à GCO-std
  (le tableau §3 dit seulement « autorisés » pour GCO-ext).
- A4. Les durées sont exprimées en secondes de jeu et doivent tomber sur un tick physique entier (120 Hz).
- A5. Le hash couvre la config, les règles dérivées, la version de la spec et la version du schéma de hachage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import Any

SPEC_VERSION = "0.2"
"""Version de `docs/GCO_SPEC.md` implémentée."""

HASH_SCHEMA_VERSION = 1
"""À incrémenter à chaque changement de sémantique du calcul qui ne modifie aucun champ de config (spec §7.8)."""

PHYSICS_TICK_RATE_HZ = 120
"""Ticks physiques par seconde de jeu dans Rocket League (constante du jeu, pas du simulateur)."""


class RewardVariant(StrEnum):
    """Niveaux de définition de la spec §3, plus l'ablation hors spec « scorer-only »."""

    GCO_S = "GCO-S"
    GCO_STD = "GCO-std"
    GCO_EXT = "GCO-ext"
    SCORER_ONLY = "scorer-only"


@dataclass(frozen=True, slots=True)
class VariantRules:
    """Ce qu'une variante autorise, transcrit du tableau §3 (lignes « masquage d'actions » et « entropie » omises :
    autorisées partout). Ces drapeaux sont déclaratifs : les modules env/training doivent les vérifier (C3)."""

    in_spec: bool
    """Faux pour « scorer-only » (§2.1 : hors spec)."""
    ablation_only: bool
    """Vrai si la variante ne peut jamais être la ligne principale (GCO-ext : décision G1 H3 ; scorer-only)."""
    team_attribution: bool
    """§2.1 : tous les coéquipiers reçoivent le même signal. Faux seulement pour scorer-only."""
    generic_initial_states: bool
    targeted_initial_states: bool
    """États ciblés, rejeu de replays, learned state setters."""
    external_opponents: bool
    learned_advantage_signals: bool
    """PBRS sur potentiel appris, composition/goal critic."""
    intrinsic_bonuses: bool
    """RND, world model, nouveauté."""
    learned_signals_require_c1_c4: bool
    self_imitation: bool
    auxiliary_tasks: bool
    demonstrations: bool
    """Démos, BC, KL vers un bot externe."""


_GCO_S_RULES = VariantRules(
    in_spec=True,
    ablation_only=False,
    team_attribution=True,
    generic_initial_states=False,
    targeted_initial_states=False,
    external_opponents=False,
    learned_advantage_signals=False,
    intrinsic_bonuses=False,
    learned_signals_require_c1_c4=True,
    self_imitation=False,
    auxiliary_tasks=False,
    demonstrations=False,
)

_RULES: dict[RewardVariant, VariantRules] = {
    RewardVariant.GCO_S: _GCO_S_RULES,
    RewardVariant.GCO_STD: VariantRules(
        in_spec=True,
        ablation_only=False,
        team_attribution=True,
        generic_initial_states=True,
        targeted_initial_states=False,
        external_opponents=False,
        learned_advantage_signals=True,
        intrinsic_bonuses=True,
        learned_signals_require_c1_c4=True,
        self_imitation=True,
        auxiliary_tasks=True,
        demonstrations=False,
    ),
    RewardVariant.GCO_EXT: VariantRules(
        in_spec=True,
        ablation_only=True,
        team_attribution=True,
        generic_initial_states=True,
        targeted_initial_states=True,
        external_opponents=True,
        learned_advantage_signals=True,
        intrinsic_bonuses=True,
        learned_signals_require_c1_c4=True,  # choix A3 (strict)
        self_imitation=True,
        auxiliary_tasks=True,
        demonstrations=True,
    ),
    # Choix A2 (strict) : règles de GCO-S, seule l'attribution change.
    RewardVariant.SCORER_ONLY: VariantRules(
        **{**asdict(_GCO_S_RULES), "in_spec": False, "ablation_only": True, "team_attribution": False}
    ),
}


def rules_for(variant: RewardVariant) -> VariantRules:
    """Règles déclarées de `variant` (tableau §3)."""
    return _RULES[variant]


def _seconds_to_ticks(name: str, seconds: float) -> int:
    ticks = seconds * PHYSICS_TICK_RATE_HZ
    rounded = round(ticks)
    # Tolérance 1e-9 tick : absorbe l'erreur d'arrondi binaire de valeurs décimales comme 20.1 s.
    if not math.isclose(ticks, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name}={seconds} s ne tombe pas sur un tick physique entier ({PHYSICS_TICK_RATE_HZ} Hz)")
    return int(rounded)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Config de la récompense et des fins d'épisode (§2.1, §2.2). Valeurs par défaut = spec v0.2."""

    variant: RewardVariant = RewardVariant.GCO_STD
    no_touch_timeout_s: float = 20.0
    """Troncature si aucune touche de balle depuis cette durée de jeu (§2.2 ; ablation 20/45 s prévue)."""
    max_episode_s: float = 120.0
    """Troncature à cette durée d'épisode (§2.2)."""

    def __post_init__(self) -> None:
        if not isinstance(self.variant, RewardVariant):
            raise TypeError(f"variant doit être un RewardVariant, reçu {type(self.variant).__name__}")
        for name in ("no_touch_timeout_s", "max_episode_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} doit être un nombre, reçu {type(value).__name__}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} doit être fini et > 0, reçu {value}")
            # Normalise 20 → 20.0 pour que le hash ne dépende pas de l'écriture de la config.
            object.__setattr__(self, name, float(value))
            _seconds_to_ticks(name, float(value))

    @property
    def rules(self) -> VariantRules:
        return rules_for(self.variant)

    @property
    def no_touch_ticks(self) -> int:
        return _seconds_to_ticks("no_touch_timeout_s", self.no_touch_timeout_s)

    @property
    def max_episode_ticks(self) -> int:
        return _seconds_to_ticks("max_episode_s", self.max_episode_s)

    def to_dict(self) -> dict[str, Any]:
        """Forme sérialisable (JSON) de la config, telle qu'écrite dans une config de run ou un checkpoint."""
        return {
            "variant": self.variant.value,
            "no_touch_timeout_s": self.no_touch_timeout_s,
            "max_episode_s": self.max_episode_s,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RewardConfig:
        """Construit une config depuis un dict (YAML/JSON). Toute clé inconnue est refusée (pas de réglage muet)."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"clés de config de récompense inconnues : {sorted(unknown)}")
        kwargs = dict(data)
        if "variant" in kwargs:
            kwargs["variant"] = RewardVariant(kwargs["variant"])  # ValueError si nom inconnu
        return cls(**kwargs)

    def hash_payload(self) -> dict[str, Any]:
        """Tout ce qui détermine le signal : config, règles dérivées, versions de spec et de schéma, tick rate."""
        return {
            "hash_schema_version": HASH_SCHEMA_VERSION,
            "spec_version": SPEC_VERSION,
            "physics_tick_rate_hz": PHYSICS_TICK_RATE_HZ,
            "config": self.to_dict(),
            "rules": asdict(self.rules),
        }

    def config_hash(self) -> str:
        """SHA-256 hexadécimal du JSON canonique de `hash_payload()` (clés triées, sans espaces, ASCII).

        Stable entre processus, machines et versions de Python ; à écrire dans chaque checkpoint (§7.7)."""
        canonical = json.dumps(self.hash_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()
