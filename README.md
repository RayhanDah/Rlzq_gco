# RL-GCO

Agent PPO pour Rocket League entraîné sous récompense **Goal/Concede Only** (GCO) : +1 au but marqué, −1 au but
encaissé, rien d'autre. Objectif final : un bot jouable dans RLBot v5 contre humains et bots.

Le GCO naïf explore très mal (la plupart des actions ne touchent jamais la balle). Le projet étudie ce qui le fait
marcher sans ajouter de récompense : signaux d'exploration appris, self-play, réglages PPO. Spécification complète :
[`docs/GCO_SPEC.md`](docs/GCO_SPEC.md).

## Principes
- **GCO strict** dans la ligne principale ; toute récompense additionnelle reste une ablation étiquetée.
- **Mesures reproductibles** : chaque chiffre est lié à une commande, une seed, un commit et un matériel, consignés dans
  un ADR ou un rapport.
- **Décisions tracées** dans [`docs/adr/`](docs/adr/), résultats dans [`docs/reports/`](docs/reports/).

## État (novembre 2026)
Phase 4a, expériences d'exploration sur 1 milliard de timesteps par run :
- 3 runs GCO de référence : l'agent bat nettement l'aléatoire, mais touche rarement la balle
  ([`run_4a.md`](docs/reports/run_4a.md)) ;
- interventions testées une à une, chacune sur une seed : γ 0,998, SIL, LR ×3, témoin dense hors spécification
  (rapports dans `docs/reports/`) ;
- suite : tête auxiliaire « prochain but » (ADR-010), puis multi-seeds et préparation du premier run long.

## Architecture
| Dossier | Contenu |
|---|---|
| `src/env/` | environnement batché (RocketGPU en CPU, RocketSim pour la parité) |
| `src/obs/`, `src/actions/` | observations, table de 90 actions, symétrie miroir |
| `src/rewards/` | récompense GCO et variantes d'ablation |
| `src/models/`, `src/training/` | politique, valeur, learner PPO (PyTorch) |
| `src/eval/` | matchs, Elo/SPRT, IC |
| `src/rlbot_bot/` | intégration RLBot v5 (phase 5) |
| `scripts/` | entraînement, éval, bancs, Kaggle |
| `configs/` | configurations de runs versionnées |

## Installation
Python 3.12, [`uv`](https://docs.astral.sh/uv/) :

```bash
uv sync
uv run python scripts/fetch_meshes.py      # meshes de collision (non versionnés)
uv run python scripts/build_rocketgpu.py   # compile le simulateur
uv run pytest -q tests/unit
uv run python scripts/train.py configs/smoke.yaml
```

Matériel principal : laptop Intel Core Ultra
(learner PyTorch XPU), Kaggle T4 en appoint ([`docs/adr/001-infra.md`](docs/adr/001-infra.md)).

## Crédits
Simulation : [RocketSim](https://github.com/ZealanL/RocketSim), RocketGPU. Déploiement : [RLBot](https://rlbot.org).
