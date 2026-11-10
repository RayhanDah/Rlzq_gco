# ADR-002 — Récompense GCO : interface, variantes, choix d'ambiguïté

- Statut : proposé (Phase 3 anticipée ; merge sur `main` seulement après le GO G2)
- Date : 2026-11-10
- Décideurs : reward-engineer (session de travail), revue code-reviewer ; à valider par rl-algo-lead et research-director

## Contexte
La spec `docs/GCO_SPEC.md` v0.2 (validée au G1) fixe le signal (§2.1), les fins d'épisode (§2.2), trois niveaux de
définition (§3) et les tests exigés (§7). Le §7.8 demande un ADR pour toute implémentation ou modification de
`src/rewards/`. Contraintes propres à ce lot :
- le simulateur n'est pas tranché (RocketGPU batché pressenti, RocketSim 2.2.1 en référence, `ADR-infra.md`) : la
  récompense ne doit rien savoir du simulateur ;
- la collecte sera batchée sur des milliers d'arènes : le calcul doit être vectorisé (numpy) ;
- la règle 7 de `CLAUDE.md` impose une récompense en Python.

## Options
| Option | Pour | Contre |
|---|---|---|
| A. Fonction pure sur un événement de but (filet touché + équipes), vectorisée numpy, variantes et règles déclarées par config | indépendante du simulateur ; aucune fuite possible (signature fermée) ; un seul chemin batché | l'adaptateur simulateur → `GoalNet` reste à écrire dans `src/env/` |
| B. Classe de récompense qui lit l'état du simulateur (style `RewardFunction` de RLGym) | intégration directe | couple la récompense au simulateur non tranché ; l'état complet est accessible, la fuite n'est exclue que par revue |
| C. Récompense en natif dans le simulateur (RocketGPU) | aucun coût Python | exclu par la règle 7 |

## Mesures
Aucune mesure de performance : le coût du calcul n'a pas été mesuré, décision réversible sur la base de la règle 7 et de
la vectorisation numpy (« estimé, non mesuré » : négligeable devant le pas de simulation). Preuves de correction :
`.venv/Scripts/python -m pytest -q tests/unit/test_rewards_gco.py` (52 tests, seed 20261110), Windows 11, Core Ultra 5
125H, Python 3.12.6, numpy 2.5.4, commit de ce lot sur la branche `claude/phase3-gco-reward-fd3080`, 2026-11-10.

## Décision
Option A. Paquet `src/rewards/` (importé comme `rewards`, layout `src/` ; `pythonpath = ["src"]` ajouté à la config
pytest).

### Interface
- Entrées : `goal_net` (E,) entier, `GoalNet.NONE = -1`, `BLUE = 0` (balle dans le but bleu, orange marque),
  `ORANGE = 1` ; `agent_team` (A,) ou (E, A), `Team.BLUE = 0`, `Team.ORANGE = 1` ; pour les drapeaux seulement,
  `ticks_since_touch` et `episode_ticks` (E,) en ticks physiques à 120 Hz.
- `gco_team_rewards(goal_net, agent_team)` → (E, A) float32 : +1 à l'équipe qui marque, −1 à celle qui encaisse, 0 sinon.
  Le signe vient du filet touché ; le dernier toucheur n'est jamais une entrée (csc inclus).
- `episode_end(goal_net, ticks_since_touch, episode_ticks, config)` → `(terminated, truncated)` (E,).
- `RewardFunction` (via `make_reward(config | dict)`) : `rewards`, `episode_end`, `step` → `StepOutcome`,
  `rewards_one` (une arène, même chemin vectorisé), `config_hash`.
- `RewardConfig(variant, no_touch_timeout_s=20.0, max_episode_s=120.0)`, `from_dict` (clés inconnues refusées),
  `to_dict`, `config_hash()` : SHA-256 du JSON canonique (clés triées) de la config, des règles dérivées, de
  `SPEC_VERSION = "0.2"`, de `HASH_SCHEMA_VERSION = 1` et du tick rate. Ce hash est destiné à chaque checkpoint (§7.7).

### Variantes (`RewardVariant`, sélection par config)
| Variante | Signal | Règles déclarées (`VariantRules`) |
|---|---|---|
| `GCO-S` | §2.1 | tableau §3, colonne GCO-S |
| `GCO-std` (défaut) | §2.1 | colonne GCO-std |
| `GCO-ext` | §2.1 | colonne GCO-ext, `ablation_only` (G1 H3) |
| `scorer-only` | +1 au seul buteur désigné | **hors spec**, `ablation_only`, `in_spec = False` |

Les règles sont déclaratives : `src/env/` et `src/training/` devront les vérifier (C3). Ce lot ne les applique pas.

### Choix d'ambiguïté (le plus strict à chaque fois, aussi écrits dans le code)
| # | Point ambigu | Choix |
|---|---|---|
| A1 | La variante change-t-elle le signal ? | Non : §2.1 est identique pour S/std/ext (tableau §3, ligne « Récompense »). La variante ne porte que des règles. |
| A2 | Règles de scorer-only | Celles de GCO-S (niveau le plus strict) ; seules l'attribution et les étiquettes changent. |
| A3 | C1–C4 en GCO-ext (« autorisés » sans condition) | **Extension**, pas transcription : la spec ne lie C1–C4 qu'à GCO-std (§3) ; on les exige aussi en GCO-ext. Sans objet en GCO-S (signaux interdits). |
| A4 | Unité des durées | Secondes de jeu, converties en ticks à 120 Hz ; une durée qui ne tombe pas sur un tick entier est refusée. |
| A5 | Contenu du hash | Config + règles dérivées + version de spec + version du schéma + tick rate. `20` et `20.0` donnent le même hash. |
| A6 | But et troncature au même step | `terminated = True`, `truncated = False` (§7.4 : but ⇒ non tronqué). |
| A7 | Seuil de troncature | Choix d'interprétation (`>` serait défendable) : déclenchement dès que le seuil est atteint (`>=`), lecture de « aucune touche depuis 20 s ». Figé par un test aux bornes. |
| A8 | Équipes inégales | Refusées : la somme nulle §2.1 n'est définie que pour NvN. |
| A9 | Définition de scorer-only (absente de la spec) | Le −1 va à toute l'équipe qui encaisse ; le +1 au seul buteur désigné (`scorer`, entrée explicite fournie par l'appelant) s'il est dans l'équipe qui marque, sinon à personne (csc). Somme non nulle. En 1v1, identique à GCO sauf en cas de csc. |
| A10 | Buteur passé à une variante GCO | Erreur, pas d'argument ignoré : le signal GCO ne peut pas dépendre du toucheur. |
| A11 | Échelle (±1 vs ±150) | ±1 fixe, aucun paramètre d'échelle (§2.1). |

### Couverture des tests §7 (`tests/unit/test_rewards_gco.py`)
- §7.1 à §7.4, §7.6, §7.7 : couverts (dont propriété sur ≥ 1000 épisodes, seed 20261110, trois fins d'épisode).
- §7.5 : partie récompense couverte (miroir ⇒ récompenses opposées, réétiquetage ⇒ identiques). **Non couvert** :
  observations miroir identiques (lot `src/obs/`).
- §7.6 : garantie structurelle (signatures fermées) ; les seules autres entrées acceptées, les compteurs de ticks, sont
  vérifiées sans effet sur la récompense.
- §7.7 : hash stable (valeur figée, autre processus) et sensible à chaque champ, aux versions, au tick rate et aux règles
  dérivées. **Non couvert** : écriture du hash dans le checkpoint (lot checkpoints, mlops-lead).
- Lot `rlbot_bot` : `rewards_one` exige la composition complète des deux équipes (A8) ; un appelant qui ne passe que ses
  propres agents est refusé.
- §7.8 : processus (cet ADR) ; la spec n'est pas modifiée par ce lot.

## Conséquences
- Plus facile : brancher n'importe quel simulateur via un adaptateur `état → GoalNet` ; auditer la conformité (signature
  fermée, 0 terme hors §2.1) ; détecter un checkpoint entraîné sous une autre config (hash).
- Plus difficile : toute modification de la sémantique sans changement de champ exige d'incrémenter
  `HASH_SCHEMA_VERSION` (le test du hash figé le rappelle).
- `numpy` vient du groupe `sim` de `pyproject.toml`, pas des dépendances runtime : à régulariser quand les dépendances
  runtime seront fixées (ADR-infra).
- Révision si : la spec définit scorer-only autrement (A9), si le simulateur retenu ne fournit pas le filet touché, ou si
  le 2v2/3v3 asymétrique (démo, joueur absent) doit être pris en charge (A8).
