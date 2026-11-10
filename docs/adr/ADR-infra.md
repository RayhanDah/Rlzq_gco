# ADR-001 — Infrastructure d'entraînement (simulateur, collecte, plateforme)

- Statut : **proposé** (brouillon Phase 2, mesures partielles ; la décision finale attend la sonde Kaggle)
- Date : 2026-11-10
- Décideurs : tech-director + infra-perf-lead ; GO de l'utilisateur au Gate G2

## Contexte
- PPO et rewards en Python (règle 7) ; la collecte (RocketSim) est CPU-bound, l'apprentissage d'un MLP 3×256 est léger.
- Plateformes : laptop (Core Ultra 5 125H, 18 threads logiques, pas de CUDA) et Kaggle, compte par membre du groupe
  (`docs/research/kaggle.md` §5–6). TPU v5e-8 : file ≥ 35 h, ≈ 2,5B steps/session rapportés par l'utilisateur (non mesuré).
- RLGym 2.0.1 (PyPI) impose numpy < 2 ; `main` (2fc8d43) accepte numpy < 3 mais n'est pas publié et se construit en deux
  temps (exécution de scripts `setup_*.py` tiers). Le build local depuis git a été refusé par le classifieur de sécurité de
  la session le 2026-11-10 ; en attente d'une décision de l'utilisateur.

## Options
| Option | Pour | Contre |
|---|---|---|
| A1. rlgym-learn + RLGym `main` + rocketsim 2.2.1 | collecte parallèle en Rust, PPO Python fourni, API RLGym v2 | build de `main` à faire ; rlgym-learn 2.0.0 instable (breaking changes) |
| A2. Env maison minimal sur `rocketsim` 2.2.1 (PyPI) + PPO maison torch | aucune dépendance non publiée ; contrôle total (obs, reset, GCO) | réécrire obs/actions/parité RLGym ; collecte multi-process à écrire |
| B. A1/A2 + chemins chauds natifs (obs, collecte) | débit | après mesure seulement (règle 7) |
| C. GigaLearn (C++) | débit élevé | PPO et rewards en C++ : **exclu** par la règle 7 ; plafond de référence au mieux |

## Mesures
### M1 — Débit brut RocketSim, laptop (plafond de collecte, sans obs ni politique)
Commande : `.venv/Scripts/python scripts/benchmark_sim.py --procs 1,2,4,6,8,12,14,18 --seconds 20` (seed 0, 4 arènes 1v1
par processus, tick-skip 8, actions aléatoires). Matériel : Zenbook 14, Core Ultra 5 125H, Windows 11, Python 3.12.6,
numpy 2.5.4, rocketsim 2.2.1. Commit : **54a10a6** (arbre propre), 2026-11-10 12:36 CET. Meshes : RLGym @2fc8d43
(`scripts/fetch_meshes.py`, hash vérifiés). Rapport brut : `runs/bench/bench_sim_20261110_123933.json` (local).

| Processus | env-steps/s | agent-steps/s (×2) | ticks/s | Gain vs 1 proc |
|---:|---:|---:|---:|---:|
| 1 | 21 257 | 42 514 | 170 054 | 1,0× |
| 2 | 28 695 | 57 390 | 229 561 | 1,35× |
| 4 | 38 137 | 76 275 | 305 099 | 1,79× |
| 6 | 45 748 | 91 496 | 365 985 | 2,15× |
| 8 | 52 322 | 104 643 | 418 573 | 2,46× |
| 12 | 56 510 | 113 019 | 452 076 | 2,66× |
| 14 | 71 869 | 143 739 | 574 955 | 3,38× |
| 18 | 77 610 | 155 220 | 620 878 | 3,65× (sursouscription : 18 = threads logiques) |

Lecture :
- Un passage antérieur (commit 12dad2d-dirty, sans barrière de départ) donnait des valeurs proches : 21,7k à 1 proc,
  65,2k à 14 procs, 66,3k à 18 procs. L'écart à 14–18 procs (≈ 10–15 %) donne l'ordre de grandeur de la variance.
- Scaling très sous-linéaire (3,4× pour 14 procs) : 4 P-cores rapides + 8 E-cores + 2 LP E-cores, turbo qui baisse avec la
  charge. Interprétation, non profilée.
- C'est un **plafond** : pas d'obs, pas de politique, pas de PPO. Le débit bout-en-bout sera nettement plus bas (M3).
- Durée 20 s par point : **ne mesure pas le throttling** (un run ≥ 30 min est prévu).

### M2 — Sonde Kaggle T4 / TPU
_À faire_ : `python scripts/kaggle_probe.py --user <pseudo> --acc t4,tpu --push` (jeton de l'utilisateur requis).
Build à blanc vérifié sur 54a10a6 (`--user test-user`, sans `--push`) : métadonnées T4 et TPU générées, kernel compilé.
Pousser T4 puis TPU à la suite teste aussi la concurrence GPU + TPU sur un compte.

### M3 — Bout-en-bout (obs + politique + PPO)
_À faire_ après le choix A1/A2.

## Temps de calcul
_Estimation à produire à partir de M1–M3 et du budget de `docs/GCO_SPEC.md` §6._

## Décision
_En attente des mesures M2–M3._

## Conséquences
_À compléter._
