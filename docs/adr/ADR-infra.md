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
| **A3. RocketGPU** (`thecosmicskye/RocketGPU`, MIT) : sim batchée CPU ou CUDA, API numpy (contrôles et états en tableaux), paquet « RocketSim-style » | revendique ≈ 4,2M arena-ticks/s CPU (9800X3D, 14 threads) et 12,4M en CUDA (RTX 5090), 100 % bit-exact vs jeu réel ; **le mode CUDA rendrait les T4 utiles** ; pas d'objets Python par voiture | tiers, créé le 2026-10-27, 12 ★, un seul auteur, aucun wheel : build CMake + compilateur C++17 (code tiers à exécuter) ; chiffres **non reproduits** |
| A4. RocketSim v3 (Rust) + bindings PyO3 maison | officiel, MIT | pas de binding Python ; toolchain Rust absente ; moins rapide et moins exact que A3 d'après le tableau de A3 (tiers) |
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

### M1b — RocketSim 2.2.1 en un seul processus avec `Arena.multi_step` (threads C++)
Commande : `.venv/Scripts/python scripts/benchmark_sim.py --procs 1 --arenas <A> --engine multi_step --seconds 20`
(et `--procs 2,4 --arenas 64`). Commit **5e1bf19**, même machine, 2026-11-10, seed 0, actions aléatoires posées en Python.

| Processus × arènes | env-steps/s | ticks/s |
|---:|---:|---:|
| 1 × 16 | 39 283 | 314 260 |
| 1 × 64 | 57 599 | 460 794 |
| 1 × 256 | 66 146 | 529 171 |
| 2 × 64 | 77 752 | 622 015 |
| 4 × 64 | 74 726 | 597 806 |

Lecture :
- `multi_step` est multithreadé : CPU/wall ≈ 7,2 à 16 arènes (mesure ponctuelle `time.process_time`, script jetable,
  non versionné).
- Le plafond du laptop avec RocketSim v2 est ≈ 75k env-steps/s (≈ 600k ticks/s), atteint par 14 processus (M1) ou par
  2 processus `multi_step`. Un seul processus en obtient ≈ 85 %, sans IPC : architecture de collecte retenue a priori
  (1 processus, N arènes, inférence torch batchée dans le même processus).
- Sur ce chemin, la pose des contrôles en Python (une affectation par voiture) devient le goulot à 256 arènes
  (interprétation : CPU/wall ≈ 9 sur 18 threads). Une API par tableaux (A3) le supprimerait.
- Comparaison avec les chiffres de A3 **impossible en l'état** : autre CPU (9800X3D), autre charge (mix 1v1/2v2/3v3,
  contrôles rejoués). À re-mesurer sur ce laptop, même charge, avant toute décision.

### M1c — RocketGPU CPU, build local (autorisé par l'utilisateur le 2026-11-10)
Source : `github.com/thecosmicskye/RocketGPU` @ **f0508a1** (MIT, dérivé non officiel de RocketSim), cloné dans
`vendor/src/` (non versionné). Build inspecté avant exécution : aucun téléchargement ; `execute_process` limité à
`nvidia-smi`, `ispc --help`, `python -m pybind11`.
Toolchain : MinGW-w64 gcc 13.2.0, CMake 4.4.4, Ninja 1.13.2, pybind11 3.1.0 (groupe uv `native`), `-march=native`
(défaut du projet), sans ISPC (non installé). Compilation : 1 min 37 s, avertissements seulement.
```
cmake -S vendor/src/RocketGPU -B vendor/build/rocketgpu-gcc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \
      -DROCKETGPU_BUILD_PYTHON=ON -DROCKETGPU_BUILD_BENCH=ON -DPython_EXECUTABLE=.venv/Scripts/python.exe
cmake --build vendor/build/rocketgpu-gcc -j 12
```
PGO (gcc) : build `-DROCKETGPU_PGO=GENERATE`, profil = notre propre charge (`benchmark_sim.py --engine rocketgpu
--arenas 256 --seconds 30`), puis build `-DROCKETGPU_PGO=USE`. Avertissements gcc « does not start with profile prefix »
(séparateurs `\` vs `/` sous Windows) : le profil n'est **peut-être appliqué qu'en partie**.

Cohérence physique (script jetable, 2026-11-10) : voiture seule, plein gaz, depuis un kickoff, |v| RocketSim 2.2.1 vs
RocketGPU : 346,2 vs 354,0 (30 ticks), 623,6 vs 627,4 (60), 1002,1 vs 1001,6 (120) ; avec boost 1617,4 vs 1614,1 à
120 ticks ; hauteur au repos 17,01 dans les deux. Écart ≤ 2,2 % à 30 ticks, ≤ 0,3 % à 120 ticks (spots de kickoff
différents). Ce n'est **pas** un test de parité ; la parité avec le jeu reste à mesurer en Phase 5.

Débit, même charge que M1/M1b (1v1, contrôles aléatoires tirés en numpy pour toutes les voitures, tenus 8 ticks, reset au
but ou à 300 s), 1 processus, threads internes par défaut. Commande :
`.venv/Scripts/python scripts/benchmark_sim.py --procs 1 --arenas <A> --engine rocketgpu --seconds 20`
(`ROCKETGPU_PY_DIR=vendor/build/rg-pgo/python` pour le build PGO). Commit **f1b9d7e**, même machine, seed 0.

| Arènes | Build simple : env-steps/s | Build PGO : env-steps/s | PGO : ticks/s |
|---:|---:|---:|---:|
| 64 | 159 031 | — | — |
| 256 | 159 741 | 181 118 | 1 448 946 |
| 1 024 | 161 254 | 193 430 | 1 547 439 |
| 4 096 | 173 753 | 198 394 | 1 587 156 |

Lecture :
- RocketGPU PGO ≈ **198k env-steps/s**, soit **≈ 2,5×** le meilleur RocketSim 2.2.1 mesuré (77,8k, M1b) sur ce laptop,
  et ≈ 2,2× sans PGO. Le gain annoncé par le README (≈ 6× vs nos chiffres) ne se reproduit pas : autre CPU, autre charge.
- Le débit est déjà élevé à 64 arènes : le batch suffit à occuper les threads, et la pose des contrôles en numpy ne coûte
  plus rien en Python par voiture.
- Une mesure par point, 20 s : variance non estimée, throttling non mesuré.

### M2 — Sonde Kaggle T4 / TPU
_À faire_ : `python scripts/kaggle_probe.py --user <pseudo> --acc t4,tpu --push` (jeton de l'utilisateur requis).
Build à blanc vérifié sur 54a10a6 (`--user test-user`, sans `--push`) : métadonnées T4 et TPU générées, kernel compilé.
Pousser T4 puis TPU à la suite teste aussi la concurrence GPU + TPU sur un compte.

### M3 — Bout-en-bout (obs + politique + PPO)
_À faire_ après le choix A1/A2.

## Temps de calcul
_Estimation à produire à partir de M1–M3 et du budget de `docs/GCO_SPEC.md` §6._

## Décision
_En attente des mesures M2–M3._ Orientation proposée après M1c (non actée) : **RocketGPU comme simulateur principal**
(CPU sur laptop et hôte TPU, CUDA à évaluer sur T4), commit épinglé, build scripté ; **RocketSim 2.2.1 conservé comme
référence** (comparaisons de physique, repli si RocketGPU casse).

## Conséquences
_À compléter._
