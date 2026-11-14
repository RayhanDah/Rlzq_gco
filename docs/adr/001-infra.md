# ADR-001 — Infrastructure d'entraînement (simulateur, collecte, plateforme)

- Statut : **accepté** au Gate G2 (GO de l'utilisateur le 2026-11-11 ; sonde TPU encore en file : partie de la décision réversible, voir M2c)
- Date : 2026-11-10, mis à jour le 2026-11-11
- Décideurs : tech-director + infra-perf-lead ; GO de l'utilisateur au Gate G2

## Contexte
- PPO et rewards en Python (règle 7) ; la collecte (RocketSim) est CPU-bound, l'apprentissage d'un MLP 3×256 est léger.
- Plateformes : laptop (Core Ultra 5 125H, 18 threads logiques, pas de CUDA) et Kaggle, compte par membre du groupe
  (`docs/research/kaggle.md` §5–6). TPU v5e-8 : file ≥ 35 h, ≈ 2,5B steps/session rapportés par l'utilisateur (non mesuré).
- RLGym 2.0.1 (PyPI) impose numpy < 2 ; `main` (2fc8d43) accepte numpy < 3 mais n'est pas publié et se construit en deux
  temps (exécution de scripts `setup_*.py` tiers). Le build local depuis git a été refusé par le classifieur de sécurité de
  la session le 2026-11-10. L'option A1 a ensuite perdu son intérêt : A3 (RocketGPU) l'emporte en débit (M1c, M3) et ne
  dépend pas de RLGym. Seuls les meshes de RLGym sont réutilisés (fichiers de données, `scripts/fetch_meshes.py`).

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

### M1d — Variance et effet du PGO (builds scriptés)
Builds produits par `scripts/build_rocketgpu.py` (sans option → `rocketgpu-gcc` manuel équivalent ; `--pgo` →
`rocketgpu-cpu-pgo`, profil de 30 s sur notre charge, durée totale ≈ 20 min dont l'essentiel en compilation instrumentée).
Série alternée, 3 paires, 4 096 arènes, 15 s chacune, commit **2a49494**, 2026-11-10 ≈ 14:30, laptop chaud (sorti de 20 min
de compilation) :

| Paire | Build simple : env-steps/s | Build PGO : env-steps/s |
|---:|---:|---:|
| 1 | 128 849 | 183 088 |
| 2 | 169 994 | 174 282 |
| 3 | 127 183 | 156 202 |
| moyenne | 142 009 | 171 191 |

Un passage alterné antérieur (même minute, arbre non commité, chemin CPU fonctionnellement identique) donnait
193,8k / 160,1k (PGO manuel), 206,4k / 196,1k (PGO scripté) et 217,6k (simple).
Lecture :
- **Variance forte** : ≈ 127k à 218k env-steps/s pour un même binaire selon l'état thermique et le turbo du laptop.
  Toute comparaison demande des séries alternées ; un point isolé ne vaut rien.
- PGO devant le build simple sur les 3 paires du commit propre (+20 % en moyenne), mais derrière une fois dans le
  passage antérieur : **gain du PGO probable, non démontré** (n = 3).
- Ordre de grandeur RocketGPU CPU sur ce laptop : **≈ 130–215k env-steps/s**, contre ≈ 65–78k pour RocketSim (M1, M1b,
  mêmes réserves de variance).

### M2a — Sonde Colab, session GPU T4 (exécutée par l'utilisateur)
Fichier `build/colab/rlgco_probe.py` du commit **c110603**, lancé dans Colab le 2026-11-10 à 15:02 UTC
(`!python rlgco_probe.py`), durée 283 s. Résultats rapatriés (`probe.json` + `bench/`), non versionnés.

Machine relevée : 2 vCPU (Intel Xeon @ 2,00 GHz, 1 cœur physique × 2 threads), 12 Go de RAM, Tesla T4 15 Go
(compute capability 7.5), nvcc CUDA 13.0, g++ 13.3, Python 3.13.16, torch 2.14.1+cu130, jax 0.11.1 (CudaDevice).

| Étape | Résultat |
|---|---|
| RocketSim 2.2.1, step par arène, 1 / 2 / 4 processus | 14 607 / 17 553 / 16 973 env-steps/s |
| RocketSim `multi_step`, 1 processus × 256 arènes | 2 180 env-steps/s (threads C++ en sursouscription sur 2 vCPU) |
| Build RocketGPU CPU (`-march=native`) | réussi, 140 s |
| RocketGPU CPU, 1 024 / 4 096 arènes | 26 413 / 29 189 env-steps/s |
| Build RocketGPU **CUDA** | **échec** après 28 s ; l'erreur du compilateur a été tronquée par le journal |

Lecture :
- Avec 2 vCPU, une session Colab T4 ne vaut que par son GPU : la collecte CPU y plafonne vers 29k env-steps/s.
- L'échec CUDA reste à diagnostiquer. Depuis 3e41d8a, la sonde garde le journal complet de chaque étape
  (`<WORK>/logs/`) et les lignes d'erreur dans `probe.json`. Relance prévue : `--parts rg_cuda`.

### M2b — Sonde Kaggle T4 ×2 (2026-11-11, commit a74ac7f)
Détail : `docs/research/kaggle_probe.md`. 4 vCPU (Xeon 2,0 GHz, 2 cœurs × 2 threads), 31 Go, 2 × T4, CUDA 13.0,
Python 3.13.
- RocketSim : 36,5k env-steps/s au mieux (4 processus) ; RocketGPU CPU : 55,3k env-steps/s (4 096 arènes).
- Build RocketGPU CUDA : **échec** dans `cicc` (vérificateur LLVM de NVVM, CUDA 13.0, sm_75), reproduit sur Colab.
- Version 2 (b4b0481) avec CUDA 12.9 installé par apt : **même échec**. Le défaut n'est pas propre à CUDA 13.0.
- Compilation par architecture en session CPU (c8f865c) : l'objet CUDA ne compile que pour **sm_120** (585 s). Il
  échoue pour sm_75, 80, 86, 89 et 90. Aucun GPU Kaggle ou Colab ne peut donc exécuter RocketGPU CUDA @f0508a1 ;
  RocketGPU y tourne en CPU.
- Conséquence provisoire : en session T4, la simulation tourne sur CPU (≈ 55k env-steps/s, environ 3 fois moins que
  le laptop) tant que le build CUDA n'est pas réparé. Le GPU T4 reste utilisable pour le learner (torch cu130).

### M2c — Sonde Kaggle TPU v5e-8 (en file)
`python scripts/kaggle_probe.py --user rayhanda --acc tpu --push`, commit a74ac7f, poussée le 2026-11-11 vers 11:05 UTC
pendant que la sonde T4 tournait : acceptée, statut QUEUED, toujours en file à la rédaction. Elle relèvera les cœurs de
l'hôte TPU (la spec GCP annonce 224 vCPU, sans preuve que Kaggle les expose) et les débits RocketSim et RocketGPU CPU.
Aucun chiffre TPU n'entre dans la décision ci-dessous ; seule l'affectation du TPU en dépend.

### M3 — Learner seul et bout-en-bout, laptop (CPU vs GPU intégré Arc en XPU)
Commit **c110603**, arbre propre, 2026-11-10 ≈ 15:30, secteur, seed 0. Deux sessions Claude parallèles étaient au repos
(revues en lecture seule possibles). torch 2.15.0+cpu (`.venv`, 14 threads) et torch 2.15.0+xpu (`vendor/venv-xpu`,
Intel Arc Graphics, pilote 32.0.101.8860). Réseaux PPO v0 : π 3×256 → 90 logits, V 3×512, Adam, grad clip 0,5.
Contournement XPU : `x[idx]` lève `UR_RESULT_ERROR_OUT_OF_RESOURCES`, d'où `index_select` (identique sur CPU).

**Learner seul** (`scripts/bench_learner.py --device <d> --batch 50000 --minibatch 25000 --epochs 2`, obs 107 dims,
données synthétiques) :

| | CPU | XPU | Ratio |
|---|---:|---:|---:|
| Inférence, lot 1 024 (obs/s) | 81 015 | 184 263 | 2,3× |
| Inférence, lot 8 192 (obs/s) | 87 182 | 1 802 107 | 20,7× |
| Update PPO (timesteps/s) | 9 040 | 136 358 | 15,1× |

**Bout-en-bout** (`scripts/bench_e2e.py --device <d> --envs <E> --steps <T> --iters 5`, 4 itérations après
échauffement, ≈ 51–57k timesteps agents par itération, RocketGPU CPU build simple `rocketgpu-gcc`, obs provisoire de
89 dims en numpy, politique aléatoire initiale, synchrone sans recouvrement collecte/update) :

| Config | Collecte (ts/s) | Apprentissage (ts/s) | **Bout-en-bout (ts/s)** |
|---|---:|---:|---:|
| CPU, 1 024 arènes × 25 steps | 105 163 | 19 511 | **16 455** |
| XPU, 1 024 arènes × 25 steps | 107 330 | 100 216 | **51 530** |
| XPU, 4 096 arènes × 7 steps | 141 510 | 98 168 | **57 868** |

Lecture :
- Sur CPU, **l'apprentissage est le goulot** (≈ 84 % du temps). Le GPU intégré Arc le multiplie par ≈ 5 dans la boucle
  réelle et rééquilibre collecte et apprentissage : bout-en-bout ×3,1 à ×3,5.
- Écart non expliqué : l'update CPU vaut 9,0k ts/s dans `bench_learner` mais ≈ 19,5k dans `bench_e2e` (obs 107 vs 89
  dimensions, données et charge différentes). À investiguer avant d'utiliser le chiffre CPU isolé.
- Décomposition XPU 4 096 arènes, par itération : obs numpy 0,15 s, inférence 0,08 s, sim 0,16 s, update 0,58 s.
  Prochains leviers, à mesurer : recouvrement collecte/update (double buffer, staleness ≤ 1), obs en natif ou sur
  le device.
- Limites : politique aléatoire (presque aucun but, peu de resets), ≈ 30 s par point, une seule série, pas de
  throttling mesuré, pas de critiques d'exploration GCO-std (surcoût de ≈ 5 % à ≈ ×0,5 selon la spec §6.2).

### M4 — Optimisation du learner et de la boucle, puis banc soutenu (2026-11-11)
Code des bancs commité en **07571ca** (les variantes ont été mesurées sur l'arbre de travail équivalent, juste avant ce
commit). Laptop, torch 2.15.0+xpu, réseaux PPO v0, lot de 50k, minibatch de 25k, 2 epochs.

**Learner seul** (`vendor/venv-xpu/Scripts/python scripts/bench_learner.py --device xpu --iters 6 --skip-inference
<variante>`). Depuis 07571ca, les statistiques ne sont lues qu'en fin d'itération, au lieu de 2 synchronisations hôte
par minibatch.

| Variante | Update (timesteps/s) |
|---|---:|
| base | 137 052 |
| Adam `fused=True` | 162 984 ; 162 519 |
| autocast bf16 | 135 740 |
| autocast fp16 | 146 295 |
| fused + bf16 | 152 446 |
| fused + fp16 | 163 753 |
| fused + `torch.compile` | échec (`InductorError`) |

Retenu : Adam fused en fp32 (+19 %). La précision réduite n'apporte rien de plus.

**Threads de simulation** (RocketGPU PGO, 4 096 arènes, 12 s, `ROCKETGPU_CPU_THREADS`, 2 passages) : 4 → 162 860 /
132 594 ; 8 → 163 728 / 172 036 ; 12 → 191 897 / 200 184 ; 16 (défaut) → 203 040 / 203 506 ; 18 → 207 452 /
200 694 env-steps/s. Défaut conservé.

**Recouvrement** (`bench_e2e.py --device xpu --envs 4096 --steps 7 --iters 6 --fused [--overlap]`, 2 passages) :

| Build RocketGPU | Mode | Passage 1 | Passage 2 |
|---|---|---:|---:|
| `rocketgpu-gcc` | synchrone | 96 845 | 78 467 |
| `rocketgpu-gcc` | recouvrement | 99 155 | 96 273 |
| correctif GIL (expérimental) | synchrone | 57 632 | 76 145 |
| correctif GIL (expérimental) | recouvrement | 85 306 | 95 104 |

- Recouvrement retenu : stable à ≈ 96–99k, contre 78–97k en synchrone sur des séries courtes.
- Le correctif expérimental (`py::call_guard<py::gil_scoped_release>` sur `step`, `step_with_device_controls`,
  `apply_env_commands` et `reset_to_kickoff`) n'apporte rien : le thread learner attend surtout le GPU, ce qui relâche
  déjà le GIL. Il a été abandonné et supprimé, pour ne pas modifier de code tiers.
- Une file XPU séparée pour l'inférence (`--actor-stream`, laptop chaud, 25 itérations) donne 47 944 / 44 249 contre
  45 202 / 50 144 sans : écart dans le bruit. Abandonnée.

**Banc soutenu de 57 min** :
- Commande : `ROCKETGPU_PY_DIR=vendor/build/rocketgpu-cpu-pgo/python vendor/venv-xpu/Scripts/python
  scripts/bench_e2e.py --device xpu --envs 4096 --steps 7 --iters 3000 --fused --overlap`.
- Commit 07571ca (marqué `-dirty` à cause de modifications de documentation sans lien) ; départ 2026-11-11 15:28 ;
  secteur.
- Rapport : `runs/bench/bench_e2e_20261111_152829.json` (local).
- Résultat : 3 000 itérations, 172,0M timesteps en 57,0 min, soit une **moyenne de 50 323 timesteps/s**. Toutes les
  pertes et observations restent finies ; 2 850 buts (politique aléatoire).

| Décile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ts/s | 75 021 | 60 593 | 48 919 | 49 142 | 44 972 | 48 325 | 50 768 | 45 280 | 45 655 | 46 558 |

Lecture :
- **Throttling réel** : le débit baisse de 35 à 40 % après ≈ 10–15 min, puis se stabilise vers ≈ 45–50k ts/s. Par
  itération, la simulation passe de 0,183 à 0,238 s (+30 %) et l'apprentissage de 0,73 à 1,19 s (+62 %).
- CPU et GPU intégré partagent la même enveloppe de puissance : les gains mesurés sur 30 s surestiment le régime de
  croisière. **Le chiffre de planification est le plateau soutenu**, pas les mesures courtes (M3, M4 hors banc soutenu).

### M5 — Banc soutenu de la vraie boucle, pool et évaluation actifs (2026-11-14, tâche A5)
Commande : `vendor/venv-xpu/Scripts/python scripts/bench_run.py --minutes 45 --checkpoint-probe` ; seed 0 ; commit
dce396b (arbre propre) ; laptop Core Ultra 5 125H + Arc (XPU), 16 Go, sur secteur ; 2026-11-14 15:49 → 16:34 UTC ;
configs `configs/bench_4a.yaml` et `configs/bench_4a_eval.yaml` (paramètres de l'ADR-005). Smoke test PASS sur le même
commit (`runs/smoke_dce396b/smoke_summary.json`). Données brutes et résumé : `runs/bench_4a_20261114T154909Z/`.
Un seul passage : pas de répétition, thermique non contrôlée.

- **Débit** (timesteps appris/s, mural = évaluations, checkpoints et journal compris) : **46,0k en moyenne sur 45 min**
  (collecté : 51,2k), 2 409 itérations, 124,3M appris, 0 restauration. Par tranche de 5 min : 57,3k ; 49,9k ; 51,2k ;
  46,1k ; 45,9k ; 43,8k ; 39,8k ; 39,7k ; 40,8k. Dernière tranche / première = 0,68 (débit de boucle).
  **Le débit baisse encore à 30 min et n'est stable (≈ 40k) que sur les 15 dernières minutes** ; le pool grandit
  pendant tout le banc (7 → 49 instantanés, plafond 100 non atteint) : la part due au throttling et celle due au pool
  ne sont **pas séparées** par cette mesure, et le débit à pool plein (100) n'est **pas mesuré**.
- **K6** (< 25k appris/s pendant 30 min) : non déclenché, mesure concluante (45 min de boucle). Critère de révision
  de cet ADR (≥ 25k timesteps/s) : tenu.
- **Décomposition** (fil principal) : collecte 96,9 % du temps mural, dont simulation 17,2 %, obs + récompense + resets
  18,4 %, **inférence acteur + pool + tampons 61,3 %** (non décomposé : pas de point de mesure dans `Collector`).
  Apprentissage en recouvrement : 34,7 % (fil learner). Le poste dominant n'est ni le simulateur ni `build_obs`.
- **Évaluations** (512 + 512 épisodes, 256 arènes) : 2, à 13,1 s et 15,7 s, 165–169k steps d'arène/s, **1,1 %** du
  temps mural. Aucune évaluation lente : le cas à 179 s d'un mini-run d'A4 n'est pas reproduit (ni ici ni sur
  l'essai de 3 min) et reste inexpliqué. Politique encore quasi initiale : des épisodes plus longs coûteront plus.
- **Checkpoints** : 115,8 Mo à l'itération 0, 186,9 Mo à 49 instantanés (+0,7 Mo par instantané), écriture 1,1–1,7 s,
  **0,6 %** du temps à un checkpoint toutes les 250 itérations. Sonde directe à 100 instantanés : 224,3 Mo, écriture
  1,30 s, relecture 0,39 s (l'estimation de 74 Mo du sprint était fausse d'un facteur 3).
- **RAM** : RSS max 7,86 Go, mémoire privée max 11,8 Go, **mémoire disponible minimale 588 Mo sur 16 Go** ; XPU
  réservé max 464 Mo. La RSS finit à 3,9 Go alors que la mémoire privée reste haute : le système a vraisemblablement
  réduit l'ensemble de travail sous pression mémoire (interprétation, non vérifiée). La pente de RSS calculée par
  le script (négative) n'a donc pas de sens et n'est pas reprise.
- **Journaux** : 3,7 Mo en 45 min, ≈ 29 Mo à 1B (extrapolé).
- **Durée d'un run de 1B** (extrapolé) : 6,0 h au débit moyen du banc, **6,9 h à 40k/s** (dernières tranches) ;
  au-delà de 8 h si le débit descend sous ≈ 34,7k/s à pool plein, ce que ce banc ne permet pas d'exclure.

### M6 — Banc soutenu à pool plein (100 instantanés dès l'itération 0, 2026-11-14)
Commande : `vendor/venv-xpu/Scripts/python scripts/bench_run.py --minutes 35 --prefill-pool 100` ; seed 0 ; commit
d017826 (arbre propre) ; même laptop, sur secteur ; 2026-11-14 18:10 → 18:45 UTC ; mêmes configs que M5. Smoke test
PASS sur le même commit (`runs/smoke_d017826/smoke_summary.json`). Données : `runs/bench_4a_20261114T181031Z/`.
Un seul passage, lancé juste après le smoke test et un essai de 1,5 min (machine déjà chaude). **Limite** : le pool
est rempli par 99 copies de θ₀ (réseaux distincts en mémoire et à l'inférence, adversaires aléatoires) ; le coût est
celui d'un pool plein, la dynamique de jeu non. Non comparable tranche à tranche à M5.

- **Débit** (appris/s, mural) : **32,2k en moyenne sur 35 min** (collecté 35,8k), 1 310 itérations, 67,6M appris,
  0 restauration. Par tranche de 5 min : 35,3k ; 34,5k ; 34,4k ; 34,6k ; 33,3k ; **27,0k ; 26,3k**. Plat à ≈ 34,5k
  pendant 25 min, puis chute de ≈ 22 % (p10 de la tranche 25–30 : 23,9k).
- **Effet du pool** : à 100 instantanés, ≈ 35k dès les 5 premières minutes, contre 57,3k à 7 instantanés dans M5.
  La baisse observée dans M5 vient donc surtout du pool qui grandit, pas du throttling. « Inférence acteur + pool +
  tampons » : **76,0 %** du temps mural (61,3 % dans M5) ; simulation 13,5 % ; obs + récompense + resets 8,7 %.
- **Chute à 25 min** : cause non établie. Elle coïncide avec la mémoire disponible sous 1 Go (min par tranche :
  3,75 ; 3,33 ; 2,83 ; 1,50 ; 1,82 ; 0,92 ; 0,68 Go) ; pagination ou throttling non départagés.
- **K6** : non déclenché (35 min de boucle), mais les deux dernières tranches sont à 1–2k du seuil de 25k.
- **RAM** : RSS max 7,78 Go, privée max 9,22 Go, **disponible min 715 Mo**. La mémoire disponible baisse pendant tout
  le banc alors que le pool est déjà plein : la croissance ne vient pas des instantanés (origine non identifiée).
- **Checkpoints** : 225,0 Mo à 100–101 instantanés, écriture 1,5–3,3 s, 0,5 % du temps. Évaluation : 1, 12,8 s.
- **Durée d'un run de 1B** (extrapolé) : 8,6 h au débit moyen, 8,1 h à 34,5k/s, **≈ 10,4 h à 26,6k/s** (deux
  dernières tranches). **Un run de 1B ne tient pas dans une session de 8 h** dans la configuration de l'ADR-005
  (seuil : 34,7k/s) ; en 8 h à 26,6k/s : ≈ 0,77B (extrapolé).

### M7 — Coût du pool et croissance mémoire : causes et correctifs (2026-11-14, commit b43211d)
Même laptop, sur secteur, XPU. Sondes jetables (hors dépôt) sur les vrais composants, puis banc court.

- **Coût du pool** (sonde : acteur + 100 instantanés, 8 192 agents dont 819 gelés, 1 500 steps) : acteur seul
  3,7 ms/step ; avec un passage de réseau par instantané 72–82 ms/step (≈ 0,7 ms par réseau : coût de lancement des
  opérations XPU, pas du calcul). Correctif : un seul passage groupé sur les poids empilés
  (`training.selfplay._Stack`, `torch.baddbmm`) : **9–12 ms/step**. Logits égaux à ≈ 1e-5 près à ceux d'un passage
  par réseau (ordre des additions) : trajectoires non identiques bit à bit à celles d'avant b43211d, reprise exacte
  conservée (tests).
- **Croissance mémoire** (M6 : mémoire privée +95 Mo/min, linéaire, ≈ 2,3 Mo par itération) : ni le simulateur ni la
  collecte (sondes de 75 s : +0 Mo), ni le learner dans le thread principal (150 mises à jour : +5 Mo à K fixe,
  plateau à +75 Mo à K variable). **Cause : un thread learner neuf par itération** (`OverlapTrainer`) : +339 Mo en
  150 mises à jour, soit ≈ 2,3 Mo par thread, contre un plateau à +35 Mo avec un thread persistant. Correctif :
  thread learner persistant, arrêté par `close()`.
- **Banc court après correctifs** : `vendor/venv-xpu/Scripts/python scripts/bench_run.py --minutes 10 --prefill-pool
  100 --slice-min 2` ; seed 0 ; commit b43211d (arbre propre, smoke test PASS : `runs/smoke_b43211d/`) ; 19:53 →
  20:03 UTC ; données `runs/bench_4a_20261114T195348Z/`. **58,9k appris/s en moyenne sur 10 min à pool plein**
  (M6 : 35,3k sur les 5 premières minutes) ; par tranche de 2 min : 54,2k ; 53,3k ; 63,2k ; 62,0k ; 61,8k.
  Mémoire privée entre 5,9 et 6,3 Go de 1 à 10 min, sans pente (M6 : 5,7 → 6,6 Go sur la même durée). Les deux
  premières tranches coïncident avec ≈ 1,5 Go de mémoire disponible (autre processus, non identifié), puis 4 Go.
  « Inférence acteur + pool + tampons » : 50,6 % du temps mural (M6 : 76,0 %).
- **Limites** : 10 min, un seul passage, K6 non concluant ; le débit soutenu (throttling au-delà de 10 min) et la
  tenue mémoire sur 8 h ne sont **pas mesurés** après correctifs. 1B en 4,7 h au débit de ce banc (extrapolé).

### M8 — Banc soutenu de 2 h à pool plein, après correctifs (2026-11-14)
Commande : `vendor/venv-xpu/Scripts/python scripts/bench_run.py --minutes 120 --prefill-pool 100 --slice-min 10` ;
seed 0 ; commit 6998993 (arbre propre ; smoke test PASS sur ce commit, `runs/smoke_6998993/`) ; même laptop, sur
secteur ; 2026-11-14 20:10 → 22:10 UTC ; données `runs/bench_4a_20261114T201013Z/`. Un seul passage ; pool
pré-rempli par 99 copies de θ₀ (mêmes limites que M6 : coût d'un pool plein, pas sa dynamique de jeu).

- **Débit** (appris/s, mural) : **55,1k en moyenne sur 2 h** (collecté 61,3k), 7 691 itérations, 396,9M appris,
  0 restauration. Par tranche de 10 min : 67,0k ; 60,1k ; 60,9k ; 56,4k ; 51,0k ; 51,2k ; 53,1k ; 51,6k ; 51,3k ;
  53,5k ; 51,8k ; 53,6k. **Plateau à 51–54k de 40 min à 2 h, sans pente** ; dernière tranche / première = 0,83.
- **K6** : non déclenché, concluant ; marge d'un facteur 2 sur le seuil de 25k.
- **Mémoire** : privée médiane 6,06 à 6,30 Go par tranche de 20 min, sans pente (max 7,23 Go) ; RSS max 6,39 Go ;
  **disponible min 2,73 Go** (M6 : 0,72 Go en 35 min). La croissance de M6 a disparu.
- **Décomposition** : inférence acteur + pool + tampons 50,4 % ; simulation 21,0 % ; obs + récompense + resets
  23,3 % ; évaluations 1,7 % (7, de 13,9 à 20,0 s) ; checkpoints 1,0 % (32 de 225 Mo, 7,2 Go sur disque) ;
  « reste » 2,2 %.
- **Durée d'un run de 1B** (extrapolé) : 5,0 h au débit moyen, **≈ 5,3 h au plateau de 52k/s**. Un run de 1B tient
  dans une session de 8 h (seuil 34,7k/s) avec ≈ 2,7 h de marge. Non couvert : au-delà de 2 h, et des épisodes plus
  longs quand la politique progresse (évaluations et pas d'env plus coûteux).

## Temps de calcul
Base **mesurée** : plateau soutenu du laptop ≈ 50k timesteps/s (M4, 57 min, politique aléatoire). Seconde colonne :
facteur ×0,7, **estimé, non mesuré**, pour les critiques d'exploration GCO-std et l'évaluation (spec §6.2).

| Steps (jalon médian, spec §6.1) | À 50k ts/s (mesuré, soutenu) | À 35k ts/s (×0,7, estimé) |
|---|---:|---:|
| 1B (≈ S1 amorçage, médiane 0,7B) | 5,6 h | 7,9 h |
| 3B (≈ S2 bat ATBA, médiane 2B) | 17 h | 24 h |
| 10B (≈ S3 bot faible, médiane 8B) | 2,3 jours | 3,3 jours |
| 40B (≈ S4 Nexto, médiane 40B) | 9,3 jours | 13 jours |
| 100B | 23 jours | 33 jours |

Par rapport à la spec §6.2 (scénarios 5k / 20k / 60k SPS en CPU seul), le laptop se place entre les scénarios
« hybride » et « optimiste ». Sur Kaggle T4, seule la collecte est mesurée (≈ 55k env-steps/s, soit ≈ 110k timesteps
agents/s, sur les 4 vCPU qui servent aussi à l'apprentissage) : le bout-en-bout n'y est **pas mesuré**. Le TPU attend
sa sonde (M2c).

## Décision
Option **A3**, acceptée au G2 (GO du 2026-11-11) :
1. **Simulateur** : RocketGPU @f0508a1, build **CPU** scripté et épinglé (`scripts/build_rocketgpu.py`, PGO optionnel).
   RocketSim 2.2.1 (PyPI) est conservé comme référence de physique et comme repli. Le build CUDA est écarté pour
   l'instant : il ne compile que pour sm_120 (M2b).
2. **Environnement** : maison, batché sur l'API numpy de RocketGPU (contrôles et états en tableaux, sans objet Python
   par voiture). Pas de dépendance d'entraînement à RLGym. Les obs, actions et la parité avec le déploiement RLBot sont
   du ressort de la Phase 3 (test de parité exigé au G3).
3. **Collecte** : un seul processus, ≥ 1 024 arènes par lot (4 096 mesurées). L'apprentissage du lot k se fait dans
   un thread pendant la collecte du lot k+1 (staleness 1, `docs/01_papers.md` §1.3), avec une copie de la politique
   pour l'inférence (M4).
4. **Learner** : PPO et récompense en Python/torch (règle 7). Sur le laptop, torch XPU sur le GPU intégré Arc
   (`vendor/venv-xpu`, torch 2.15.0+xpu, `index_select` au lieu de `x[idx]`, Adam `fused=True`, fp32,
   statistiques lues une fois par itération), repli CPU. Sur Kaggle T4, torch CUDA de l'image.
5. **Plateformes** : le laptop est la plateforme principale mesurée (≈ 50k timesteps/s soutenus sur 57 min, M4). Les sessions
   Kaggle T4 sont un appoint (collecte CPU ≈ 55k env-steps/s, bout-en-bout non mesuré). Le rôle du TPU sera fixé par sa
   sonde (M2c) : c'est la seule partie réversible à court terme.

## Conséquences
- **Dépendance à un projet tiers jeune** (un auteur, créé le 2026-10-27). Mitigations : commit épinglé, build scripté et
  testé, RocketSim gardé en référence. La cohérence physique n'est vérifiée que sur une accélération en ligne droite
  (≤ 0,3 % à 120 ticks, M1c) : un test de comparaison RocketGPU / RocketSim sur des épisodes complets entre en Phase 3.
- **Deux environnements torch** sur le laptop : `.venv` (CPU) et `vendor/venv-xpu` (XPU, 3,8 Go). Le code
  d'entraînement doit tourner dans les deux ; le contournement `index_select` est testé.
- **Throttling mesuré** : −35 à −40 % après ≈ 10–15 min (M4). Ce plateau suppose une politique aléatoire et pas
  encore de critiques d'exploration. Le run court 4a le remesurera avec la vraie boucle avant tout run long (G3).
- **Kaggle T4 faible pour la simulation** (4 vCPU, GPU inutilisable pour RocketGPU). Le relais entre comptes
  (`claude/phase2-kaggle-relay-954373`) reste utile pour enchaîner des sessions, mais le gain par session est modeste.
- **Critères de révision** :
  - la sonde TPU mesure un hôte nettement plus rapide que le laptop ;
  - RocketGPU CUDA compile pour sm_75 ou sm_89 (correctif en amont), ce qui rendrait les GPU Kaggle et Colab utiles ;
  - un écart de physique RocketGPU / RocketSim apparaît en Phase 3 ;
    - **Examiné le 2026-11-11** (`docs/reports/physics_parity.md`, commit f4f1a45) : accord à ≤ 8 uu sur 120
      ticks hors contrôles aléatoires. Un écart net : au sol, boost avec gaz ≤ 0,01, RocketGPU freine
      (≈ 6,5 uu/s à 24 ticks) là où RocketSim accélère (≈ 464 uu/s). La table de 90 actions n'a aucune action
      « boost sans plein gaz » (0 sur 90, vérifié sur `bench_e2e.action_table`) : l'entraînement n'y passe pas.
      **Décision : A3 maintenue**, à condition que la règle « boost ⇒ plein gaz » soit testée dans
      `src/actions` (T1). Non comparé au jeu (Phase 5).
  - le débit soutenu de la vraie boucle (Phase 4a) tombe sous 25k timesteps/s, soit la moitié du plateau de M4.
