# Sonde Kaggle : résultats

Sonde : `scripts/kaggle_probe.py` + `kaggle/probe/probe_kernel.py`, kernels privés sur le compte de l'utilisateur, lancés
avec son accord explicite le 2026-11-11. Sorties rapatriées dans `kaggle_out/` (non versionné). Synthèse et décisions :
`docs/adr/001-infra.md` (M2).

## T4 ×2 — `<pseudo>/rlgco-probe-t4`, version 1
- Commande : `python scripts/kaggle_probe.py --user <pseudo> --acc t4 --push` (commit **a74ac7f**). Parties par défaut
  avec GPU : `rg_cuda, rocketsim, rg_cpu`. Démarrage 2026-11-11 11:00:35 UTC, durée 305 s, statut COMPLETE.
- Le kernel TPU a été poussé pendant que celui-ci tournait : accepté et mis en file (QUEUED). La concurrence effective
  GPU + TPU n'est pas encore observée.

### Machine relevée
| Élément | Valeur |
|---|---|
| CPU | 4 vCPU : Intel Xeon @ 2,00 GHz, 2 cœurs × 2 threads ; cgroup `cpu.max` = 400000/100000 |
| RAM | 31 Go, sans swap |
| GPU | 2 × Tesla T4, 15 Go chacun, compute capability 7.5 ; jax voit 2 `CudaDevice` |
| Disque `/kaggle/working` | 20 Go |
| Toolchain | nvcc CUDA 13.0 (V13.0.88), g++ 13.3, Python 3.13.16 |
| Paquets présents | torch 2.11.0+cu130, jax 0.11.1, numpy 2.1.3 ; installés par la sonde : rocketsim 2.2.1, cmake 3.31.10, ninja 1.13.2, pybind11 3.1.0 |

Le rapport tiers (« 4 cœurs, 29 Go », `docs/research/kaggle.md` §2) est confirmé à l'arrondi près.

### Débits (15 s par point, actions aléatoires, 1v1, tick-skip 8)
| Mesure | env-steps/s |
|---|---:|
| RocketSim 2.2.1, 1 / 2 / 4 / 8 processus | 14 955 / 28 830 / 36 492 / 32 677 |
| RocketSim `multi_step`, 1 processus × 256 arènes | 18 510 |
| RocketGPU CPU (build 88 s), 1 024 arènes | 50 422 |
| RocketGPU CPU, 4 096 arènes | 55 253 |

### Build RocketGPU CUDA : échec
- Étape `build_rocketgpu_cuda`, 67,7 s, code 1. La cible `src/rocketgpu_cuda.cu.o` échoue avec
  `--generate-code=arch=compute_75,code=[compute_75,sm_75]`.
- La cause est dans le journal complet (`logs/build_rocketgpu_cuda.log`). Ce n'est pas une erreur de C++ : c'est le
  vérificateur LLVM interne de `cicc` (NVVM, « Producer: 'LLVM7.0.1' ») qui rejette le code intermédiaire généré.
  Messages : « Stored value type does not match pointer operand type! » (sur des pointeurs vers
  `CarSharedSplitContactRow`, `CarSharedVelocityContactRow`, `CarSharedFrictionContactRow`), puis « Explicit load/store
  type does not match pointee type of pointer operand ».
- Même échec sur Colab T4 (CUDA 13.0, M2a de l'ADR) : le problème est reproductible avec ce couple CUDA 13.0 / sm_75.
- Hypothèses, **non vérifiées** : défaut du compilateur CUDA 13.0 pour sm_75, ou motif de code RocketGPU jamais
  compilé pour Turing par son auteur (benchmarks publiés sur RTX 5090).
- Pistes, toutes à faire avec l'accord de l'utilisateur :
  1. compiler avec une toolchain CUDA 12.x (paquets pip NVIDIA ou conda) dans le kernel ;
  2. signaler le problème en amont (dépôt RocketGPU) ;
  3. renoncer au GPU T4 pour la simulation : RocketGPU CPU sur les 4 vCPU, et le GPU pour le learner.

## T4 ×2 — version 2 : RocketGPU CUDA avec CUDA 12.9
- Commande : `python scripts/kaggle_probe.py --user <pseudo> --acc t4 --parts rg_cuda12 --push` (commit **b4b0481**).
  Démarrage 2026-11-11 11:26:51 UTC, durée 116 s.
- `install_cuda12` (apt, dépôt NVIDIA ubuntu2404, `cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-cccl-12-9`) : réussi en
  19,7 s. CMake identifie « NVIDIA 12.9.86 » et compile avec `/usr/local/cuda-12.9/bin/nvcc`.
- `build_rocketgpu_cuda12` : **même échec** que sous CUDA 13.0, pour la même cible (`rocketgpu_cuda.cu.o`, sm_75), avec
  les mêmes messages de NVVM (« Stored value type does not match pointer operand type! », « Producer: 'LLVM7.0.1' »).
- Conclusion : ce n'est **pas** propre à CUDA 13.0. Le défaut tient au couple code RocketGPU @f0508a1 / compilation
  NVVM pour sm_75, ou au code lui-même. Aucun seuil `__CUDA_ARCH__` numérique dans le source (recherche du 2026-11-11).
- Hypothèse **non vérifiée** : NVVM suit un chemin interne différent selon l'architecture visée, et celui de sm_75
  (LLVM 7) rejette ce code alors que celui des GPU récents (RTX 5090 de l'auteur) l'accepte. Test possible sans quota
  GPU : compiler pour sm_75, sm_86, sm_89 et sm_120 dans une session Kaggle CPU.

## Session CPU — compilation de l'objet CUDA par architecture (`<pseudo>/rlgco-probe-cpu`, version 1)
- Commande : `python scripts/kaggle_probe.py --user <pseudo> --acc cpu --parts cuda_archs --push` (commit **c8f865c**).
  Session sans accélérateur (4 vCPU, pas de nvcc dans l'image CPU), démarrage 2026-11-11 11:59:11 UTC, durée 684 s.
- CUDA 12.9 installé par apt (24,8 s). Cible Ninja seule : `CMakeFiles/rocketgpu.dir/src/rocketgpu_cuda.cu.o`, avec
  `-DCMAKE_CUDA_ARCHITECTURES=<arch>` et le reste des options du projet inchangé.

| Architecture | GPU typiques | Résultat |
|---|---|---|
| sm_75 | T4 | échec en 13,5 s (NVVM « LLVM7.0.1 », mêmes messages) |
| sm_80 | A100 | échec en 10,0 s |
| sm_86 | RTX 30xx | échec en 11,5 s |
| sm_89 | RTX 40xx, L4 | échec en 10,3 s |
| sm_90 | H100 | échec en 10,4 s |
| sm_120 | RTX 50xx (Blackwell) | **réussi** en 585 s |

- **Conclusion (mesurée)** : avec CUDA 12.9, RocketGPU @f0508a1 ne compile en CUDA que pour sm_120 parmi les
  architectures testées. Toutes les architectures jusqu'à sm_90 échouent dans le même chemin NVVM « LLVM7.0.1 ». C'est
  cohérent avec les benchmarks de l'auteur, faits sur RTX 5090.
- Interprétation, non vérifiée : NVVM traite Blackwell par un autre chemin interne, qui accepte ce code.
- **Conséquence** : aucun GPU accessible au projet (Kaggle T4, Colab T4, L4 ou A100) ne peut exécuter la version CUDA
  actuelle de RocketGPU. Sur ces plateformes, RocketGPU tourne en CPU.

## TPU v5e-8 — `<pseudo>/rlgco-probe-tpu`, version 1
Poussé le 2026-11-11 (commit a74ac7f), en file d'attente (QUEUED) au moment de la rédaction. Résultats à venir.
