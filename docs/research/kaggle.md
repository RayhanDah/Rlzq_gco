# Kaggle comme plateforme d'entraînement (relevé 2026-11-10)

Contexte : au G1, l'utilisateur a indiqué que les entraînements tourneront sur Kaggle via clé API, en **2× T4** ou en **TPU v5e-8**.
Ce document sépare ce qui est **vérifié en source**, ce qui est **rapporté par des tiers**, et ce qui reste **à mesurer**.

## 1. Vérifié en source primaire
- **CLI `kaggle`** (`github.com/Kaggle/kaggle-api`, `docs/kernels.md`, `docs/kernels_metadata.md`) :
  - commandes `kaggle kernels push -p <dossier> [--accelerator <ID>] [-t <secondes>] [--no-run]`,
    `kaggle kernels status <user/slug>`, `kaggle kernels output <user/slug> -p <dossier> [--file-pattern <regex>]` ;
  - IDs d'accélérateurs listés « as of Sep 2026 » : `NvidiaTeslaT4` (T4 ×2), `TpuV5E8` (TPU v5e-8), `TpuV6E8`, `NvidiaL4`,
    `NvidiaL4X1`, `NvidiaTeslaA100`, `NvidiaH100`, `NvidiaRtxPro6000`. Certains sont réservés à des compétitions ou aux admins ;
  - `kernel-metadata.json` : `id`, `title`, `code_file`, `language` (`python`), `kernel_type` (`script`|`notebook`),
    `is_private` (défaut `true`), `enable_gpu`, `enable_internet` (défaut `false`), `machine_shape` (ex. `NvidiaTeslaT4`,
    `TpuV5E8`), `dataset_sources`, `kernel_sources`, `model_sources`, `competition_sources`.
- **Authentification** (`docs/README.md`) : `kaggle auth login` (OAuth), variable `KAGGLE_API_TOKEN`, fichier
  `~/.kaggle/access_token`, ou ancien `~/.kaggle/kaggle.json`.
  **Règle projet** : l'utilisateur installe lui-même le jeton. Il ne passe jamais par le chat, n'est jamais lu par un agent,
  jamais commité (garde `.gitignore` + hook).
- **Spec GCP d'une VM TPU v5e 8 puces** (`docs.cloud.google.com/tpu/docs/v5e`) : 224 vCPU, 384 Go de RAM, 2 nœuds NUMA.
  **Rien ne prouve que Kaggle expose cette VM entière.**

## 2. Rapporté par des tiers (fastero.com, 2026-10-31, d'après la doc Kaggle ; non revérifié : les pages kaggle.com/docs sont rendues en JS)
| Élément | Valeur rapportée |
|---|---|
| Quota GPU | ≈ 30 h/semaine (« or sometimes higher »), partagé entre types de GPU |
| Quota TPU | ≈ 20 h/semaine (« the TPU figure comes from users ») |
| Durée max d'une session | 12 h en CPU/GPU, **9 h en TPU** |
| CPU / RAM en session T4 ×2 | **4 cœurs, 29 Go** |
| CPU / RAM en session TPU | 96 cœurs, 330 Go, mais pour la **TPU v3-8** |
| Disque persistant | 20 Go dans `/kaggle/working`, le reste est effacé en fin de session |
| Exécution en arrière-plan | « Save & Run All » continue après fermeture de l'onglet ; une session interactive s'arrête après 20 min d'inactivité |
| GPU et internet | exigent un numéro de téléphone vérifié |

## 3. Conséquences pour RL-GCO
1. **La collecte (RocketSim) est CPU-bound, l'apprentissage d'un MLP PPO est léger.**
   - En T4 ×2, on n'a que **4 cœurs CPU**, moins que le laptop (Core Ultra 5 125H : 14 cœurs / 18 threads). Les T4 resteront
     presque inactives sur un MLP de 3×256. C'est donc probablement **la pire option** pour notre charge [inférence, à mesurer].
   - En TPU v5e-8, l'intérêt réel est **l'hôte CPU**. S'il expose 96 à 224 cœurs, la collecte pourrait monter à
     ~10⁵ SPS [hypothèse].
     - Learner : JAX sur TPU (Python, conforme à la règle), ou PyTorch CPU sur l'hôte.
     - À trancher par mesure en Phase 2.
2. **Le budget est en heures par semaine, pas en jours continus.** Avec ≈ 20 h TPU + ≈ 30 h GPU par semaine :
   - **pas de run long continu** : l'entraînement est une **chaîne de sessions ≤ 9 h** (TPU) ou ≤ 12 h (GPU) ;
   - checkpoints atomiques, reprise exacte (optimiseur, normaliseurs, RNG, pool self-play, compteurs) ;
   - sauvegarde régulière dans `/kaggle/working`, avant le timeout.
3. **Orchestration depuis le laptop** :
   - `kaggle kernels push` d'un script versionné (commit git dans les métadonnées du run) ;
   - le checkpoint précédent est passé en entrée (`kernel_sources` ou dataset privé) ;
   - les sorties sont récupérées par `kaggle kernels output` ;
   - TensorBoard est rapatrié ;
   - une skill `/kaggle-run` encapsule tout ça.
4. **Dépendances** :
   - `rocketsim` et RLGym doivent s'installer sur l'image Kaggle (Linux x86_64), avec internet activé ou des wheels
     poussées en dataset privé ;
   - les **collision meshes** passent par un dataset privé (jamais public, licence des assets du jeu) ;
   - l'image fournit JAX / torch, versions à relever par la sonde.
5. **Déploiement** : inchangé. Inférence CPU sur le laptop, sous RLBot v5. Si le learner est en JAX, il faut :
   - un export vers ONNX, ou une réimplémentation numpy/torch de la politique ;
   - un **test de parité numérique** obligatoire.
6. **Risques ajoutés** :
   - **R7** : quotas et durées de session modifiables à tout moment par Kaggle (mesurer le quota réel au premier run) ;
   - **R8** : Kaggle pourrait interrompre une session à accélérateur qui ne l'utilise pas. **Non vérifié** dans les CGU,
     à lire avant le G2 ;
   - **R9** : 4 cœurs seulement en T4, ce qui plafonne la collecte ;
   - **R10** : fuite du jeton API. Mitigation : jamais dans le dépôt, hook de refus, jeton révocable.

## 4. Sonde à lancer en Phase 2 (avant tout calcul lourd)
Un kernel `probe` par accélérateur, durée ≤ 15 min. Il relève :
- `nproc`, `lscpu` (modèle CPU, NUMA), `free -g`, `df -h /kaggle/working` ;
- `python -V`, `pip list` (torch, jax, numpy), `jax.devices()`, `nvidia-smi` ;
- un micro-benchmark RocketSim 1v1 à 1, 4, N processus (steps/s au tick-skip 8) ;
- un step PPO factice sur l'accélérateur et sur le CPU.

Les résultats vont dans `docs/adr/001-infra.md`, avec la commande, le commit et la date.

## 5. Mise à jour du 2026-11-10 : disponibilité réelle et mutualisation (informations de l'utilisateur et CGU vérifiées)
**Disponibilité rapportée par l'utilisateur** (non mesurée par nous) :
- les GPU (T4 ×2) démarrent immédiatement ;
- le TPU v5e-8 impose une file d'attente d'**au moins 35 h** avant chaque session de 9 h, soit au mieux ≈ 9 h utiles toutes
  les 44 h et environ 20 % du temps.
⇒ Le **TPU devient opportuniste**. La ligne principale tourne sur des **sessions T4 chaînées**, plus le laptop.

**Idée « une clé API par camarade » : vérification des CGU** (lues le 2026-11-10 sur `kaggle.com/terms` et `kaggle.com/aup`) :
- Terms : « You will not share your account or password with anyone » et « You are not allowed to allow multiple individuals
  to operate under one Kaggle User ID » ; une violation est un motif de résiliation du compte.
- AUP : interdit « to disable, interfere with or circumvent any aspect of the Services » et « to abuse resources ».
- **Conséquence** : centraliser les clés des camarades et lancer nous-mêmes des kernels sur leurs comptes revient à partager
  ces comptes. C'est **contraire aux CGU** et met leurs comptes en risque de résiliation. Le studio **ne collecte ni ne
  stocke aucune clé tierce**.
- **Variante compatible** : chaque camarade **opère lui-même son compte**, et il garde son jeton sur sa machine.
  - Il lance le kernel du projet en exécutant `kaggle kernels push` ou « Copy & Edit » d'un notebook partagé.
  - Les checkpoints passent par un **dataset Kaggle privé partagé en collaborateurs** (ou une release GitHub privée).
  - **Mécanisme exact à vérifier en Phase 2** : droits des collaborateurs d'un dataset, montage d'un dataset en entrée,
    publication d'une nouvelle version depuis un kernel.
- **Zone grise restante** : pooler les quotas de plusieurs personnes pour un même projet pourrait être lu comme un
  contournement de limite (« circumvent »). L'usage en équipe, où chaque membre utilise son propre quota, est courant en
  compétition, mais aucune règle explicite n'a été trouvée.
  **Décision de l'utilisateur requise** (voir `production/state.md`).

**Capacité si la variante compatible est retenue** (hypothèses, non mesuré) :
- 6 comptes × ≈ 30 h/semaine ≈ 180 h de T4 par semaine. Deux usages possibles :
  - une chaîne 24/7 (168 h), ou mieux **3 seeds en parallèle × ≈ 60 h**, puisque PPO séquentiel ne se distribue pas entre
    kernels ;
  - à 6k SPS (4 cœurs, estimation) : ≈ 3,6B steps/semaine pour la chaîne 24/7.
- Coût humain : un relais toutes les ≤ 12 h. Un camarade absent casse la chaîne, mais pas le run, grâce à la reprise
  depuis checkpoint.

## 6. Mise à jour du 2026-11-10 (suite) : groupe de 3, débit TPU rapporté, concurrence
**Décision de l'utilisateur** : groupe de 3, donc 3 comptes. Chaque membre opère son compte depuis son PC (`git pull` puis
une commande, avec son propre jeton). Notebook commun et datasets partagés si possible. Le studio ne manipule que le compte
de l'utilisateur.

**Débit TPU rapporté par l'utilisateur** (non mesuré) : ≈ 2,5B steps par session de 9 h, soit ≈ 77k SPS, une session tous les
≈ 48 h. Calculs dérivés (estimés, non mesurés) :
- le quota TPU rapporté (≈ 20 h/semaine) autorise au plus 2 sessions de 9 h par compte et par semaine, soit ≈ 5B steps ;
- 3 comptes : jusqu'à ≈ 15B steps/semaine si chacun obtient ses créneaux ;
- une session T4 à ≈ 6k SPS (4 cœurs, estimation) donne ≈ 0,26B steps en 12 h, soit ≈ 10× moins qu'une session TPU.
⇒ Si la sonde confirme l'ordre de grandeur, le TPU redevient la ligne principale et les T4 servent d'appoint (seeds,
ablations, éval).

**GPU et TPU en même temps** :
- Quotas GPU et TPU distincts ; le quota GPU « flottant » ne s'applique pas au TPU
  ([kaggle.com/product-feedback/173129](https://www.kaggle.com/product-feedback/173129)).
- Ancienne annonce Kaggle : 1 session GPU interactive + 2 sessions GPU en commit simultanées
  ([kaggle.com/discussions/general/105509](https://www.kaggle.com/discussions/general/105509), date et validité actuelle
  non vérifiées). Aucune limite TPU concurrente trouvée.
- **À tester par la sonde** : pousser un kernel T4 et un kernel TPU sur le même compte et relever `kaggle kernels status`.
- Même si c'est possible, un run PPO est séquentiel : GPU et TPU en parallèle = **deux runs distincts** (seeds, ablation,
  pool d'adversaires), pas un même run accéléré. Alterner TPU et T4 sur un même run est possible si le checkpoint est
  portable entre les deux (même framework, learner indépendant du device).
