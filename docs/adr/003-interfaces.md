# ADR-003 — Contrat d'interface env / obs / actions / PPO / checkpoint (v0, 1v1)

- Statut : **accepté** (les six points validés par l'utilisateur le 2026-11-11)
- Date : 2026-11-11
- Décideurs : tech-director ; relecture rl-algo-lead (§5–7) et integration-lead (§3, §4, §7)

## Contexte
T1 (`src/obs`, `src/actions`), T2 (`src/env`) et T4 (`src/models`, `src/training`) partent en parallèle en worktrees ;
T5 et T7 suivent. Il faut un contrat figé : tableaux d'état, obs, actions, sortie de `step`, tampon, checkpoint.
Cadre déjà décidé : infra A3 (ADR-001), récompense `src/rewards` (ADR-002), spec §2.2, §2.3, §7.5, PPO v0
(`docs/01_papers.md` §2). Le prototype `scripts/bench_e2e.py` (obs 89 dims) est provisoire.
Abréviations des sources : `RG` = `vendor/src/RocketGPU` @f0508a1 ; `H` = `RG/include/rocketgpu/rocketgpu.h` ;
`B` = `RG/python/rocketgpu_bindings.cpp` ; `V` = `RG/python/rocketgpu_py_cuda.py` ; `E` =
`RG/include/rocketgpu/detail/step/env_step.hpp` ; `P2` = `repos/pulsar2/GigaLearnCPP/RLGymCPP/src/RLGymCPP`
(donnée, citée, jamais copiée).

## Options
| Sujet | Retenue | Écartée | Raison |
|---|---|---|---|
| Format v0 | 1v1, `n_cars` paramétré | 1v1 + 2v2 | G1 H5 « 1v1 d'abord » ; 2v2 double les tests |
| État → obs | `BatchState`, vues sans copie | copie dense | zéro copie (`V:92-105`) ; même type côté RLBot |
| Miroir | rotation 180° autour de z | réflexion y → −y | rotation propre : pas de miroir d'action |
| Obs | 109 dims, type « Advanced » | prototype 89 | `01_papers.md` §2 ; corrige 2 défauts (§3) |
| V(s′) tronqué | obs finale, V par le learner | V par le collecteur | collecteur = copie de π seule (ADR-001) |

## Mesures
Aucune. Décision réversible sur la base du source de RocketGPU @f0508a1 et du prototype `bench_e2e.py`. Tout effet
sur le débit (obs à 109 dims au lieu de 89) est estimé, non mesuré : T2 remesure `benchmark_sim`, T6 le bout-en-bout.

## Décision

### 1. Format : 1v1 uniquement en v0
Ferme : 2v2/3v3, coéquipiers dans l'obs, plusieurs agents par équipe (déjà gérés par `rewards`, A8).
Reste extensible : `n_cars` (pair) est un paramètre partout ; slot `k < n_cars/2` = bleu, sinon orange ; la dimension
d'obs suit une formule (§3). Passer au 2v2 = nouvel `OBS_SCHEMA_VERSION`, pas un nouveau contrat.

### 2. `BatchState` (défini dans `src/obs/state.py`, pur numpy, sans import de RocketGPU)
E = `n_envs`, C = `n_cars` (= `SimConfig.max_cars_per_env`, `H:748-750`), P = 34 pads (`detail/constants.hpp:16`).
Slot de voiture `e·C + k` (`E:6435`), pads `e·P + i` (`src/rocketgpu.cpp:2312-2313`). Les vues viennent de
`host_struct_array` (lecture seule, zéro copie, `V:92-105`), remodelées `(E, C)` puis indexées par champ (vues
strided). Elles **aliasent la mémoire du sim** : valides jusqu'au prochain `step`/`apply_env_commands`/upload ; aucun
consommateur ne les conserve.

| Champ | Shape | dtype | Unité | Source (`B` = layout, `H` = struct) |
|---|---|---|---|---|
| `ball_pos`, `ball_vel`, `ball_ang_vel` | (E, 3) | f32 | uu, uu/s, rad/s | `B:717-720`, `H:446-449` |
| `car_pos`, `car_vel`, `car_ang_vel` | (E, C, 3) | f32 | uu, uu/s, rad/s | `B:643-646`, `H:179-184` |
| `car_quat` | (E, C, 4) | f32 | (w, x, y, z), corps → monde | `B:644`, `H:92-96`, `detail/math.hpp:865-869` |
| `car_boost` | (E, C) | f32 | 0–100 | `B:652`, `H:197`, `boost_max` `H:644` |
| `car_{on_ground,has_jumped,has_double_jumped,has_flipped,is_demoed}` | (E, C) | u8 | 0/1 | `B:661-671`, `H:239-252` |
| `car_air_time_since_jump` | (E, C) | f32 | s | `B:651`, `H:196` |
| `pad_active` / `pad_cooldown` | (E, P) | u8 / f32 | 0/1 / s (unité supposée) | `B:739-740`, `H:483-485` |
| `car_team` | (C,) | i64 | `rewards.Team` | constant ; `rg.Team` Blue=0, Orange=1 (`H:102-105`) |

Repère : monde du sim, z vers le haut ; filet bleu en y < 0 (balle à y ≤ −seuil ⇒ orange marque, `E:6446-6454`).
`vel`/`ang_vel` portent la valeur RLBot après plafonds 2300 uu/s et 5,5 rad/s (`H:181-182`) : utile pour la parité.
Les champs privés de l'env (scores, `tick`, `ball_hit_info`) ne sont pas dans `BatchState` : l'obs n'y a pas accès.

### 3. Observations : `OBS_SCHEMA_VERSION = 1`, `OBS_DIM = 51 + 29·C` = 109 en 1v1, float32, `(n_agents, OBS_DIM)`
Ordre figé (blocs contigus, tranches exportées par `src/obs/schema.py`) :

| Bloc | Dims | Contenu (après miroir si orange) | Normalisation |
|---|---|---|---|
| balle | 9 | pos, vel, ang_vel | pos ÷ (4096, 5120, 2044) ; vel ÷ 2300 ; ang_vel ÷ 6 (`H:638`) |
| action précédente | 8 | ligne de la table §4 envoyée au step précédent ; 0 après reset | aucune |
| pads | 34 | 1 si actif, sinon 1/(1 + cooldown), dans l'ordre miroir | déjà dans ]0, 1] |
| soi | 29 | bloc voiture | voir ci-dessous |
| autres voitures | 29·(C−1) | v0 : 1 adversaire ; 2v2 : coéquipiers puis adversaires, ordre de slot | idem |

Bloc voiture (29) : pos 3, avant 3 (R·x̂), haut 3 (R·ẑ), vel 3, ang_vel 3 (monde, miroir) ; ang_vel locale 3 ;
balle − voiture en repère local : position 3, vitesse 3 ; boost/100 ; `on_ground` ; `has_flip_or_jump` ; `is_demoed` ;
`has_jumped`. Échelles voiture : ang_vel ÷ 5,5 (`H:641`), le reste comme la balle. Repère local = (R·x̂, R·ŷ, R·ẑ)
calculé du quaternion (`math.hpp:865-869`) ; le signe de R·ŷ face au repère RLBot est à vérifier en T7.
`has_flip_or_jump` = `on_ground | (¬has_flipped & ¬has_double_jumped & air_time_since_jump < 1,25)` : même formule
que `CarState::HasFlipOrJump` (`H:211-213`), méthode C++ non exposée, recalculée en numpy.
Miroir orange : M = diag(−1, −1, 1) sur tout vecteur monde ; les features locales en découlent sans traitement ;
pads permutés par `perm` tel que `loc[perm[i]] = M·loc[i]` (positions : `rg.boost_pad_locations()`, `B:562-567`,
recopiées en constante dans `src/obs/pads.py` et testées contre RocketGPU). Test §7.5 : obs(orange, s) = obs(bleu,
miroir(s)).
Écarts au prototype (`bench_e2e.py:80-115`) : (a) pads non permutés sous miroir (`bench_e2e.py:5-6,114`) — corrigé ;
(b) `can_flip` sans fenêtre de 1,25 s (`bench_e2e.py:84`) — aligné sur `H:211-213` ; (c) ajout ang_vel locale et
balle relative locale (structure proche de `P2/ObsBuilders/AdvancedObs.cpp:7-22`, conçue ici, pas copiée) ;
(d) timers de pads (`AdvancedObs.cpp:48-55`, même idée) ; (e) `has_jumped` ; (f) blocs voiture en fin pour le 2v2.
**Exclus** : temps restant, durée d'épisode, ticks sans touche (horloges de troncature, spec §2.2), score.
Pas de normaliseur courant d'obs en v0 : coefficients fixes, rien à transporter vers le bot hormis la version.

### 4. Actions : `ACTION_SCHEMA_VERSION = 1`, `N_ACTIONS = 90`, table `(90, 8)` float32
Colonnes `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`. Ordre = celui de `bench_e2e.py:45-69`, qui
énumère comme `P2/ActionParsers/DefaultAction.cpp:13-59` :
- sol (24) : throttle, steer ∈ (−1, 0, 1), boost, handbrake ∈ (0, 1), boucles imbriquées dans cet ordre ; on saute
  boost = 1 si throttle ≠ 1 ; ligne `[throttle, steer, 0, steer, 0, 0, boost, handbrake]` ;
- air (66) : pitch, yaw, roll ∈ (−1, 0, 1), jump, boost ∈ (0, 1), dans cet ordre ; on saute jump = 1 avec yaw ≠ 0,
  et pitch = roll = jump = 0 ; handbrake = jump ∧ (pitch, yaw ou roll ≠ 0) ;
  ligne `[boost, yaw, pitch, yaw, roll, jump, boost, handbrake]`.
Pas de miroir d'action : les contrôles sont exprimés dans le repère de la voiture, invariants par la rotation M.
Écriture : `actions.write_controls(ctrl, idx)` remplit le tampon `np.zeros(car_capacity,
dtype=state_dtype("controls"))` (`bench_e2e.py:180`) champ par champ (axes f32, booléens u8, `B:589-599`,
`H:116-125`), puis `step_with_device_controls(ctrl.ctypes.data, 8, 0)` (stride 0 = contrôles tenus 8 ticks,
`benchmark_sim.py:166`, `B:1044-1056`). `ACTION_TABLE_SHA256` = sha256 des octets de la table.

### 5. Environnement (`src/env/batch_env.py`, classe `BatchEnv`)
Agents à plat `a = e·C + k` (= slot de voiture), `N = n_agents = E·C`. API :
- `reset(seed) -> obs (N, D) f32` ; `set_p_kick(p)` (spec §2.3, annealing piloté par le trainer) ;
- `step(actions: (N,) i64) -> (obs (N, D) f32, reward (N,) f32, terminated (N,) bool, truncated (N,) bool, info)`.
Ordre interne d'un step :
1. contrôles §4 ; sim 8 ticks ; `episode_ticks += 8` ;
2. but : Δscores lus dans `env_states` (`B:751-752`) ; `Δblue > 0` ⇒ `GoalNet.ORANGE`, `Δorange > 0` ⇒
   `GoalNet.BLUE` ; tester `> 0`, pas `== 1` (aucune garde vue contre plusieurs incréments, `E:6444-6454`) ;
3. touche : voiture touchée pendant le step si `ball_hit_info.is_valid` et `tick_count_when_hit ≥ tick` d'avant
   step (horodaté sur `env.tick`, `triangle_and_carcar.hpp:1216-1218`, `E:6476`) ; `ticks_since_touch` = 0 si une
   touche, sinon += 8 ; dernier toucheur = argmax de `tick_count_when_hit` (−1 sans touche depuis le reset).
   L'env ne pose jamais `reset_tick_count` ni `reset_scoreboard` (`E:5485-5491`) : horloges monotones ;
4. `RewardFunction.step(goal_net, car_team, ticks_since_touch, episode_ticks, scorer)` une fois pour les E arènes ;
   `scorer` = dernier toucheur seulement si la variante est scorer-only, sinon `None` (A10). Récompenses (E, C) → (N,) ;
5. obs de s′ pour toutes les arènes ; pour les arènes finies (term ou trunc) : `info.final_obs` (K, D) et
   `info.final_agent` (K,) i64 reçoivent ces lignes, puis reset de ces arènes (kickoff via `EnvCommand`,
   `keep_boost_pads = 0` `H:602-604`, ou état générique), compteurs et action précédente à 0, recalcul de leurs
   lignes d'obs. L'`obs` retournée pour ces agents est le premier état du nouvel épisode.
États génériques (T2) : `upload_state_array` écrit les `count` premiers structs (`B:883-907`) ⇒ copie du tableau
entier, modification des arènes concernées, `tick_count_since_update = 0` (`H:205-206`).
`info` contient aussi `goal_net` (E,), `last_toucher` (E,), `episode_ticks` (E,) pour les stats ; jamais dans l'obs.

### 6. Tampon de rollout (`src/training/rollout.py`) : T steps × N agents

| Champ | Shape | dtype |
|---|---|---|
| `obs` | (T, N, D) | f32 |
| `act` | (T, N) | i64 |
| `logp` | (T, N) | f32 |
| `rew` | (T, N) | f32 |
| `term`, `trunc` | (T, N) | bool |
| `final_obs`, `final_t`, `final_agent` | (K, D), (K,), (K,) | f32, i64, i64 (lignes tronquées seulement) |
| `last_obs` | (N, D) | f32 (bootstrap à t = T) |
| `learn_mask` | (N,) | bool (tout vrai en v0 ; pool self-play plus tard) |
| `policy_version` | scalaire | i64 (version de π du collecteur) |

GAE : `V_next[t]` = V(`obs[t+1]`) (ou `last_obs` à t = T−1) ; = V(`final_obs`) si `trunc` ; = 0 si `term` ;
continuation = ¬(term ∨ trunc). Le prototype (`bench_e2e.py:126`) bootstrappe sur V(s_t) : à ne pas reprendre.
Les valeurs sont calculées par le learner. Recouvrement : le thread learner reçoit la propriété du lot k (dict de
tableaux numpy jamais réécrits ; double tampon côté collecteur) et ne touche ni au sim ni à `BatchState` ; le
collecteur utilise une copie de π rafraîchie après `join` (staleness 1, ADR-001 §3).

### 7. Checkpoint minimal (T5) et recalcul côté RLBot (T7)
Contenu : poids π et V ; état Adam ; RNG python, numpy, torch, et graine/état de l'env ; compteurs (itération, steps
agents, `policy_version`, `p_kick`) ; config d'entraînement complète et son hash ; `RewardFunction.config_hash`,
`SPEC_VERSION`, `HASH_SCHEMA_VERSION` (`src/rewards/config.py:25-29`) ; `OBS_SCHEMA_VERSION`, `OBS_DIM`,
`ACTION_SCHEMA_VERSION`, `ACTION_TABLE_SHA256`, `tick_skip` ; commit git, commit RocketGPU, versions torch/numpy.
Reprise exacte : checkpoint à la frontière d'itération, avec le lot k+1 déjà collecté (recouvrement) et l'état du
sim (`download_internal_state`, `B:918-944`, + tableaux d'état et compteurs de l'env). La restauration bit à bit de
RocketGPU n'est pas vérifiée : T5 la teste ; sinon l'exactitude est définie « modulo reset des arènes » et consignée.
Le bot doit recalculer à l'identique : `BatchState` (E = 1) depuis le paquet RLBot, `build_obs` (même fonction),
`perm` des pads par position, action précédente = ligne réellement envoyée, table §4, action tenue 8 ticks. Il refuse
un checkpoint dont les versions ou `ACTION_TABLE_SHA256` diffèrent des siennes.
Note (2026-11-22, écriture en flux) : format de fichier inchangé (`MAGIC | sha256(corps) | corps`, octets identiques
à l'ancien `_pack`, testés). `CheckpointManager.save` écrit le `.part` en flux (digest calculé à la volée, posé par
`seek`), `_verified` hache par blocs de 1 Mo. Pic mémoire sur un état synthétique de 0,49 Go : 4,8x -> 0,81x la
taille du fichier (mlops-lead, 2026-11-22, CPU, torch 2.15.0+cpu, psutil 7.2.2, script jetable non versionné : indicatif,
non reproductible). Mesure de référence : chute de RAM par checkpoint du banc `bench_int_aux` qui suit ce commit
(`aux_summary.json`, `ram.checkpoints`), à comparer aux 1 663 Mo de chute médiane avant ce changement (`runs/bench_int_aux_l0p1_20261122T103801Z`).
La règle d'aliasing d'`encode` (`.clone()` des ndarray) est inchangée.

### 8. Arborescence et propriété (un writer par ensemble)

| Tâche | Fichiers |
|---|---|
| T1a (1er commit de T1, mergé avant T2) | `src/obs/state.py` (`BatchState` + `validate()`) |
| T1 | `src/obs/{schema,pads,builder}.py`, `src/actions/table.py`, tests obs/actions |
| T2 | `src/env/{batch_env,sim_adapter,initial_states}.py`, `tests/unit/test_env.py` |
| T4 | `src/models/{policy,value}.py`, `src/training/{rollout,gae,ppo}.py`, tests PPO/GAE |
| T5 | `src/training/{checkpoint,guards,logging}.py`, `tests/unit/test_checkpoint.py` |
| T7 | `tests/integration/test_parity.py`, `docs/adr/004-parite.md` ; puis adaptateur RLBot → `BatchState` |

T4 ne dépend que de `OBS_DIM`, `N_ACTIONS` et §6 : il travaille sur un env jouet sans attendre T1/T2.

## Conséquences
- Trois writers parallèles sans fichier commun ; seul `src/obs/state.py` sérialise T1a avant T2.
- Toute modification d'obs ou de table incrémente sa version, casse le chargement des vieux checkpoints (voulu) et
  met à jour le test de parité.
- Révision si : la parité RLBot (T7) ne peut pas reproduire un champ (ex. `pad_cooldown`, `air_time_since_jump`) ;
  le coût mesuré de l'obs à 109 dims dégrade nettement le débit ; passage au 2v2.

## Précisions apportées à l'implémentation (T1, 2026-11-11, revue APPROVE)
- §3 : les vecteurs en repère local (balle − voiture, vitesse relative) utilisent une échelle scalaire (÷ 5120 pour la
  position, ÷ 2300 pour la vitesse), pas l'échelle par axe monde de la balle, qui n'a pas de sens dans un repère tourné.
- §3 : miroir d'orientation R′ = M·R ; les features locales sont invariantes.
- §3 : `prev_actions` est un tableau d'indices int64 ; −1 (après reset) donne un bloc de zéros.
- §3 : un cooldown de pad légèrement négatif (état légitime de RocketGPU avant respawn) est ramené à 0.
- §4 : contrainte testée « boost = 1 ⇒ throttle = 1 » (écart de physique A1, `docs/reports/physics_parity.md`).
- Pads : RocketSim place 4 pads à ±2300 là où RocketGPU les place à ±2302 ; la correspondance entre simulateurs se
  fait par plus proche voisin (tolérance 4 uu).
- §2 (T7b, ADR-004 §3.1) : `car_on_ground` = `air_state == 0` (OnGround) de RocketGPU, comme RLBot, et non `is_on_ground` ; `OBS_SCHEMA_VERSION` reste 1.

## Hypothèses vérifiées par la mesure (T2, 2026-11-11, RocketGPU CPU @f0508a1, `tests/integration/test_env_rocketgpu.py`)
- Pads de chaque arène contigus en mémoire (index e·34 + i) ; gros pads aux indices 28 à 33.
- `pad_cooldown` en secondes (4 s petit pad, 10 s gros pad) ; minimum observé −1/120 s avant respawn.
- Le score monte à chaque tick tant que la balle reste dans le filet (jusqu'à +8 par step) : but = Δscore > 0, compté
  une fois, arène réinitialisée dans le même step.
- Quaternion `rot` en (w, x, y, z) ; dérive de norme ≤ 1,6e-7 sur 400 steps × 64 arènes.
- Drapeaux u8 toujours à 0 ou 1 (`is_demoed` jamais observé à 1 : aucune démolition dans les runs).
- Téléporter une arène laisse les autres bit-exactes (80 steps, voiture en l'air et contact voiture-balle).
- §5 précisé : `Simulator.step` reçoit des indices d'action ; `BatchEnv.obs_dim` est connu dès la construction ;
  `info.final_obs` couvre toutes les arènes finies (le tampon ne garde que les troncatures).
- Débit de `BatchEnv.step` avec les vraies obs (4 096 arènes, 15 s, commit 585b712, thermique non contrôlée) :
  146–166k env-steps/s à p_kick = 1,0 et 116–119k à p_kick = 0,2 ; `build_obs` ≈ 9–10 ms par step (27 à 36 %).

## Reprise exacte mesurée (T5, 2026-11-13, `tests/integration/test_resume_rocketgpu.py`)
- §7 : RocketGPU se restaure **bit à bit**, y compris dans une autre instance, à condition de sauver les tableaux
  publics ET le blob interne (`download_internal_state`) et de recharger en `PreserveInternal`. Le point 6 ci-dessous
  (reprise « modulo reset des arènes ») reste un repli disponible, non utilisé par défaut.
- Reprise exacte de bout en bout (RocketGPU, Adam réel, recouvrement) : 3 itérations == 1 + reprise + 2, en CPU.
  Non testé sur XPU ni CUDA. Blob interne ≈ 20 Ko par arène (≈ 80 Mo à 4 096 arènes : estimé, non mesuré).

## Points à valider par l'utilisateur
1. v0 en 1v1 uniquement, contrat paramétré en `n_cars` pour le 2v2.
2. Obs de 109 dims (structure type « Advanced ») au lieu des 89 du prototype ; aucun normaliseur courant.
3. Miroir par rotation de 180° autour de z, sans miroir d'action.
4. Table de 90 actions dans l'ordre de `bench_e2e.py:45-69` (énumération identique à la référence pulsar2).
5. V(s′) des troncatures évalué par le learner à partir de l'obs finale stockée.
6. Reprise « exacte » : si RocketGPU ne se restaure pas bit à bit, accepter l'exactitude modulo reset des arènes.
