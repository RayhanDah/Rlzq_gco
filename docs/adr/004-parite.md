# ADR-004 — Parité entraînement / RLBot v5 : adaptateur paquet -> `BatchState`

- Statut : **accepté** par l'utilisateur le 2026-11-13, y compris `on_ground` défini comme RLBot (`air_state == 0`).
  Délai d'action d'un tick de RLBot : non modélisé, à traiter en ablation (Phase 4a).
- Date : 2026-11-13
- Décideurs : integration-lead (rédaction) ; tech-director (arbitrage du schéma d'obs, §Décision 3) ; code-reviewer avant merge

## Contexte
Le bot doit recalculer, à partir du `GamePacket` RLBot de chaque tick, les mêmes observations et actions que
l'entraînement (ADR-003 §7). Faits vérifiés (2026-11-13) :
- Paquet Python officiel : **`rlbot` 2.0.0b56** (PyPI, pré-release du 2026-10-02 ; dépôt `RLBot/python-interface`),
  `requires_python >=3.11`, dépend de `psutil==7.*` et `rlbot_flatbuffers~=0.19.0`. La « latest » stable PyPI 1.68.0 est v4.
  Installé dans le venv, groupe `rlbot` de `pyproject.toml`, lock mis à jour. Types : `rlbot_flatbuffers` 0.19.0,
  `.venv/Lib/site-packages/rlbot_flatbuffers/__init__.pyi` (réexporté par `rlbot.flat`, `rlbot/flat/__init__.py:2`).
- `Bot.get_output(packet) -> ControllerState`, appelé sur le **dernier** paquet seulement : les paquets en retard sont
  sautés (`rlbot/managers/bot.py:142-183`, `:311`). `MatchInfo.frame_num` « peut augmenter de plus d'un » (`.pyi:4073`).
- Nexto v5 / rlgym-compat existent (`docs/02_ecosystem.md` §5-6) ; R3 (RLGym 2.0.1 exige numpy < 2) visait cet outil.

Types utiles (`.pyi`, numéros de ligne) : `Physics` (`location`, `rotation` Euler, `velocity`, `angular_velocity`, :2409),
`Rotator` (:2132, radians), `PlayerInfo` (:5580), `BoostPadState` (:1893), `BoostPad` (:3724), `FieldInfo` (:5045),
`GamePacket` (:5500), `ControllerState` (:1994), `AirState` (:47), `MatchPhase` (:1049), `MatchInfo` (:4033).

## Options
| Option | Pour | Contre |
|---|---|---|
| A. Adaptateur maison `src/rlbot_bot/adapter.py` -> `BatchState` + `build_obs` partagé | une seule implémentation d'obs ; testable sans jeu ; numpy 2 ; ~200 lignes | à maintenir si RLBot change (pré-release) |
| B. `rlgym-compat` 2.3.6 | maintenu officiellement, v5 natif, `numpy<3.0` (métadonnées PyPI, non installé) | produit des objets GameState **RLGym v2** : il faudrait un second builder d'obs, donc deux implémentations à garder identiques ; nos obs ne viennent pas de RLGym |

## Mesures
Machine : Zenbook 14 (Core Ultra 5 125H), RocketGPU CPU `vendor/build/rocketgpu-cpu-pgo`, base `98734e5`, 2026-11-13.
- `pytest tests/integration/test_parity.py` : 10 passés en 5,6 s. 150 décisions × 8 arènes × 2 voitures, scénarios
  « kickoff » et « generic » (états génériques de T2), seed 7, 2400 comparaisons par scénario, bleu et orange.
  Écarts max sur l'obs (adaptateur - `build_obs` du sim) : copies directes (balle, pads, pos, vel, boost, drapeaux) **0** ;
  blocs dépendant de l'aller-retour quaternion -> Euler -> quaternion ≤ **7,2e-7** (avant, haut, ang_vel locale,
  balle locale, autre voiture). Tolérances : 1e-6 et 1e-5 (règle `rlbot.md`).
- Quantification des angles à 2π/65536 (source tierce, non vérifiée) : écart ≤ **1,2e-4** sur ces blocs.
- Écart `on_ground` avant correction : 3 puis 4 lignes sur 2400 (0,13 à 0,17 %), toutes en AirState Jumping avec `is_on_ground = 1` ; après correction (§3.1) : 0.
  Relevé séparé (64 arènes, 1200 ticks, actions aléatoires, seed 3, échantillon tous les 8 ticks) : 18 états sur 858 en
  Jumping ont `is_on_ground = 1` (2,1 %), z de 24 à 258 : non séparables par l'altitude.
- Couverture (assertée) : pads inactifs, vol, double saut, flip, fenêtre de dodge. Démolition : jamais observée dans le sim.
- Témoin négatif : roll de signe opposé dans l'adaptateur -> écart sur `soi_haut` > 100 × tolérance (test vert).

## Décision
### 1. Outil : option A (adaptateur maison). `rlgym-compat` écarté : double implémentation d'obs, aucun gain de parité.
R3 est levé côté déploiement : rien n'importe RLGym ; seul `rlbot` (numpy non requis) et numpy 2.5.4 cohabitent.

### 2. Correspondance `BatchState` <- paquet RLBot v5 (D = disponible tel quel, C = calculé, R = reconstruit, I = impossible)
| Champ `BatchState` | Statut | Source RLBot et règle |
|---|---|---|
| `ball_pos/vel/ang_vel` | D | `balls[0].physics` ; uu, uu/s ; `angular_velocity` : unité non documentée dans le stub, lue en rad/s monde (**non vérifié**) |
| `car_pos/vel/ang_vel` | D | `players[i].physics`, même lecture |
| `car_quat` | C | Euler (pitch, yaw, roll) -> matrice ZYX Unreal -> quaternion (`euler_to_quat`, docstring = convention). Précision : ≤ 7,2e-7 (angles f32). Singularité à pitch = ±π/2 (yaw/roll indéterminés) non mesurée |
| `car_boost` | D | `boost` 0-100, borné |
| `car_on_ground` | C, **écart** | `air_state == OnGround`. RLBot donne priorité à Jumping/Dodging ; le sim lit `is_on_ground` (voir Décision 3) |
| `car_has_jumped` | D | `has_jumped` (RocketGPU reporte la même valeur : `rocketgpu.h:241-243`, `env_step.hpp:5635`) |
| `car_has_double_jumped` | D | `has_double_jumped` |
| `car_has_flipped` | D | `has_dodged` |
| `car_is_demoed` | C | `demolished_timeout > 0` (sinon -1 d'après le stub) ; frontière à 0 non testée en jeu |
| `car_air_time_since_jump` | R | pas de champ homonyme, mais `dodge_timeout` = 1,25 - temps depuis la fin du saut tant que le double saut est possible (`env_step.hpp:5640-5645`, stub :5612). Règle : sol ou Jumping -> 0 ; `dodge_timeout >= 0` -> 1,25 - `dodge_timeout` ; sinon 1,25 si `has_jumped`, 0 sinon (chute = flip reset). Erreur : nulle sur les quantités que l'obs utilise (`has_flip_or_jump` identique sur 4800 lignes) ; la valeur brute ne l'est pas hors fenêtre. Pas de suivi d'état nécessaire |
| `pad_active` | D | `GamePacket.boost_pads[j].is_active` |
| `pad_cooldown` | C | inactif : `durée - timer`, durée 10 s (`FieldInfo.is_full_boost`) ou 4 s (stub :1893-1910) ; actif : 0. Erreur : 0 dans la boucle fermée ; précision du timer du jeu non mesurée |
| ordre des pads | C | `FieldInfo` trie par y puis x (:5056), RocketGPU et RocketSim ont chacun leur ordre (`physics_parity.md`) : permutation par position en xy (z ignoré), `match_pads` tol. 4 uu (±2300 / ±2302). Calculée une fois à la connexion |
| `car_team` | C | rang par (équipe, indice) ; équipes équilibrées exigées |
| action précédente | R | suivi côté bot : indice réellement envoyé ; -1 après reset/hors Kickoff-Active. L'obs d'un agent ne lit que sa propre action |
| tick-skip | R | décision si `frame_num - dernière >= 8` ou si le compteur recule ; action tenue sinon (paquets manqués gérés) |
| Aucun champ **impossible** | | tous les champs de l'obs v1 ont une source RLBot ; l'incertitude porte sur la fidélité du jeu, pas sur la disponibilité |

Contrôles : `ControllerState(throttle, steer, pitch, yaw, roll: float ; jump, boost, handbrake, use_item: bool)` (:1994).
Indice -> ligne de la table -> contrôles, et retour exact (90/90, testé) ; boost => plein gaz au sol (testé).

### 3. Points à escalader (aucun fichier `src/obs` ni `src/env` modifié ici)
1. **`on_ground`** : **décision appliquée** (2026-11-13, T7b). `BatchState.car_on_ground` = `air_state == 0` (OnGround) de
   RocketGPU (`rocketgpu.h:324`, calculé en fin de tick `env_step.hpp:5640`), lu par `RocketGPUSimulator` dans un tampon
   u8 rempli sur place après step, reset, `set_states` et `load_state_dict` (une vue directe est impossible : `air_state`
   vaut 0 à 4 ; coût mesuré ~1,4 µs pour 256 arènes, ~2,3 µs pour 1024, contre ~1,6 ms et ~2,7 ms par step). L'écriture de
   `is_on_ground` dans `set_states` et `state_dict` sont inchangés ; juste après `set_states`, `car_on_ground` vaut la valeur
   injectée (air_state n'est recalculé qu'au tick suivant). Mesure après correction (`test_parity.py`, seed 7, 150
   décisions, 8 arènes, 2 voitures) : **0 ligne divergente sur 2400** dans chaque scénario (avant : 3 puis 4), colonne de
   l'autre voiture comprise ; le test est strict. `OBS_SCHEMA_VERSION` reste **1** : aucun checkpoint n'existe encore.
2. **Délai d'action d'un tick de RLBot** (`docs/02_ecosystem.md:191`, `rlbot_delay`) : `BatchEnv` applique l'action sans
   délai. Non modélisé ici ; effet à mesurer en Phase 5 ; à arbitrer par infra-perf-lead / rl-algo-lead (délai à l'entraînement ?).
3. Si une mesure Phase 5 montre un champ faux (ex. `angular_velocity`), le retirer de l'obs = nouveau `OBS_SCHEMA_VERSION`.

## Ce que le test prouve et ne prouve pas
- **Prouve** : l'adaptateur est cohérent avec notre lecture écrite des conventions RLBot : `export_packet` (test) est écrit
  à partir du stub et lit des champs `air_state`, `dodge_timeout`, `demo_respawn_timer` que RocketGPU produit pour RLBot ;
  il n'utilise pas l'adaptateur. Il prouve aussi : tick-skip, action précédente, permutation des pads, aller-retour actions.
- **Ne prouve pas** : que le jeu réel émet ces champs avec ces conventions (ordre Euler, signe du roll, unité de
  `angular_velocity`, ordre et z réels des pads, précision du timer, quantification 16 bits, `air_state` pendant un saut,
  latence d'action, démolitions). Les deux côtés sont écrits par la même équipe : une erreur de lecture commune passerait.
  Mesure requise en Phase 5, jeu lancé par RLBot (pas de dumper mémoire, EAC).

## Conséquences
- `src/rlbot_bot/{__init__,adapter}.py` : l'adaptateur n'importe RLBot que pour fabriquer un `ControllerState` (import
  paresseux). Reste à faire (hors tâche) : classe `Bot`, `bot.toml`, chargement du modèle, mesure de latence < 8,33 ms.
- Révision si : un champ du jeu diffère en Phase 5 ; RLBot 2.0.0b57+ change `flat` (pré-release, ruptures possibles).
