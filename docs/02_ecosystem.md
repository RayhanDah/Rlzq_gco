# État de l'écosystème Rocket League RL — novembre 2026 (sim-engineer + rlbot-engineer)

Date de recherche : 2026-11-10. Cible : bot 1v1, PPO + rewards en Python, reward GCO (goal/concede only),
laptop Win11 / Core Ultra 5 125H / 15,4 Go RAM / Intel Arc iGPU (pas de CUDA) / Python 3.12.6 / numpy 2.5.3.

Méthode : uniquement des sources primaires lues pendant cette session (API GitHub `api.github.com` en lecture,
`raw.githubusercontent.com`, JSON PyPI `pypi.org/pypi/<pkg>/json`, docs officielles). Rien n'a été installé ni exécuté.
Légende : **[V]** = vérifié sur l'URL citée ; **[D]** = déduit de sources vérifiées (raisonnement explicité) ;
**[A]** = attente/estimation de notre part, NON mesurée ; **non vérifié** = aucune source primaire trouvée.

---

## 1. RLGym v2 (repo `RLGym/rlgym`, ex-`lucas-emery/rocket-league-gym`)

| Champ | Valeur |
|---|---|
| Version | `rlgym` **2.0.1** (méta-paquet), `rlgym-rocket-league` **2.0.1**, `rlgym-api` **2.0.0** [V] |
| Date | rlgym/rlgym-rocket-league 2.0.1 : 2025-04-27 ; rlgym-api 2.0.0 : 2025-01-06 [V]. Dernier commit `main` : 2026-05-16 « Bump version requirements (#90) » — **non publié sur PyPI** [V] |
| URL | https://pypi.org/pypi/rlgym/json · https://pypi.org/pypi/rlgym-rocket-league/json · https://pypi.org/pypi/rlgym-api/json · https://github.com/RLGym/rlgym (l'API GitHub redirige `lucas-emery/rocket-league-gym` → `RLGym/rlgym`) |
| État | Maintenu à bas régime (3 commits en 2026). Le jeu réel n'est plus un backend : release notes 2.0.0 « Removed rocket-league game package » → RocketSim est le seul TransitionEngine RL fourni [V] (https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/version.py) |

**Abstractions v2 (signatures lues dans `rlgym/api/config/*.py` et `rlgym/api/rlgym.py`)** [V]
- `RLGym(state_mutator, obs_builder, action_parser, reward_fn, transition_engine, termination_cond=None, truncation_cond=None, shared_info_provider=None, renderer=None)` ; `reset() -> Dict[AgentID, Obs]` ; `step(actions)` renvoie `(obs, rewards, terminated, truncated)` (dicts par agent, cf. `example.py`) ; aussi `set_state`, `agents`, `action_spaces`, `observation_spaces`, `render`, `close`.
- `TransitionEngine` : `agents`, `max_num_agents`, `state`, `config` (get/set), `step(actions, shared_info) -> State`, `create_base_state()`, `set_state(desired_state, shared_info)`, `close()`.
- `StateMutator.apply(state, shared_info)`.
- `ObsBuilder` : `get_obs_space(agent)`, `reset(agents, initial_state, shared_info)`, `build_obs(agents, state, shared_info)`.
- `ActionParser` : `get_action_space(agent)`, `reset(...)`, `parse_actions(actions, state, shared_info)`.
- `RewardFunction` : `reset(agents, initial_state, shared_info)`, `get_rewards(agents, state, is_terminated, is_truncated, shared_info) -> Dict[AgentID, Reward]`.
- `DoneCondition` : `reset(...)`, `is_done(agents, state, shared_info)`. `SharedInfoProvider` : `create`, `set_state`, `step`.
- Implémentations RL fournies : `RocketSimEngine(rlbot_delay=True, game_mode=GameMode.SOCCAR)` (rlbot_delay « emulates RLBot's 1-tick action delay, default is True »), `LookupTableAction` (table de 90 actions — comptage [D] à partir du code), `RepeatAction(parser, repeats=8)`, `DefaultObs`, `GoalReward`, `TouchReward`, `CombinedReward`, `GoalCondition`, `NoTouchTimeoutCondition(timeout_seconds)`, `TimeoutCondition(timeout_seconds)`, `AnyCondition/AllCondition`, `FixedTeamSizeMutator`, `KickoffMutator`, `MutatorSequence`, `RLViserRenderer` (dans `rlgym.rocket_league.rlviser`). Sources : https://raw.githubusercontent.com/RLGym/rlgym/main/example.py , https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/sim/rocketsim_engine.py
- Collision meshes : **embarqués dans le repo** (`rlgym/rocket_league/sim/collision_meshes/{soccar(16),hoops(12),dropshot(2)}/*.cmf`) et déclarés en `package_data` ; `RocketSimEngine` fait `rsim.init(<dossier du paquet>/collision_meshes)` [V]. Leur présence effective dans le sdist PyPI 2.0.1 (78 840 octets) : non vérifié.

**rlgym-sim vs rlgym[rl-sim]**
- `rlgym-sim` (https://github.com/AechPro/rocket-league-gym-sim) : API **v1** (`rlgym_sim.make()`), v1.2.6, dernier push 2024-09-09, README : « TEMPORARY STOP-GAP… while RLGym 2.0 is in development », exige de dumper les meshes soi-même, `numpy>=1.19, <2.0` [V]. Pas sur PyPI. → **obsolète pour nous**.
- L'extra s'appelle `rl-sim` (et `rl`, `rl-rlviser`, `all`) ; « rl-rocketsim » n'existe pas [V] (README + `setup_rlgym.py`).

**Compatibilité** : Python — rlgym 2.0.1 `requires_python >=3.7` [V] (main : `>=3.9`) → 3.12 OK (pur Python).
numpy 2 — **PROBLÈME** : le diff du commit 2fc8d43 (2026-05-16) montre que l'état précédent, donc celui publié en 2.0.1
(aucun commit touchant le setup entre 2025-04-27 et 2026-05-16), était `numpy >=1.19,<2`, `rocketsim >=2.1`, `rlviser-py ==0.6.*` ;
main passe à `numpy >=1.19,<3.0.0`, `rocketsim >=2.0.0,<3.0.0` [D] (https://api.github.com/repos/RLGym/rlgym/commits/2fc8d43).
Win11 / no-CUDA : sans objet (pur Python + RocketSim CPU).

**Risques** : (1) `pip install rlgym[rl-sim]` depuis PyPI ⇒ numpy<2 imposé (downgrade de 2.5.3 dans le venv). Contournements :
build depuis `main` (procédure en 2 temps : `python setup_rocket_league.py` génère `setup.json`, puis `setup.py` — cf. `release.bat`), ou
`--no-deps` + vendoring. (2) `rlviser-py 0.6.14` exige `numpy>=2.0.0` → conflit avec 2.0.1 ; le résolveur reculera sur une 0.6.x plus ancienne [A].
(3) Release sans activité depuis 18 mois ; numpy<3 seulement sur main.

---

## 2. rlgym-ppo (https://github.com/AechPro/rlgym-ppo) — et son successeur de fait rlgym-learn

| Champ | Valeur |
|---|---|
| Version | `__version__ = "1.3.13"` (setup.py), pas de tag, pas sur PyPI (install `pip install git+https://github.com/AechPro/rlgym-ppo`) [V] |
| Date | Dernier commit 2025-01-06 « Fix v2 wrapper bug » ; push 2025-01-06 [V] |
| URL | https://raw.githubusercontent.com/AechPro/rlgym-ppo/main/setup.py · .../requirements.txt · https://api.github.com/repos/AechPro/rlgym-ppo/issues |
| État | Gelé depuis ~22 mois. Tuto officiel rlgym.org l'utilise encore (`from rlgym_ppo import Learner`, `RLGymV2GymWrapper`) [V] https://rlgym.org/Rocket%20League/training_an_agent/ |

- **Support RLGym v2** : oui via `rlgym_ppo/util/rlgym_v2_gym_wrapper.py` (`RLGymV2GymWrapper`) + `rlgym_v2_example.py` [V].
  MAIS l'exemple v2 du repo est désynchronisé de rlgym 2.0.x : `NoTouchTimeoutCondition(timeout=...)` alors que 2.0.0 a renommé en `timeout_seconds`,
  et `from rlgym.rocket_league.sim import RocketSimEngine, RLViserRenderer` alors que `sim/__init__.py` n'exporte que `RocketSimEngine` [V] → l'exemple ne tourne pas tel quel [D]. Le tuto rlgym.org, lui, utilise `timeout_seconds` [V].
- **Architecture multi-process** [V] (`batched_agents/batched_agent_manager.py`, `batched_agent.py`) : 1 process par env, start method `forkserver` si dispo sinon `spawn` (⇒ `spawn` sous Windows) ; **actions envoyées par socket UDP localhost** (`socket.SOCK_DGRAM` bindé sur 127.0.0.1) ; **obs/rewards retour par mémoire partagée** `multiprocessing.sharedctypes.RawArray('f', n_proc * shm_size)` (`shm_buffer_size=8192`) ; inférence batchée côté learner (`min_inference_size`).
- **Learner** : `n_proc=8`, `ts_per_iteration=50000`, `ppo_batch_size=50000`, `policy_layer_sizes=(256,256,256)`, `device="auto"` → `cuda:0` si dispo sinon `cpu` ; tout autre string passé tel quel (donc `"xpu"` possible syntaxiquement — fonctionnement non vérifié) [V] `learner.py`. `wandb` importé inconditionnellement en tête de `learner.py` [V].
- **SPS rapporté** : aucun chiffre dans le README rlgym-ppo. Chiffre tiers du même écosystème : « upwards of 70ksps … in C++, vs 10k in Python » (i5-11400 + 3060 Ti) — README RLGymPPO_CPP [V] https://github.com/ZealanL/RLGymPPO_CPP . RLGymPPO_CPP liste « Support for collection during learn » comme ajout ⇒ rlgym-ppo alterne collecte/apprentissage [D].
- **Windows** : pas de mention explicite ; chemin `spawn` + `kbhit` présents [V]. **CUDA vs CPU** : README « If you would like to use a GPU install PyTorch with CUDA » (optionnel) [V].
- **Issues connues** [V] : #16 « Multi-Discrete Policy NaNs on CPU » (ouverte depuis 2023), #33 « checkpoint_load_folder=latest can resume from an older run » (2026-10-12, ouverte), #30 sauvegarde au timestep_limit (ouverte).
- **numpy 2** : `requirements.txt` épingle `numpy>1.21, <2.0` (setup.py : `numpy>1.21` sans borne) et dépend de `gym>0.23` + `gym-notices==0.0.8` [V]. `gym` est figé à 0.26.2 (2022-10-04) [V] https://pypi.org/pypi/gym/json ; la PR « np.bool8 → np.bool_ » est toujours ouverte [V] https://github.com/openai/gym/pull/3258 . rlgym-ppo n'utilise que `gym.spaces` [V] ; fonctionnement réel sous numpy 2.5 : non vérifié.

**rlgym-learn (https://github.com/JPK314/rlgym-learn) + rlgym-learn-algos** — à considérer comme remplaçant :
- `rlgym-learn` **2.0.0** (2026-08-29), `requires_python >=3.10,<=3.14`, **wheels cp312 win_amd64** publiées ; deps `rlgym>=2.0.0`, `pydantic>=2.13`, extras `rl` (`rlgym[rl]`), `tensor` (`torch>1.13`) [V] https://pypi.org/pypi/rlgym-learn/json
- `rlgym-learn-algos` **0.4.2** (2026-09-06), wheels cp312 win_amd64, deps `torch>1.13`, `numpy>1.21`, `rlgym-learn>=2.0.0` [V] https://pypi.org/pypi/rlgym-learn-algos/json
- README : « Full support for all generics of the RLGym v2 API », « Fast parallelization of environments using Rust and shared memory », « Provided optimized PPO implementation », « Multi-platform (Windows, Linux) » [V]. PPO = classes Python/torch (`PPOAgentControllerConfigModel`, `PPOLearnerConfigModel`, `GAETrajectoryProcessorConfigModel`) [V] `quick_start_guide.py`. Les rewards restent des `RewardFunction` RLGym en Python (exécutées dans les process env) [D].
- Breaking changes nombreux en 2.0.0 (CHANGELOG) ; doc rlgym.org « upcoming project that is not finished yet » [V]. `quick_start_guide.py` met `device="cuda:0"` — à changer [V].
- **Risque** : hérite de la contrainte numpy<2 de `rlgym[rl]` 2.0.1 PyPI [D].

---

## 3. RocketSim (C++ v2, Rust « v3 ») + bindings Python

| Composant | Version / date | URL | État |
|---|---|---|---|
| RocketSim C++ (branche `main`) | **v2.2.1**, release 2026-04-19 ; dernier commit main 2026-07-14 (README) [V] | https://github.com/ZealanL/RocketSim/releases | Maintenance (bugfix) |
| RocketSim **v3 = réécriture Rust** (branche `v3-rust`) | crate `rocketsim` **0.3.5** (crates.io 2026-11-03 ; 0.3.0 le 2026-10-19 ; 0.2.x en oct. 2026) ; releases GitHub v0.3.1→v0.3.5 (2026-10-21→11-03, binaires `rocketsim_viewer` seulement) [V] | https://raw.githubusercontent.com/ZealanL/RocketSim/v3-rust/README.md · https://crates.io/api/v1/crates/rocketsim | Très actif (10 commits du 1er au 3 nov.), API instable (MIGRATING 0.2→0.3) |
| Binding Python `rocketsim` (pip, module `RocketSim`) | **2.2.1**, 2026-04-24, `requires_python >=3.9`, wheels `cp36-abi3` dont **win_amd64** ; deps `cmeel`, `numpy` ; extra build `numpy<3,>=2.0` [V] | https://pypi.org/pypi/rocketsim/json · https://github.com/mtheall/RocketSim (branche `python-dev`, merge tag v2.2.1 le 2026-04-24) | Suit le C++ v2 |
| `rlviser-py` | 0.6.14, 2026-06-11, `numpy>=2.0.0`, `rocketsim>=2.2.1`, wheel abi3 win_amd64 [V] | https://pypi.org/pypi/rlviser-py/json | OK |

- **v3 Rust** : « RocketSim simulates Rocket League car and ball physics in Rust » ; « The core simulation does not need a C++ compiler » ; MIT ; modes Soccar/Hoops/Dropshot/Snowday/Heatseeker ; `step_tick` = 1/120 s ; modes mémoire `Balanced` (≈133 KiB/arène 1v1), `Light`, `Heavy` [V]. **Aucun binding Python dans l'arbre v3** (crates : `rocketsim`, `rocketsim_vis`, `rocketsim_viewer`, `rocketsim_test`) [V] → le paquet pip reste C++ v2. RLGym main borne `rocketsim <3.0.0` [V].
- **Meshes** : C++ v2 — README « Use RLArenaCollisionDumper to dump all of Rocket League's arena collision meshes » [V]. v3 — « The crates do not include these files », attend `collision_meshes/soccar/` (16 .cmf), `hoops/` (12), `dropshot/` (2) [V]. Mêmes cardinalités que les meshes livrés par RLGym ⇒ probablement réutilisables [D/A], compatibilité de format v3 non vérifiée.
  RLArenaCollisionDumper « Patches machine code » dans le process RL en cours d'exécution [V] https://github.com/ZealanL/RLArenaCollisionDumper → à éviter (EAC depuis 2026-04-28, voir §5) ; utiliser les meshes de RLGym.
- **Ticks/s rapportés** : C++ README main : « single thread … two cars … around 20 minutes of game time every second » (≈144 000 ticks/s [D]) ; tableau historique 2v2, i5-11400, MSVC, Win10 : v2.1.0 = **114 481 tps** single-thread [V]. v3 : le README ne donne que des comparaisons relatives entre modes mémoire [V]. Chiffres **tiers** (repo créé le 2026-10-27, 12★) : RocketSim v3 0.3.5 = **2,91–3,79 M arena-ticks/s** sur Ryzen 7 9800X3D 14 threads, charge mixte 55 % 1v1/27 % 2v2/18 % 3v3 [V-tiers] https://github.com/thecosmicskye/RocketGPU
- **Déterminisme** : v3 README « Use the same initial state, random seed, and inputs when you need repeatable runs » [V] (répétabilité même machine ; cross-machine non vérifié). C++ v2 : non vérifié. Précision vs jeu réel (tiers) : v3 0.3.5 = 4,32 % de body-ticks bit-exact, 86,15 % « segment pass » sur 120 ticks [V-tiers] https://github.com/thecosmicskye/RocketGPU-fullstate-suite . README C++ : « tiny errors will accumulate over time » [V].
- **PerfectSim (équipe Titan)** : aucune trace publique trouvée (recherche web) → **non vérifié / pas public à notre connaissance**. À surveiller à la place : **RocketGPU** (MIT, C++17, build CPU sans CUDA possible, binding Python `rocketgpu_py` + paquet « RocketSim-style `rocketsim` », revendique 100 % bit-exact ; meshes à dumper ; CMake 3.21+) [V-tiers, non reproduit].
- **Compat** : Win11 + Py3.12 + numpy2 + no-CUDA = OK pour `rocketsim` 2.2.1 (wheel abi3 win_amd64, build numpy 2) [V]. v3 Rust : utilisable seulement via nos propres bindings PyO3 (toolchain Rust absente de la machine) [D].
- **Risques** : divergence v2 (pip) / v3 (Rust) en cours ; RLGym ne suit pas v3 ; meshes v3 non garantis compatibles.

---

## 4. GigaLearnCPP, RLGymPPO_CPP, pulsar2

**GigaLearnCPP** — https://github.com/ZealanL/GigaLearnCPP-Leak
| Champ | Valeur |
|---|---|
| Version/date | Pas de version ; dernier push 2025-08-23 ; **repo archivé** (`archived: true`) [V] |
| État | README : « This is the version that cheaters decided to leak », publié par l'auteur « for anyone to use » ; « no installation guide for now » [V]. Listé sur le wiki RLBot v5 [V] |
| Build | CMake ≥3.8, **C++20**, `find_package(Torch REQUIRED)` (libtorch obligatoire), CUDA **optionnel** (`if (TORCH_CUDA_LIBRARIES) … RG_CUDA_SUPPORT`), `LearnerDeviceType { AUTO, CPU, GPU_CUDA }` ⇒ **CPU-only possible** ; blocs `if (MSVC)` (copie DLL Python/libtorch) ⇒ MSVC visé ; pybind11 en sous-module, utilisé en `pybind11::embed` pour les scripts Python `metric_receiver.py` (wandb) / `render_receiver.py` [V] CMakeLists.txt, LearnerConfig.h |
| Rewards | Classes C++ (`GetReward(const Player&, const GameState&, bool isFinal) override`) ; **pas de binding pour écrire des rewards en Python** trouvé [V/non vérifié pour les forks] |
| Defaults | `numGames=300`, `tickSkip=8`, `actionDelay=7` ; exemple : `NoTouchCondition(10)` (secondes, `deltaTime`), `KickoffState`, `AdvancedObs`, layers 256×3 + shared head 256×2 [V] ExampleMain.cpp |
| RLBot | sous-module `RLBotCPP` = https://github.com/kipje13/RLBotCPP (interface **v4**) [V] → pas v5 natif [D] |
| SPS | Relatif seulement : « Collection … around 2x faster … than RLGymPPO-CPP, and around 10x faster than RLGym-PPO (on my machine) » [V] |

Guide communautaire (secondaire) : Windows + **libtorch CPU** + VS Build Tools 2026 + CMake + Python 3.11 [V] https://github.com/PiggyVevo/GigalearnGuide

**RLGymPPO_CPP** — https://github.com/ZealanL/RLGymPPO_CPP : dernier push 2024-10-27, abandonné (« which I have stopped developing », README GigaLearn) ; libtorch CUDA 11.8 **ou CPU** ; ~70k SPS C++ vs 10k Python (i5-11400 + 3060 Ti) ; threads au lieu de process [V].

**pulsar2** — https://github.com/lucamignatti/pulsar2 : « gigalearn fork implementing pulsar », C++, branche par défaut `2.1`, branches `2.2`, `2.6`, `3.0-psd`, `4.0-team`, `main` ; push 2026-09-23 ; README = celui de GigaLearn inchangé ; commits de recherche (ex. 2026-07-31 « the geometric critic as a permanent 4th rung ») [V]. Pas de doc, pas de chiffres → SPS / build : non vérifié.

**Compat Win11 / no-CUDA** : CPU-only supporté par le code [V]. **Contrainte GCO violée** : PPO et rewards en C++ ⇒ incompatible avec « PPO + rewards MUST be Python » [D].
**Risques machine** : pas de cmake sur le PATH ; VS Build Tools **2019** (v142) — Microsoft : « the linker should only work with inputs built by build tools that are the same version (or earlier) as itself » [V] https://learn.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017 ⇒ si libtorch 2.15 est compilé avec v143+, linker avec VS2019 n'est pas supporté (toolset réel de libtorch : non vérifié). libtorch CPU Windows 2.15.0 existe (`libtorch-win-shared-with-deps-2.15.0+cpu.zip`, 204 Mo, HTTP 200) [V] ; variante `+xpu` au même chemin : 404 [V].

---

## 5. RLBot v5

| Champ | Valeur |
|---|---|
| Paquet Python | **`rlbot` 2.0.0b56** (pré-release, 2026-10-02), `requires_python >=3.11`, deps `psutil==7.*`, `rlbot_flatbuffers~=0.19.0` (0.19.0, wheel abi3 win_amd64) ; la « latest » stable PyPI 1.68.0 est **v4** ⇒ installer `rlbot==2.0.0b56` / `--pre` [V] https://pypi.org/pypi/rlbot/2.0.0b56/json · `__version__ = "2.0.0-beta.56"` https://raw.githubusercontent.com/RLBot/python-interface/master/rlbot/version.py |
| Serveur | RLBot/core **v5.0.0-rc18** (2026-10-22) [V] https://github.com/RLBot/core/releases |
| État | Wiki : « RLBot v5 is in pre-release! », interfaces « may see breaking changes » [V] https://wiki.rlbot.org/v5/framework/v5/ |

- **API bot** [V] (`rlbot/managers/bot.py`) : `from rlbot.managers import Bot` ; sous-classer, surcharger `get_output(self, packet: flat.GamePacket) -> flat.ControllerState`, option `initialize()`, `retire()` ; `Bot.__init__(default_agent_id=None)` ; `run(wants_match_communications=True, wants_ball_predictions=True)`. Boucle : vide la file de messages puis ne traite **que le dernier GamePacket** (les paquets en retard sont sautés). Aussi `set_game_state`, `set_loadout`, `send_match_comm`. Port TCP par défaut 23234 (`RLBOT_SERVER_PORT`), `RLBOT_AGENT_ID` [V] https://wiki.rlbot.org/v5/framework/sockets-specification/
- **bot.toml** [V] https://wiki.rlbot.org/v5/botmaking/config-files/ : `[settings]` `agent_id` (« author/bot-name »), `name`, `loadout_file`, `root_dir`, `run_command`, `run_command_linux`, `hivemind`, `logo_file` ; `[details]` (GUI : description, fun_fact, source_link, developer, language, tags). Fichiers `bot.toml` ou `*.bot.toml`, schéma `#:schema https://rlbot.org/schemas/agent.json`.
- **match.toml** [V] même page : `[rlbot]` `launcher` = `"Steam"` (défaut) | `"Epic"` | `"Custom"` (+`launcher_arg` `"legendary"`/`"heroic"`) | `NoLaunch`, `auto_start_agents`, `wait_for_agents` ; `[match]` `game_mode`, `game_map_upk` (« The Epic version supports only Psyonix maps »), `skip_replays`, `start_without_countdown`, `existing_match_behavior`, `enable_rendering`, `enable_state_setting`… ; `[mutators]` `match_length`, `max_score`… ; `[[cars]]` `team`, `type` = `"RLBot"`|`"Human"`|`"Psyonix"`, `skill` (Psyonix), `config_file`.
- **Lancer un match** : GUI RLBot5 (« Add » → `bot.toml` → « Start Match ») [V] https://github.com/Martico2432/Rlgym-v2-to-rlbot-v5 ; ou Python `with MatchManager(rlbot_server_path) as man: man.start_match("match.toml")` [V] https://raw.githubusercontent.com/RLBot/python-interface/master/tests/run_match.py . Humain vs bot : `[[cars]] type = "Human"` + `[[cars]] config_file = "necto/bot.toml"` [V] `tests/human_vs_necto.toml`. Bot vs bot : deux `[[cars]]` RLBot ; vs IA Psyonix : `type="Psyonix"` + `skill`.
- **Epic + EAC** : EAC sur RL PC depuis le 2026-04-28 (source rocketleague.com via recherche, page 403 en fetch) ; RLBot core rc15/PR #172 « Steam no EAC launch + fix Epic launch » — Epic « should launch much more reliably » mais « doesn't load player settings beyond the username » [V] https://github.com/RLBot/core/pull/172 ; rc16 « Add `-noeac` arg for heroic/legendary » [V].
- **Botpack / Nexto v5** : port v5 présent — `Botpack/NectoFamily` (https://github.com/VirxEC/NectoFamily, « port the bot to RLBot v5 for use in the v5 botpack ») [V] https://raw.githubusercontent.com/RLBot/botpack/master/.gitmodules ; aussi `tests/nexto/` dans python-interface (bot.toml `agent_id = "Rlgym/Nexto"`, `tick_skip = 8`, modèle TorchScript `nexto-model.pt`) [V].
- **torch / ONNX dans un bot packagé** : page botpack — dépôt open source, « only use public dependencies », `requirements.txt` pour Python, build par **Bob** (Docker → binaires Windows/Linux), « Submissions that include pre-built binaries will not be accepted », plafond de niveau « at most as strong as Nexto » ; **aucune règle explicite torch/ONNX** [V] https://wiki.rlbot.org/v5/community/rlbot-pack/ . Précédent : NectoFamily = `torch==2.4.1+cpu`, `numpy==2.*`, `rlbot>=2.0.0.beta`, rlgym_compat, PyInstaller [V]. ONNX Runtime : non vérifié (rien d'interdit trouvé).
- **Budget latence** : « All bots are expected to run at 120hz » [V] ⇒ **8,33 ms/tick** [D] ; le serveur expose des métriques de « missed responses by bots » [V]. Avec tick_skip 8 on n'infère qu'1 tick sur 8, mais `get_output` doit rester < 8,33 ms à chaque tick d'inférence [D].
- **RLBot Championship** : édition **2025** publique [V] https://docs.google.com/document/d/1g5hNT_B7Sqp8HWo-wmCBc7RRgJM-O6JcQXpzhQVU_wc : deux brackets (hardcoded / **ML**), **2v2** (la soumission joue avec une copie d'elle-même), 5 min, mercy rule +6, « The bots will be run at 120 Hz », RLBot v5, bleu+orange, battre Psyonix Pro, Python = `requirements.txt` suffit, GigaLearn : « dynamically linking libtorch_cpu.dll » recommandé, mise à jour de policy autorisée en cours de tournoi. **« RLBC 2026 » : aucune règle publique trouvée → non vérifié.** ⇒ un bot 1v1-only ne serait pas éligible au format 2025 [D].

---

## 6. rlgym-compat

| Champ | Valeur |
|---|---|
| Version | **2.3.6** (PyPI 2026-07-21 ; repo officiel `RLGym/rlgym-compat` push 2026-07-21 ; fork `JPK314/rlgym-compat` push 2026-05-02) [V] https://pypi.org/pypi/rlgym-compat/json |
| Compat | `python_requires >=3.11`, `numpy<3.0` (numpy 2 OK) [V] setup.py |
| v5 | **Oui** : `game_state.py` importe `from rlbot.flat import …` (API v5) ; fournit `GameState.create_compat_game_state(...)` / `update(...)` (objets RLGym **v2**) et `v1_game_state.V1GameState` (utilisé par Nexto v5) [V] |
| But | « A library of RLBot compatibility objects for RLGym Agents » : reconstruire `GameState`/`Car`/`PhysicsObject` RLGym depuis le `GamePacket` RLBot pour réutiliser obs builder/action parser à l'inférence [V] |

Release notes : 2.3.4 « fix dtypes in physics objects to be consistent with rocketsim engine behavior in rlgym v2 » ; 2.3.1 `ball_touch_ticks` ; `SimExtraInfo` (RocketSim optionnel) [V].
Risques : README minimal qui renvoie vers `RLGymExampleBot`, lui encore **v4** (`rlbot.cfg`, `appearance.cfg`) [V] ; template v5 tiers : https://github.com/Martico2432/Rlgym-v2-to-rlbot-v5 (push 2026-01-07) [V].

---

## 7. Runtimes d'inférence/entraînement sur CE hardware

| Runtime | Version / date | Py3.12 Win wheel | Statut pour Arc iGPU Meteor Lake | URL |
|---|---|---|---|---|
| PyTorch CPU | **2.15.0** (2026-10-28), `>=3.11` ; sur Windows les deps CUDA ne s'appliquent pas (`platform_system == "Linux"`) | `torch-2.15.0-cp312-cp312-win_amd64.whl` [V] | CPU | https://pypi.org/pypi/torch/json |
| PyTorch **XPU** | 2.15.0+xpu (aussi 2.13/2.14.x) | `torch-2.15.0+xpu-cp312-cp312-win_amd64.whl` [V] | Doc 2.15 : liste validée inclut « Intel Core Ultra Processors with Intel Arc Graphics (CodeName: Meteor Lake-H) », OS Windows 11 ; « Intel GPUs support (Prototype) is ready from PyTorch 2.5 » ; `torch.compile` Windows depuis 2.7 ; install `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu` [V] | https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html · https://download.pytorch.org/whl/xpu/torch/ |
| ONNX Runtime CPU | **1.31.1** (2026-10-26), `>=3.11` | cp312 win_amd64 [V] | CPU | https://pypi.org/pypi/onnxruntime/json |
| ORT DirectML | 1.24.4 (2026-03-17) — en retard de 7 mineures | cp312 win_amd64 [V] | « DirectML is in sustained engineering » (nouveautés → WinML) ; mem pattern off, exécution séquentielle, 1 thread `Run` par session [V] | https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html |
| ORT OpenVINO | `onnxruntime-openvino` 1.24.1 (2026-02-26, EP « legacy » marqué déprécié) ; plugin `onnxruntime-ep-openvino` 1.8.1 (2026-10-26, `py3-none-win_amd64`) | oui [V] | devices `CPU, NPU, GPU…` (« Intel integrated and discrete graphics ») ; Windows : `setupvars.bat` de l'installeur OpenVINO requis selon la doc [V] | https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html |
| OpenVINO | 2026.4.1 (2026-10-01) | cp312 win_amd64 [V] | GPU/NPU Intel | https://pypi.org/pypi/openvino/json |
| IPEX | 2.8.0 (2025-08-12), plus de release depuis | — | statut (retraite ?) non vérifié | https://pypi.org/pypi/intel-extension-for-pytorch/json |

- CPU : Core Ultra 5 125H = 14 cœurs (4P + 8E + 2 LP-E) / 18 threads, Meteor Lake, base 28 W / turbo max 115 W, NPU 11 TOPS ; **la marque « Arc » exige ≥16 Go en dual-channel** (sinon « Intel Graphics ») — Intel ARK via résultat de recherche (page non fetchée) https://www.intel.com/content/www/us/en/products/sku/236848/intel-core-ultra-5-processor-125h-18m-cache-up-to-4-50-ghz/specifications.html . 15,4 Go visibles ≈ 16 Go installés [A] ; vérifier que Windows nomme bien le GPU « Intel Arc Graphics ».
- **Attente [A]** : pour un MLP PPO typique (≈256–1024 neurones, batch inférence 10–100), le dispatch XPU/DirectML coûtera plus que le calcul ; CPU (torch ou ORT) pour l'inférence de collecte et le bot RLBot. XPU éventuellement utile pour l'update PPO (minibatchs 50k) — à benchmarker, le support Windows reste « Prototype ».
- Risques : XPU = prototype + driver Intel requis ; mémoire iGPU partagée avec les 15,4 Go ; IPEX non pertinent (support XPU désormais upstream dans torch).

---

## 8. Leçons des bots communautaires (Necto / Nexto) et reward sparse

- **Necto/Nexto** (https://github.com/Rolv-Arild/Necto, push 2024-09-17, non maintenu) [V] : V1 Necto ≈ Diamond ; V2 **Nexto ≈ GC1** en 1v1/2v2/3v3 (« top 0.12%, 0.95%, 0.46% ») ; V3 Tecko annulé ; entraînés avec RLGym (v1 ; « several games playing at super speeds » ⇒ instances du jeu réel [D]) et un système distribué (rocket-learn + `RedisRolloutGenerator`, cf. `training/learner.py`) ; conseil : « Make sure fps is set to 120 and VSync is turned off ».
- **Archi/obs/actions** (code `training/`, version Tecko) [V] : acteur/critique `EARLPerceiver(256, 4, 8, 1, query_features=36, key_value_features=25+30)` (attention, nombre de joueurs variable), sortie discrète **90 actions** (lookup table identique à `LookupTableAction` RLGym), PPO : `n_steps=1_000_000`, `batch_size=100_000`, `minibatch_size=10_000`, `epochs=30`, `gamma=0.995`, `ent_coef=0.01`, lr 1e-4. Inférence bot : TorchScript (`torch.jit.load`), `tick_skip=8`, kickoffs **codés en dur** (`KICKOFF_CONTROLS`), `beta` = déterminisme/échantillonnage [V] `tests/nexto/bot.py`, `agent.py`.
- **Steps nécessaires** : aucun chiffre officiel Nexto [non vérifié]. Lucy-SKG (arXiv 2305.15801, 2023) : tous les modèles comparés entraînés à 1B steps, Necto aussi à 2B ; Lucy-SKG dépasse le Necto-1B dès 200M steps (≈5× plus sample-efficient) ; ~4000 FPS (obs simplifiée) / ~2000 FPS sur « our hardware » ; rewards **shapées** (potential-based, « KRC ») ; évaluation 2v2, 300 matchs « first goal » — c'est un Nexto **ré-entraîné 1B steps**, pas la release publique [V] https://ar5iv.labs.arxiv.org/html/2305.15801
- **Sim-to-real** : RocketSim « tiny errors will accumulate over time » [V] ; RLGym `rlbot_delay=True` et GigaLearn `actionDelay = tickSkip - 1` (« Normal value in other RLGym frameworks ») pour reproduire le délai d'action RLBot [V] ; Pleines et al. 2022 (arXiv 2205.05061) : sim → jeu, gardien ≈100 % d'arrêts, attaquant ≈75 % de buts, simulation « to some extent inaccurate » [V].
- **GCO / reward purement sparse** : aucune writeup publique RL/Rocket League évaluant un reward goal/concede seul trouvée → **non vérifié**. Lucy-SKG motive explicitement le shaping par la rareté des buts mais ne teste pas le goal-only [V]. Tous les exemples officiels (tuto rlgym.org, rlgym-ppo, GigaLearn) ajoutent touch/vitesse/boost [V].
- [A] Implication : sans shaping, l'exploration jusqu'au premier but repose sur le hasard + les resets (kickoff, états aléatoires proches du but) + un timeout no-touch court ; il faudra probablement > 1B steps pour un niveau Nexto, donc maximiser le SPS est prioritaire.

---

## 9. Faits d'ingénierie Rocket League (sourcés)

| Fait | Valeur | Source |
|---|---|---|
| Tick physique | **120 Hz** : `TICKS_PER_SECOND = 120` ; RocketSim v3 « Each `step_tick` advances … `1/120` second » ; RLBot « All bots are expected to run at 120hz » | [V] common_values.py · v3 README · wiki v5 |
| Tick skip standard | **8** (≈15 décisions/s) : tuto rlgym.org `action_repeat = 8`, `RepeatAction(repeats=8)` par défaut, exemple v2 rlgym-ppo, GigaLearn `tickSkip = 8`, Nexto `tick_skip = 8` | [V] |
| Délai d'action | 1 tick (RLBot) : `rlbot_delay=True` (RLGym 2.0.0+) / `actionDelay = tickSkip - 1` (GigaLearn) ; Nexto calcule l'action au tick `tick_skip - 1` | [V] |
| Durée de match | **5 min** : RLBC 2025 « Time limit: 5 min » ; exemples RLGym `TimeoutCondition(timeout_seconds=300.)` | [V] |
| Timeouts RLGym usuels | Tuto officiel : `NoTouchTimeoutCondition(30)` + `TimeoutCondition(300)` ; exemple v2 rlgym-ppo : no-touch 10 s ; GigaLearn : `NoTouchCondition(10)` (secondes, truncation) ; terminaison sur `GoalCondition` | [V] |
| Kickoff | `KickoffMutator` : 5 spawns par équipe (ex. bleu `[-2048,-2560,17]`, `[2048,-2560,17]`, `[-256,-3840,17]`, `[256,-3840,17]`, `[0,-4608,17]`), boost 33.3, balle au repos z = 93.15 (`BALL_RESTING_HEIGHT`) | [V] kickoff_mutator.py, common_values.py |
| Constantes | `SIDE_WALL_X=4096`, `BACK_WALL_Y=5120`, `CEILING_Z=2044`, `BALL_RADIUS=91.25`, `CAR_MAX_SPEED=2300`, `SUPERSONIC_THRESHOLD=2200`, `BALL_MAX_SPEED=6000` (uu) ; 50 uu = 1 m (v3 README) | [V] |

---

## Matrice de compatibilité avec ce hardware (Win11 / Py 3.12.6 / numpy 2.5.3 / pas de CUDA / Arc iGPU)

| Composant (version) | Win11 | Py 3.12 | numpy 2 | sans CUDA | Verdict |
|---|---|---|---|---|---|
| rlgym-api 2.0.0 | ✅ pur Py | ✅ | ✅ (0 dep) | ✅ | OK |
| rlgym / rlgym-rocket-league 2.0.1 (PyPI) | ✅ | ✅ | ❌ `numpy<2` [D] | ✅ | **Bloquant si on garde numpy 2** → build `main` ou venv numpy 1.26 [A] |
| rlgym `main` (2026-05-16) | ✅ | ✅ (>=3.9) | ✅ `<3.0.0` | ✅ | OK mais non publié (build manuel) |
| rlgym-sim 1.2.6 | ✅ | ✅ | ❌ `<2.0` | ✅ | Obsolète (API v1) |
| rlgym-ppo 1.3.13 (git) | ⚠️ spawn+UDP, non documenté | ✅ | ⚠️ req.txt `<2.0`, `gym` 0.26.2 non maintenu | ✅ device auto→cpu | Utilisable avec précaution ; gelé |
| rlgym-learn 2.0.0 | ✅ wheel cp312 win_amd64, « Multi-platform » | ✅ (3.10–3.14) | ⚠️ via `rlgym[rl]` 2.0.1 | ✅ (changer `cuda:0`) | **Meilleur candidat Python** |
| rlgym-learn-algos 0.4.2 | ✅ wheel | ✅ | ✅ `numpy>1.21` | ✅ | OK |
| rocketsim (pip) 2.2.1 | ✅ abi3 win_amd64 | ✅ | ✅ (build numpy≥2) | ✅ | OK |
| RocketSim v3 (crate 0.3.5) | ✅ (binaires viewer win msvc) | ❌ pas de binding | n/a | ✅ | Seulement via binding maison (Rust à installer) |
| rlviser-py 0.6.14 | ✅ abi3 | ✅ | ✅ (exige ≥2) | ✅ | OK (conflit avec rlgym 2.0.1) |
| RocketGPU (tiers, oct. 2026) | ? (CMake, clang/MSVC non précisé) | build local | ? | ✅ build CPU | Veille seulement |
| GigaLearnCPP (leak, archivé) | ⚠️ MSVC + CMake + libtorch CPU | n/a (embed Py) | n/a | ✅ `CPU` | ❌ viole « PPO+rewards Python » ; RLBot v4 |
| RLGymPPO_CPP | ⚠️ | n/a | n/a | ✅ libtorch CPU | Abandonné |
| pulsar2 | ? | n/a | n/a | ? | Non documenté |
| rlbot 2.0.0b56 + RLBotServer rc18 | ✅ (installé) | ✅ (>=3.11) | ✅ (0 dep numpy) | ✅ | OK (pré-release, Epic : settings joueur partiels) |
| rlgym-compat 2.3.6 | ✅ | ✅ (>=3.11) | ✅ `<3.0` | ✅ | OK |
| torch 2.15.0 CPU | ✅ | ✅ | ✅ | ✅ | OK (référence) |
| torch 2.15.0+xpu | ✅ Win11 listé | ✅ wheel cp312 | ✅ [A] | ✅ Arc MTL-H validé | Prototype — à benchmarker |
| onnxruntime 1.31.1 | ✅ | ✅ | ✅ [A] | ✅ | OK (inférence bot) |
| onnxruntime-directml 1.24.4 | ✅ | ✅ | ✅ [A] | ✅ DX12 | Sustained engineering |
| ORT OpenVINO (1.24.1 / ep 1.8.1) + openvino 2026.4.1 | ✅ | ✅ | ✅ [A] | ✅ GPU/NPU Intel | Possible, config plus lourde |
| Toolchain C++ locale | MinGW g++ 13.2 / VS BT 2019 (v142) / pas de cmake | — | — | — | ⚠️ libtorch/pybind11 : préférer MSVC récent ; linker v142 vs libs v143+ non supporté (MS) |

---

## Implications pour le choix d'infra (Phase 2)

### Chiffres RAPPORTÉS (sources) — ne pas confondre avec nos attentes
| Mesure | Valeur | Contexte | Source |
|---|---|---|---|
| RocketSim C++ v2.1.0 | 114 481 ticks/s/thread | 2v2, i5-11400, MSVC, Win10 | README RocketSim main [V] |
| RocketSim C++ (README actuel) | « ~20 min de jeu / s » ≈ 144k ticks/s/thread [D] | 2 voitures, « average PC » | README RocketSim main [V] |
| RocketSim v3 0.3.5 | 2,91–3,79 M arena-ticks/s (≈ 210–270k/thread [D]) | Ryzen 7 9800X3D, 14 threads, mixte 1v1–3v3 | README RocketGPU (tiers) [V] |
| RocketGPU CPU | 3,6–4,4 M arena-ticks/s | idem | README RocketGPU (tiers) [V] |
| RLGym-PPO (Python) | ~10k SPS | i5-11400 + RTX 3060 Ti | README RLGymPPO_CPP [V] |
| RLGymPPO_CPP | ~70k SPS | idem | README RLGymPPO_CPP [V] |
| GigaLearnCPP | collecte ~2× RLGymPPO_CPP, ~10× RLGym-PPO | machine de l'auteur | README GigaLearn [V] |
| Lucy-SKG | ~2000–4000 FPS | matériel non précisé | arXiv 2305.15801 [V] |

### Option A — RLGym v2 Python + RocketSim (pip) + PPO Python
- Variante A1 : **rlgym-learn 2.0.0 + rlgym-learn-algos 0.4.2** (Rust + shared memory, wheels Win cp312, maintenu 2026). Variante A2 : rlgym-ppo (gelé, numpy<2 côté requirements, UDP+RawArray, collecte/apprentissage alternés).
- Conforme à la contrainte (PPO + rewards Python). Mise en route la plus rapide ; la reward GCO (+1 but marqué / −1 encaissé) est triviale à écrire en `RewardFunction`.
- Bloqueur à traiter d'abord : numpy<2 de `rlgym-rocket-league` 2.0.1 → (a) venv uv dédié avec numpy 1.26.x (wheels cp312 dispo [A]) ou (b) build de `RLGym/rlgym@main` (numpy<3) — (b) préférable pour garder numpy 2 partout [A].
- Débit **[A]** sur le 125H (laptop, 28 W soutenu) : collecte ~3–8k SPS agent avec 10–12 process env (Python domine : construction `GameState`, obs, reward, dicts) ; update PPO CPU MLP 256×3 / batch 50k : quelques secondes par itération ⇒ ~3–6k SPS effectifs en A2 (A1 un peu mieux grâce au backend Rust). ⇒ ~0,3–0,5 Md steps/jour. RAM : ~0,15–0,3 Go/process Python [A] ⇒ ≤ 12–16 process avec 15,4 Go.

### Option B — Hybride : hot paths C++ (ou Rust) natifs + PPO/rewards Python
- Un module natif (pybind11 3.1.0 / nanobind 3.1.0 [V PyPI], ou PyO3 + crate `rocketsim` 0.3.5) qui possède N arènes RocketSim dans **un seul process par cœur**, applique tick-skip/action-delay, kickoff/state mutators, done conditions, obs en C++, et renvoie des tableaux numpy (N × obs) + un **état compact vectorisé** (positions/vitesses balle, buts, touches) ; la reward GCO reste en **Python numpy vectorisé** sur le batch (≈ quelques µs par batch, donc coût négligeable) ; PPO en torch Python (CPU, XPU à tester).
- Référence externe : C++ vs Python = ~7× en collecte (70k vs 10k, RLGymPPO_CPP [V]) — mais avec PPO aussi en C++.
- Débit **[A]** : collecte 20–50k SPS (sim ≈ 8 ticks × ~5–8 µs/tick/arène 1v1 sur cœur P, E-cores plus lents ; overhead Python amorti par batch) ; plafond pratique fixé par l'update PPO CPU ⇒ ~10–25k SPS effectifs ⇒ ~1–2 Md steps/jour. Gain ×3–5 vs A, au prix de ~1–2 semaines d'ingénierie et d'un risque de divergence sémantique avec RLGym (obs/GameState) qu'il faut couvrir par des tests d'équivalence contre RLGym + rlgym-compat.
- Prérequis machine : installer CMake (ou `scikit-build-core`/`cmake` via pip) et idéalement VS Build Tools 2022/2026 (le v142 actuel suffit pour pybind11 + RocketSim C++ compilés localement ; c'est le linkage avec des libs pré-compilées v143+ (libtorch) qui pose problème [V règle MS]) ; MinGW + CPython MSVC : non vérifié, à éviter.

### Option C — Autres
- **C1 GigaLearnCPP pur C++** : le plus rapide rapporté (≈10× RLGym-PPO [V]) mais **viole la contrainte Python** (PPO + rewards C++), repo archivé, RLBot v4, build Windows/libtorch/CMake à monter ⇒ écarté, sauf comme benchmark de plafond.
- **C2 pulsar2** : fork GigaLearn de recherche, sans doc ni chiffres ⇒ lecture d'idées seulement.
- **C3 RocketGPU (CPU)** : précision bit-exacte et débit CPU revendiqués > v3 [V-tiers], binding Python fourni, MIT ; mais repo de 2 semaines, 12★, meshes à dumper (dumper = patch mémoire du jeu, EAC) ⇒ veille ; à reconsidérer pour réduire le gap sim-to-real si le bot sous-performe dans le vrai jeu.
- **C4 RocketSim v3 Rust + PyO3 maison** : sous-cas de B ; avantage : pas de C++/MSVC ; inconvénient : toolchain Rust à installer, API 0.x instable (MIGRATING 0.2→0.3), meshes v3 non garantis identiques à ceux de RLGym.

### Recommandation provisoire [A]
1. Démarrer en **A1** (rlgym-learn + RLGym main buildé localement, numpy 2) pour valider de bout en bout : env 1v1, reward GCO, RLBot v5 via rlgym-compat 2.3.6 + `rlbot==2.0.0b56`, inférence torch CPU (< 8,33 ms/tick), `rlbot_delay=True`, tick skip 8, kickoff + timeout no-touch court.
2. Mesurer le SPS réel sur le 125H (P vs E cores, alimentation secteur, mode performance) avant d'engager **B** ; passer en B si SPS < ~5k ou si la reward sparse exige > 1 Md steps (probable [A]).
3. XPU : benchmark isolé de l'update PPO uniquement ; ne pas en dépendre.
4. Format compétition : RLBC 2025 = **2v2** ; aucune règle 2026 publique trouvée ⇒ le choix 1v1 exclut a priori ce format [D].
