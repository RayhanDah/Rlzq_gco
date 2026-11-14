# État du projet RL-GCO

Mis à jour : 2026-11-14 (**G2 validé**, Phase 3 terminée, Phase 4a : A0 à A5 faits, A6 à faire).

## Où on en est
- Phase 0 terminée. **Gate G1 validé** le 2026-11-10 (décisions : `docs/GCO_SPEC.md` §10).
- **Gate G2 validé** le 2026-11-11 (GO utilisateur ; décisions : `docs/GCO_SPEC.md` §12, `docs/adr/001-infra.md`) :
  option A3, laptop = plateforme principale (≈ 50k ts/s soutenus mesurés), Kaggle T4 = appoint, rôle du TPU après
  sa sonde.
- **Phase 1 (studio) terminée** : `CLAUDE.md`, 17 agents, 10 skills, 7 rules, hooks testés (`tests/hooks`), ADR-000, dépôt git (remote privé `origin`).
- **Phase 2 (infra) terminée** (2026-11-10 → 11). Fait : rocketsim 2.2.1 + numpy 2.5.4 dans le venv, meshes soccar
  vérifiés (`scripts/fetch_meshes.py`), plafond sim laptop mesuré (21k env-steps/s à 1 proc, ≈ 72k à 14 procs, 20 s,
  commit 54a10a6, `docs/adr/001-infra.md` M1), kernel sonde prêt (`scripts/kaggle_probe.py`). RocketGPU @f0508a1
  compilé en local (gcc, PGO) : ≈ 130–215k env-steps/s en 1 processus selon l'état thermique, ≈ 2–2,5× RocketSim
  (M1c, M1d). Build scripté : `scripts/build_rocketgpu.py` ; la sonde Kaggle le construit (CPU, puis CUDA si GPU).
  Bout-en-bout prototype (M3, c110603) : ≈ 16,5k ts/s CPU, ≈ 52–58k ts/s avec learner sur GPU intégré Arc (torch XPU,
  `vendor/venv-xpu`). Sonde portable Colab prête (`--colab`).
- **Sondes Kaggle (2026-11-11, `docs/research/kaggle_probe.md`)** : T4 = 4 vCPU, RocketGPU CPU ≈ 55k env-steps/s.
  RocketGPU CUDA ne compile que pour sm_120 (Blackwell) : aucun GPU Kaggle/Colab utilisable pour la simulation.
  Sonde TPU en file.
- **Phase 3 (implémentation) ouverte** : récompense GCO mergée (`src/rewards`, tests §7, ADR-002, 1bcd5ae).
  Lanceur de relais Kaggle mergé (`scripts/kaggle_run.py`, `docs/research/kaggle_relay.md`, revue faite) : jamais
  exécuté contre Kaggle, à valider sur un dataset jetable avant tout vrai run.
- Livrables :
  - `docs/private/00_inventory.md` (local, non versionné : machine, contenu du dossier) ;
  - `docs/01_papers.md` (papiers et config PPO v0) ;
  - `docs/02_ecosystem.md` (versions et compatibilité) ;
  - `docs/03_reference_configs.md` (studios Claude Code et leçons communautaires) ;
  - `docs/GCO_SPEC.md` (définition, budget de steps) ;
  - `docs/PLAN.md` (plan d'ensemble) ;
  - `docs/research/` (notes sur sources publiques) ; preuves issues des discussions Discord dans `docs/private/` (local, non versionné).
- Venv `.venv` (uv, Python 3.12) : groupes dev, sim (rocketsim 2.2.1, numpy 2.5.4), native (cmake, ninja, pybind11), learn (torch 2.15.0 CPU) ; `vendor/venv-xpu` : torch 2.15.0+xpu.

- **Sprint Phase 3** (`production/sprints/2026-11-11.md`) : T0 à T7 mergés (`src/{obs,actions,env,models,training,
  rlbot_bot}`, `scripts/train.py`, `configs/smoke.yaml` ; ADR-003 et ADR-004 acceptés). Reprise exacte mesurée,
  simulateur bit à bit (CPU). **Smoke test 1M steps : PASS** le 2026-11-13 sur XPU (`docs/reports/smoke_test.md`,
  commit ea0e7de : 0 NaN, reprise à l'itération 9, ≈ 82k ts/s sur 13 s, non soutenu). **Phase 3 terminée.**

## Prochaines actions
1. **Phase 4a (run court)** : sprint `production/sprints/2026-11-13.md` : A0 (ADR-005, **accepté** le 2026-11-13 : 4 runs de 1B,
   un par session de 8 h), A1 à A4 mergés (A3 pool : 445385a ; A4 évaluation en cours de run : dbd504e,
   revues faites) ; A5 fait (M5 : 46,0k appris/s sur 45 min, ≈ 40k en fin de banc) ; restent A6, A7. Choix d'A4 validés par l'utilisateur
   le 2026-11-14 (ADR-005, « Précisions validées »). Prévu au départ :
   protocole (ADR-005), durcissement de la boucle, évaluation (`src/eval`), pool d'adversaires, banc soutenu ≥ 30 min,
   puis 6 runs (2 variantes × 3 seeds). Bancs et runs à la demande de l'utilisateur.
2. Avant le run court 4a : points de la revue T6 listés dans le sprint (détection de S1, checkpoint à l'itération 0,
   journal après restauration), délai d'action d'un tick de RLBot en ablation.
3. Sonde TPU (`rayhanda/rlgco-probe-tpu`) toujours en file le 2026-11-13 : récupérer `probe.json`, compléter M2c.
4. Prochain gate : **G3**, avant le premier run long (smoke test, run court 4a, config et kill criteria).

## Risques ouverts
- **R13** : écarts RocketGPU / RocketSim **mesurés** (T3) : boost au sol sans gaz (RocketGPU freine), ordre des
  34 pads différent, boost initial 33,3 contre 33,3333. Sans effet avec la table de 90 actions (boost ⇒ plein
  gaz) ; à tester dans T1 et T7. Aucune comparaison au jeu avant la Phase 5.
- **R1** : amorçage de l'exploration en GCO à petit budget (aucune preuve publique sous 0,35B steps).
- **R2** : throttling **mesuré** : 75k ts/s les 6 premières minutes, plateau ≈ 45–50k ensuite, moyenne 50,3k sur
  57 min (M4, politique aléatoire). Vraie boucle avec pool et évaluation (M5, 2026-11-14) : 46,0k appris/s sur
  45 min, pool de 7 à 49. **Pool plein (M6, 100 copies de θ₀) : ≈ 34,5k pendant 25 min puis ≈ 26,6k, 32,2k en
  moyenne sur 35 min : 1B ne tient pas en 8 h** (8,6 à 10,4 h extrapolé). RAM disponible min 588 à 715 Mo sur 16 Go.
  À trancher avant A6 : réduire le coût du pool à l'inférence, ou réviser le budget de l'ADR-005.
- **R3** : parité RLBot : cohérence adaptateur ↔ obs **testée** (écart ≤ 7,2e-7), mais conventions réelles du jeu
  (ordre d'Euler, signe du roll, ordre des pads, délai d'action) non vérifiées avant la Phase 5. `rlbot` 2.0.0b56
  est une pré-version.
- **R4** : code pulsar2 / GigaLearn sans licence → référence uniquement.
- **R5** : RLBC 2025 était en 2v2 uniquement ; aucune règle 2026 publiée.
- **R6** : Rocket League sous EAC depuis avril 2026 → passer exclusivement par RLBot (lancé sans EAC), pas de dumper mémoire.
- **R7** : quotas Kaggle (≈ 20 h TPU / 30 h GPU par semaine, sessions ≤ 9–12 h) rapportés par des tiers et modifiables.
- **R8** : interruption possible d'une session à accélérateur sous-utilisé (CGU non lues).
- **R9** : session T4 = 4 vCPU (**mesuré**, sonde du 2026-11-11) → collecte ≈ 55k env-steps/s ; RocketGPU CUDA ne
  compile que pour sm_120.
- **R10** : fuite du jeton Kaggle → jamais dans le dépôt, hooks de refus.
- **R11** : groupe de 3 → 3 comptes Kaggle. **Décision utilisateur (2026-11-10)** : chaque membre opère son propre compte (`git pull` + une commande depuis son PC, son propre jeton) ; notebook commun et datasets partagés si possible. Le studio n'opère que le compte de l'utilisateur et ne manipule aucune clé tierce. Zone grise résiduelle : pooling de quotas (AUP « circumvent »).
- **R12** : TPU avec file d'attente ≥ 35 h ; ≈ 2,5B steps par session de 9 h rapportés par l'utilisateur (**non mesuré**,
  pipeline inconnu). À comparer au laptop seulement après la sonde TPU (en file). GPU + TPU : un kernel TPU a été accepté
  en file pendant qu'un T4 tournait ; exécution simultanée non encore observée.

## Expériences en cours
Aucune.
