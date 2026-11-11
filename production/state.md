# État du projet RL-GCO

Mis à jour : 2026-11-11 (G1 validé, Phase 2 présentée au G2).

## Où on en est
- Phase 0 terminée. **Gate G1 validé** le 2026-11-10 (décisions : `docs/GCO_SPEC.md` §10).
- **Compute** (G1) : Kaggle (2× T4 ou TPU v5e-8) ; laptop pour dev, smoke tests, orchestration, RLBot. **Proposé au G2**
  (`docs/adr/001-infra.md`) : laptop = plateforme principale mesurée, Kaggle T4 = appoint, rôle du TPU après sa sonde.
- **Phase 1 (studio) terminée** : `CLAUDE.md`, 17 agents, 10 skills, 7 rules, hooks testés (`tests/hooks`), ADR-000, dépôt git (remote privé `origin`).
- **Phase 2 (infra) en cours** (début 2026-11-10). Fait : rocketsim 2.2.1 + numpy 2.5.4 dans le venv, meshes soccar
  vérifiés (`scripts/fetch_meshes.py`), plafond sim laptop mesuré (21k env-steps/s à 1 proc, ≈ 72k à 14 procs, 20 s,
  commit 54a10a6, `docs/adr/001-infra.md` M1), kernel sonde prêt (`scripts/kaggle_probe.py`). RocketGPU @f0508a1
  compilé en local (gcc, PGO) : ≈ 130–215k env-steps/s en 1 processus selon l'état thermique, ≈ 2–2,5× RocketSim
  (M1c, M1d). Build scripté : `scripts/build_rocketgpu.py` ; la sonde Kaggle le construit (CPU, puis CUDA si GPU).
  Bout-en-bout prototype (M3, c110603) : ≈ 16,5k ts/s CPU, ≈ 52–58k ts/s avec learner sur GPU intégré Arc (torch XPU,
  `vendor/venv-xpu`). Sonde portable Colab prête (`--colab`).
- **Sondes Kaggle (2026-11-11, `docs/research/kaggle_probe.md`)** : T4 = 4 vCPU, RocketGPU CPU ≈ 55k env-steps/s.
  RocketGPU CUDA ne compile que pour sm_120 (Blackwell) : aucun GPU Kaggle/Colab utilisable pour la simulation.
  Sonde TPU en file. Gate G2 à la sortie (manque la sonde TPU).
- **Branches en attente du GO G2** : `claude/phase3-gco-reward-fd3080` (récompense GCO + tests §7 + ADR-002, Phase 3
  anticipée, revue APPROVE) ; `claude/phase2-kaggle-relay-954373` (lanceur de relais, aucun appel Kaggle, en revue).
- Livrables :
  - `docs/private/00_inventory.md` (local, non versionné : machine, contenu du dossier) ;
  - `docs/01_papers.md` (papiers et config PPO v0) ;
  - `docs/02_ecosystem.md` (versions et compatibilité) ;
  - `docs/03_reference_configs.md` (studios Claude Code et leçons communautaires) ;
  - `docs/GCO_SPEC.md` (définition, budget de steps) ;
  - `docs/PLAN.md` (plan d'ensemble) ;
  - `docs/research/` (notes sur sources publiques) ; preuves issues des discussions Discord dans `docs/private/` (local, non versionné).
- Venv `.venv` (uv, Python 3.12) : groupes dev, sim (rocketsim 2.2.1, numpy 2.5.4), native (cmake, ninja, pybind11), learn (torch 2.15.0 CPU) ; `vendor/venv-xpu` : torch 2.15.0+xpu.

## Prochaines actions
1. **Gate G2** : `gate_check G2` = PRÊT POUR REVUE HUMAINE (2026-11-11). Décision proposée : `docs/adr/001-infra.md`
   (A3 : RocketGPU CPU + env maison batché + learner torch XPU/CUDA). En attente du GO de l'utilisateur.
2. Après GO : merger les branches `claude/phase3-gco-reward-fd3080` et `claude/phase2-kaggle-relay-954373` (conflits
   attendus : `production/state.md` pour la branche relais seulement, d'après `git merge-tree`), puis Phase 3.
3. Sonde TPU (`rayhanda/rlgco-probe-tpu`) en file depuis le 2026-11-11 : récupérer `probe.json`, compléter M2c.
   Kaggle : usage autorisé par l'utilisateur le 2026-11-11 (son compte, ses quotas).

## Risques ouverts
- **R1** : amorçage de l'exploration en GCO à petit budget (aucune preuve publique sous 0,35B steps).
- **R2** : throttling **mesuré** : 75k ts/s les 6 premières minutes, plateau ≈ 45–50k ensuite, moyenne 50,3k sur
  57 min (M4, politique aléatoire). RAM (16 Go partagés) non mesurée.
- **R3** : RLGym 2.0.1 (PyPI) exige numpy < 2. Hors du chemin d'entraînement depuis A3 (env maison) ; reste à
  vérifier côté déploiement RLBot (parité obs/actions, Phase 3).
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
