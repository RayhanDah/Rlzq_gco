# État du projet RL-GCO

Mis à jour : 2026-11-10 (G1 validé, Phase 1 terminée).

## Où on en est
- Phase 0 terminée. **Gate G1 validé** le 2026-11-10 (décisions : `docs/GCO_SPEC.md` §10).
- **Compute** : entraînement sur **Kaggle** (2× T4 ou TPU v5e-8, via clé API installée par l'utilisateur). Le laptop sert au dev, aux smoke tests, à l'orchestration et à RLBot. Voir `docs/research/kaggle.md`.
- **Phase 1 (studio) terminée** : `CLAUDE.md`, 17 agents, 10 skills, 7 rules, hooks testés (`tests/hooks`), ADR-000, dépôt git (remote privé `origin`).
- **Phase 2 (infra) en cours** (début 2026-11-10). Fait : rocketsim 2.2.1 + numpy 2.5.4 dans le venv, meshes soccar
  vérifiés (`scripts/fetch_meshes.py`), plafond sim laptop mesuré (21k env-steps/s à 1 proc, ≈ 72k à 14 procs, 20 s,
  commit 54a10a6, `docs/adr/ADR-infra.md` M1), kernel sonde prêt (`scripts/kaggle_probe.py`). Gate G2 à la sortie.
- Livrables :
  - `docs/private/00_inventory.md` (local, non versionné : machine, contenu du dossier) ;
  - `docs/01_papers.md` (papiers et config PPO v0) ;
  - `docs/02_ecosystem.md` (versions et compatibilité) ;
  - `docs/03_reference_configs.md` (studios Claude Code et leçons communautaires) ;
  - `docs/GCO_SPEC.md` (définition, budget de steps) ;
  - `docs/PLAN.md` (plan d'ensemble) ;
  - `docs/research/` (notes sur sources publiques) ; preuves issues des discussions Discord dans `docs/private/` (local, non versionné).
- Venv `.venv` (uv, Python 3.12) avec outils de dev seulement : ruff 0.17.4, mypy 2.4.1, pytest 9.2.0. Aucune dépendance RL installée.

## Prochaines actions
1. Sonde Kaggle T4 + TPU (`python scripts/kaggle_probe.py --user <pseudo> --acc t4,tpu --push`).
   Prérequis utilisateur : jeton installé (`.venv/Scripts/kaggle auth login`), téléphone vérifié (internet/GPU).
2. Décision A1 (rlgym-learn + RLGym `main`, build à autoriser) vs A2 (env maison sur rocketsim PyPI), puis bench
   bout-en-bout (M3) et bench laptop ≥ 30 min (throttling) → ADR-infra → G2.

## Risques ouverts
- **R1** : amorçage de l'exploration en GCO à petit budget (aucune preuve publique sous 0,35B steps).
- **R2** : débit du laptop (3k à 25k SPS estimés, non mesuré) ; throttling thermique ; 16 Go de RAM partagés.
- **R3** : `rlgym-rocket-league` 2.0.1 sur PyPI exige numpy < 2 → build de `main` ou venv numpy 1.26.
- **R4** : code pulsar2 / GigaLearn sans licence → référence uniquement.
- **R5** : RLBC 2025 était en 2v2 uniquement ; aucune règle 2026 publiée.
- **R6** : Rocket League sous EAC depuis avril 2026 → passer exclusivement par RLBot (lancé sans EAC), pas de dumper mémoire.
- **R7** : quotas Kaggle (≈ 20 h TPU / 30 h GPU par semaine, sessions ≤ 9–12 h) rapportés par des tiers et modifiables.
- **R8** : interruption possible d'une session à accélérateur sous-utilisé (CGU non lues).
- **R9** : 4 cœurs CPU seulement en session T4 → collecte plafonnée.
- **R10** : fuite du jeton Kaggle → jamais dans le dépôt, hooks de refus.
- **R11** : groupe de 3 → 3 comptes Kaggle. **Décision utilisateur (2026-11-10)** : chaque membre opère son propre compte (`git pull` + une commande depuis son PC, son propre jeton) ; notebook commun et datasets partagés si possible. Le studio n'opère que le compte de l'utilisateur et ne manipule aucune clé tierce. Zone grise résiduelle : pooling de quotas (AUP « circumvent »).
- **R12** : TPU avec file d'attente ≥ 35 h. L'utilisateur rapporte ≈ 2,5B steps par session de 9 h (≈ 77k SPS, **non mesuré**) ; si confirmé par la sonde, le TPU (≈ 10× une session T4 estimée) devient la ligne principale malgré la file. GPU + TPU en parallèle : limite de concurrence non documentée → à tester par la sonde.

## Expériences en cours
Aucune.
