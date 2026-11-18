# État du projet RL-GCO

Mis à jour : 2026-11-18 (Phase 4a : runs GCO-std faits, protocole v2 en préparation, G3 non franchissable).

## Où on en est
- **Phases 0 à 3 terminées.** G1 validé le 2026-11-10 (`docs/GCO_SPEC.md` §10), G2 le 2026-11-11 (§12,
  `docs/adr/001-infra.md` : laptop = plateforme principale, Kaggle T4 en appoint). Studio, hooks, ADR-000 à 005.
  Smoke test 1M : PASS (`docs/reports/smoke_test.md`, ea0e7de).
- Venvs : `.venv` (uv, Python 3.12, torch 2.15.0 CPU, rocketsim 2.2.1) ; `vendor/venv-xpu` (torch 2.15.0+xpu).
  Simulateur d'entraînement : RocketGPU @f0508a1 en CPU (`scripts/build_rocketgpu.py`).
- **Phase 4a** (sprint `production/sprints/2026-11-13.md`, ADR-005) : A0 à A7 faits.
  - 3 runs GCO-std de 1B (seeds 0, 1, 2), terminés le 2026-11-15, 16 et 18 ; ≈ 5,5 h chacun ; 1 perte du GPU reprise
    automatiquement par seed, 1 restauration K4 (seed 0), aucun STOP.
  - **Rapport A7** : `docs/reports/run_4a.md`. S1-léger jamais atteint ((a) bloquée par le plancher de 20 décisives
    sur 512). Mesure hors ligne (§4 bis, f324b21) : SPRT +100 Elo contre l'aléatoire accepté sur l'instantané final
    des 3 seeds (Elo décisif +159 à +203, 20 000 épisodes), (b) vraie sur s0 et s1. **Mais l'agent ne touche la balle
    que dans 9 à 14 % des épisodes kickoff** (81–86 % sans aucune touche).
  - Avis `research-director` (2026-11-18), suivi par l'utilisateur : résultat « S1 hors ligne, post hoc,
    exploratoire » ; K1 non déclenché ; H1 non concluante ; ni prolongation ni G3.

## Prochaines actions
1. **ADR-006** (protocole S1 v2, levée de D3 pour GCO-S seed 0, règle de décision, critère de succès,
   instrumentation) : en rédaction par `research-director`, **à valider par l'utilisateur**.
2. Instrumentation exigée par ADR-006, revue, smoke test sur le même commit.
3. **GCO-S seed 0** (1B, ≈ 5,5 h) : lancé par l'utilisateur après validation d'ADR-006.
4. Puis une seule intervention sur la seed 0 (γ 0,998 si S ≥ std, annealing plus lent sinon), ≈ 5,5 h ; si succès,
   seeds 1 et 2 ; puis préparation de G3 (`configs/long_run.yaml`, kill criteria).
5. Hors chemin critique : contraste P0 depuis les états génériques ; sonde TPU (`rayhanda/rlgco-probe-tpu`) ;
   relais Kaggle (`scripts/kaggle_run.py`) jamais exécuté, à valider sur un dataset jetable.

## Risques ouverts
- **R1** : amorçage de l'exploration en GCO à petit budget. **Mesuré** : à 1B, premier contact rare (§4 bis).
- **R2** : débit **mesuré** soutenu ≈ 51k appris/s en vrai run (A7), 55,1k au banc M8 ; 1B ≈ 5,5 h.
- **R3** : parité RLBot : conventions réelles du jeu non vérifiées avant la Phase 5 ; `rlbot` 2.0.0b56 pré-version.
- **R4** : code pulsar2 / GigaLearn sans licence → référence uniquement.
- **R5** : RLBC 2025 en 2v2 uniquement ; aucune règle 2026 publiée.
- **R6** : Rocket League sous EAC → passer exclusivement par RLBot.
- **R7, R8, R9, R12** : Kaggle : quotas rapportés par des tiers ; T4 = 4 vCPU (≈ 55k env-steps/s, mesuré) ; RocketGPU
  CUDA sm_120 seulement ; TPU en file, débit non mesuré.
- **R10** : jeton Kaggle jamais dans le dépôt (hooks de refus).
- **R11** : 3 comptes Kaggle, chacun opéré par son titulaire (décision du 2026-11-10).
- **R13** : écarts RocketGPU / RocketSim mesurés (T3), sans effet avec la table de 90 actions ; écart sim/jeu non
  mesuré avant la Phase 5.
- **R14** : pertes du GPU XPU (`UR_RESULT_ERROR_DEVICE_LOST`), 4 en ≈ 22 h de runs, cause non établie (pilote Arc
  32.0.101.8860). Reprise automatique (`scripts/run_4a.py`) ; pas de garde contre un processus figé.

## Expériences en cours
- Aucune. Résultats 4a : `runs/run_4a_std_s{0,1,2}/`, évaluation hors ligne `runs/run_4a_offline_eval/`.
