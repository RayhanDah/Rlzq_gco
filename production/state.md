# État du projet RL-GCO

Mis à jour : 2026-11-19 (Phase 4a : GCO-std ×3 et GCO-S s0 faits, témoin dense en préparation, G3 non franchissable).

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
1. **ADR-006 accepté** (2026-11-18). **GCO-S s0 fait** (2026-11-19, `docs/reports/gcos_s0.md`) : S1-seed à **0,2B**,
   confirmé hors ligne ; à 1B 4,31 % de victoires par épisode contre 2,40 % pour std s0 (une graine) ; touche 14,6 %.
   Règle D5 : S ≥ std, intervention désignée (2) γ 0,998 (ADR-007, proposé).
2. **ADR-011 accepté** (GO utilisateur 2026-11-19) : témoin dense HORS SPEC, diagnostic du pipeline. **Prêt** : revue
   APPROVE, fusionné (015c65d), smoke PASS standard et dense, essai du lanceur (ADR-011 « Implémentation »).
   **Lancement seulement sur GO de l'utilisateur** : `scripts/run_4a.py --variant diag_dense --seed 0 --max-hours 3`.
3. **SIL dans main** (ADR-009 proposé, amendé après revue : mode joint, `disp_share` ; revues APPROVE), désactivée par
   défaut ; banc D7 ≤ 30 min (calibrage de c_sil, RAM, float16) avant tout run, sur demande.
4. Ordre recommandé par `research-director` : témoin dense, puis SIL, puis ADR-007, puis tâche auxiliaire (ADR-010,
   esquisse). À valider par l'utilisateur (ADR-009, ADR-007). Ensuite seeds 1 et 2, puis préparation de G3.
5. Contraste P0 générique **mesuré** (ADR-005 : 0,15 but/min par camp, 7 à 8,5 × le kickoff). Hors chemin critique : sonde TPU (`rayhanda/rlgco-probe-tpu`) ;
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
- **R14** : pertes du GPU XPU (`UR_RESULT_ERROR_DEVICE_LOST`), 5 en ≈ 27,5 h de runs (dont 1 sur GCO-S s0), cause non établie (pilote Arc
  32.0.101.8860). Reprise automatique (`scripts/run_4a.py`) ; pas de garde contre un processus figé.

## Expériences en cours
- **Témoin dense `diag_dense_s0`** (HORS SPEC, ADR-011), commit e04871e : session 1 lancée par l'utilisateur le 2026-11-20
  à 21:19, `--max-hours 3` ; arrêtée à l'itération 9 264 (0,478B, mesuré), donc **avant 0,5B** (estimation ADR fausse).
  Éval en cours de run à 0,4B (it. 7 751, 3 072 épisodes) : A touche 17,7 %, aucune touche 77,8 %, SPRT H1 ; entropie 4,29,
  KL 4e-4 (std s0 : 4,45 et 1,4e-4). Verdict D3 pas encore mesurable. Session 2 décidée par l'utilisateur (2026-11-21) : même commande avec
  `--max-hours 6` ; arrêt attendu au budget de 1B après ≈ 3,3 h (estimé, non mesuré).
- Résultats 4a : `runs/run_4a_std_s{0,1,2}/`, `runs/run_4a_s_s0/`, évaluation hors ligne `runs/run_4a_offline_eval/`.
