# État du projet RL-GCO

Mis à jour : 2026-11-21 (Phase 4a : témoin dense fini, AMBIGU ; run SIL `int_sil_s0` lancé ; G3 non franchissable).

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
2. **Témoin dense fini (ADR-011)** : verdict D3 **AMBIGU** à 1B (A touche 50,4 % [49,75 ; 51,13], 20 000 épisodes, SPRT H1 ;
   0,5B : 23,0 %). Le pipeline apprend le contact quand le signal existe, lentement. Suite D3 : **ADR-012 accepté** (utilisateur,
   2026-11-21) : audit d'un levier, LR ×3, phase A `diag_dense_lr3_s0` à 0,5B **après** la SIL ; P1 fait (×3 confirmé), P2 fait et revu (APPROVE, 7722c59) ;
   reste le smoke sur le commit du run, puis `run_4a.py --variant diag_dense_lr3 --seed 0 --max-hours 0 --max-iterations 9688`.
3. **ADR-009 SIL accepté** (utilisateur, 2026-11-21, « SIL quoi qu'il arrive »). Préalables D7 faits : smoke PASS (aedf0ca),
   2 bancs, parité float16 PASS, c_sil 0,1 retenu, fenêtre 150 pas (règle RAM). Débit 32,6k/s (KS4 en alerte).
4. Ordre : SIL (en cours), puis ADR-012 phase A (≈ 3,3 h, estimé) et phase B si SUCCÈS et GO, puis ADR-007 (à valider), puis tâche auxiliaire (ADR-010, esquisse). Ensuite seeds 1
   et 2, puis préparation de G3.
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
- **R14** : pertes du GPU XPU (`UR_RESULT_ERROR_DEVICE_LOST`), 7 en ≈ 44 h de runs (dont 1 sur GCO-S s0, 1 sur le témoin dense, 0 sur la SIL, 1 sur la phase A d'ADR-012 à l'it. 1 148), cause non établie (pilote Arc
  32.0.101.8860). Reprise automatique (`scripts/run_4a.py`) ; pas de garde contre un processus figé.

## Expériences en cours
- **Phase A d'ADR-012 `diag_dense_lr3_s0`** (HORS SPEC, graine 0, LR ×3, arrêt à l'itération 9 688 = 0,5B) : lancée le
  2026-11-21 18 h 34 sur 042b335 (smoke `runs/smoke_dense_lr3_042b335` PASS), `run_4a.py --variant diag_dense_lr3 --seed 0
  --max-hours 0 --max-iterations 9688` ; ≈ 3,1 h (estimé). θ₀ identique au témoin (SHA-256 7fab4c74…cb8a, mesuré).
  1 perte du GPU (it. 1 148) reprise. **En pause** : arrêté le 2026-11-21 ≈ 21 h 15 à la demande de l'utilisateur (machine
  requise ≈ 1 h 30), processus tués à l'it. ≈ 5 935 ; reprise depuis `ckpt_000023_it00005750.pt` par la même commande. L'instantané
  d'éval de l'it. 5 813 sera réécrit à la reprise (le plus récent fait foi).
- **Run SIL `int_sil_s0` terminé** (2026-11-21 18 h 26, 8,1 h, 1 segment, 0 restauration, aucun KS) : critère ADR-006 D6
  **NEUTRE** (1B : décisives 4,33 % [4,06 ; 4,63] < 6,28 %, SPRT H1 ; 1re paire H1 à 0,3B). Descriptif : pic à 0,7B
  (8,8 % de décisives, touche 41,7 %) puis retour à 3,7 % / 21,7 % à 1B, pendant l'annealing de c_sil ; persistance D4
  déclenchée (« gain dépendant de la SIL »). Rapport `docs/reports/int_sil_s0.md` ; avis research-director : SIL v1 non promue,
  ordre **validé par l'utilisateur (2026-11-21)** : analyse post hoc, ADR-012, ADR-007, ADR-010, variante SIL éventuelle
  (nouvel ADR, C2). Mandat du même jour : « je te laisse gérer le projet et les runs » ⇒ ADR-007 lancé après la phase A si
  celle-ci n'est pas SUCCÈS (sinon phase B d'abord, sur GO, ADR-012 D8).
- Témoin dense terminé : `runs/diag_dense_s0/`, éval hors ligne `runs/diag_dense_offline_eval/` (1 perte du GPU reprise).
- Résultats 4a : `runs/run_4a_std_s{0,1,2}/`, `runs/run_4a_s_s0/`, évaluation hors ligne `runs/run_4a_offline_eval/`.
