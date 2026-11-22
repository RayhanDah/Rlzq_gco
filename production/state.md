# État du projet RL-GCO

Mis à jour : 2026-11-22 (Phase 4a : SIL, ADR-012 phase A et ADR-007 NEUTRES ; ADR-010 ou ADR-013 ensuite, décision utilisateur ; G3 non franchissable).

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
   2026-11-21) : audit d'un levier, LR ×3. **Phase A faite, verdict D3 NEUTRE** (2026-11-22, `docs/reports/diag_dense_lr3_s0.md`) :
   touche 29,1 % [28,5 ; 29,7] à 0,5B contre 23,0 %, gain ×1,26 < ×2 ; KL ∝ LR² réfutée (rapport 1,4 à 2,1). Audit clos, LR inchangée, pas de phase B.
3. **ADR-009 SIL accepté** (utilisateur, 2026-11-21, « SIL quoi qu'il arrive »). Préalables D7 faits : smoke PASS (aedf0ca),
   2 bancs, parité float16 PASS, c_sil 0,1 retenu, fenêtre 150 pas (règle RAM). Débit 32,6k/s (KS4 en alerte).
4. Ordre (validé le 2026-11-21) : ADR-012 phase A (faite, NEUTRE), ADR-007 (fait, NEUTRE), puis ADR-010 (proposé, code fusionné 70c64ed, bancs λ à faire). Ensuite seeds 1
   et 2, puis préparation de G3.
5. **Décision utilisateur attendue (avis research-director, 2026-11-22)** : l'exception du rapport SIL §9 est armée, mais faiblement.
   Le tête-à-tête montre que 0,8B bat 1,0B et la LR ×3 est NEUTRE. Recommandation : laisser finir ADR-007, puis, s'il sort NEUTRE,
   remonter une variante SIL avant ADR-010. La variante ne change que le fondu de c_sil, qui reste à 0,1 jusqu'à 0,8B puis tombe à 0
   à 0,9B (conforme C2, config seulement, ≈ 8,5 h estimé). **ADR-013 proposé** (`docs/adr/013-sil-fondu-tardif.md`), à valider ;
   risques C4 et KS2 (RAM non plus exposée qu'en v1).
6. Contraste P0 générique **mesuré** (ADR-005 : 0,15 but/min par camp, 7 à 8,5 × le kickoff). Hors chemin critique : sonde TPU (`rayhanda/rlgco-probe-tpu`) ;
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
- **R14** : pertes du GPU XPU (`UR_RESULT_ERROR_DEVICE_LOST`), 8 en ≈ 53 h de runs (dont 1 sur GCO-S s0, 1 sur le témoin dense, 0 sur la SIL, 1 sur la phase A d'ADR-012 à l'it. 1 148, 1 sur ADR-007 à l'it. 16 119), cause non établie (pilote Arc
  32.0.101.8860). Reprise automatique (`scripts/run_4a.py`) ; pas de garde contre un processus figé.

## Expériences en cours
- **ADR-007 `int_gamma998_s0` terminé** (2026-11-22 11 h 14, 86aa57a, 1 perte du GPU reprise, 0 restauration) : verdict D6 **NEUTRE**
  (1B : 3,24 % [3,01 ; 3,50] contre 3,14 % pour std s0 ; 1re paire H1 à 0,5B). γ 0,995 conservé. Rapport `docs/reports/int_gamma998_s0.md`.
- **Préalables GPU d'ADR-010** (7910742, après l'écriture des checkpoints en flux, revue APPROVE) : smokes PASS. Banc λ = 0,1 à H = 100 :
  λ = 0,1 retenu (grad_share 0,497), RAM min 2 261 Mo (D2 OK, contre 807 Mo avant), 44,0k/s. **Parité f16 FAIL**
  (0,0137 > 0,01) : D3 littérale ⇒ float32 et H = 50 ; option float32 à H = 100 mesurée (RAM min 1 884 Mo, D2 OK), à trancher
  par l'utilisateur.
- Phase A d'ADR-012 terminée (2026-11-22 04 h 46, it. 9 688, 3 segments dont 1 perte du GPU et 1 pause) : NEUTRE, voir ci-dessus.
- **Run SIL `int_sil_s0` terminé** (2026-11-21 18 h 26, 8,1 h, 1 segment, 0 restauration, aucun KS) : critère ADR-006 D6
  **NEUTRE** (1B : décisives 4,33 % [4,06 ; 4,63] < 6,28 %, SPRT H1 ; 1re paire H1 à 0,3B). Descriptif : pic à 0,7B
  (8,8 % de décisives, touche 41,7 %) puis retour à 3,7 % / 21,7 % à 1B, pendant l'annealing de c_sil ; persistance D4
  déclenchée (« gain dépendant de la SIL »). Rapport `docs/reports/int_sil_s0.md` (§7 bis, post hoc 2026-11-22 : 0,8B bat 1,0B en tête-à-tête, Élo +135, lecture « oubli » ;
  bosse confirmée avec `--seed 1`) ; avis research-director : SIL v1 non promue,
  ordre **validé par l'utilisateur (2026-11-21)** : analyse post hoc, ADR-012, ADR-007, ADR-010, variante SIL éventuelle
  (nouvel ADR, C2). Mandat du même jour : « je te laisse gérer le projet et les runs » ⇒ ADR-007 lancé après la phase A si
  celle-ci n'est pas SUCCÈS (sinon phase B d'abord, sur GO, ADR-012 D8).
- Témoin dense terminé : `runs/diag_dense_s0/`, éval hors ligne `runs/diag_dense_offline_eval/` (1 perte du GPU reprise).
- Résultats 4a : `runs/run_4a_std_s{0,1,2}/`, `runs/run_4a_s_s0/`, évaluation hors ligne `runs/run_4a_offline_eval/`.
