# ADR-013 — Variante SIL à fondu tardif (GCO-std, graine 0, 1B) : c_sil plein jusqu'à 0,8B

- Statut : **proposé** (validation utilisateur requise ; amende l'ordre validé le 2026-11-21, voir D6)
- Date : 2026-11-22
- Décideurs : research-director (proposition) ; rl-algo-lead (config, variante du lanceur, tests) ; eval-lead (évals hors ligne) ;
  code-reviewer ; utilisateur (validation, lancement)

## Contexte
Abréviations : R = `docs/reports/int_sil_s0.md` ; L = `docs/reports/diag_dense_lr3_s0.md` ; A9 = `docs/adr/009-intervention-sil.md` ;
A10 = `docs/adr/010-intervention-aux-prochain-but.md`.
- SIL v1 (A9) : verdict ADR-006 D6 **NEUTRE**, 4,33 % de décisives [4,06 ; 4,63] à 1B, sous 6,28 % (R:13-14). Trajectoire en cloche :
  8,79 % à 0,7B, 8,25 % à 0,8B, 3,71 % à 1B (R:70-73) ; std s0 monte sur la même fenêtre (2,34 % à 0,7B → 3,14 % à 1B, R:70, R:74).
- Diagnostic de persistance A9 D4 déclenché (R:97) ; marge resserrée de 0,16 à 1,53 point par 20 000 épisodes `--seed 1` à 0,9B
  (R:204-205, post hoc). Tête-à-tête 0,8B contre 1,0B : +135 Élo [119 ; 149], 1 551 V / 715 D (R:196) ; d'après la grille écrite
  avant la mesure (R:236), c'est la lecture « oubli » qui est retenue (R:200-201), sur une graine.
- Buts d'entraînement : en hausse tant que c_sil passe de 0,069 à 0,019 (10,98 → 11,95 / it.), puis −27 % quand c_sil tombe de
  0,019 à 0 (R:153-155). La baisse touche le kickoff, pas les états génériques. Le gain se construit donc à plein coefficient (R:80-81)
  et se perd en fin de fondu. Mécanisme (oubli, ou recalage de V après le terme β, A9:140) non établi.
- Lecture concurrente « pas effectif plus grand » affaiblie : LR ×3 donne ×1,26 sur la touche à 0,5B (29,10 % contre 23,04 %, L:46-48),
  verdict ADR-012 NEUTRE ; la SIL donne ×3,1 contre std (25,4 % contre 8,1 %, R:68). Limites : récompense dense HORS SPEC, arrêt à 0,5B,
  ×1,26 sous la dispersion de 1,60 entre graines (L:53, L:73). Affaiblie, pas réfutée en GCO.
- Exception du rapport (R:228-230) : armée, faiblement (avis research-director du 2026-11-22). Question ouverte : un gain construit à
  c_sil plein **persiste-t-il** après un retrait conforme à C2 ? v1 ne le dit pas, son fondu commençant à 0,5B, avant le pic.

## Options
| Option | Pour | Contre |
|---|---|---|
| **Fondu tardif : 0,1 jusqu'à 0,8B, linéaire à 0 à 0,9B, 0 jusqu'à 1B** | un seul levier (date de début du fondu) ; même phase finale que v1 ; config seule | une graine ; fondu 4× plus raide que v1 ; choix fait après avoir vu v1 |
| Fondu plus tôt et plus lent (0,3 → 0,9B, R:234) | conforme C2 | retire la SIL quand le gain se construit (R:80-81) : écartée |
| SIL jusqu'à 1,8B sur un run de 2B (R:234) | conforme C2 | coût ≈ 2× (estimé, non mesuré), deux leviers (budget et fondu) |
| Sans libération (c_sil > 0 jusqu'à 1B) | mesure le plafond | viole C2 (`GCO_SPEC.md:84`) : ablation GCO-ext seulement, jamais ligne principale |
| Ne rien faire (ADR-010 directement) | aucun coût | laisse sans réponse la seule piste in-spec dont la borne basse a dépassé 6,28 % (0,7B et 0,8B, R:70-71 ; 8,25 % [7,88 ; 8,64], R:197) |

## Mesures
Aucune mesure nouvelle : décision réversible, fondée sur R (run a1a57c5, évals 042b335 et 86aa57a, CPU), L (86aa57a) et les bancs A9.
- v1 : débit appris 31,4 à 34,4k/s jusqu'à 0,9B, 51,0k/s ensuite (R:128) ; boucle 8,14 h (R:23-24). RAM : 1 épisode sous 1 Go pendant
  la SIL (23 s, 521 Mo), 24 après la libération (min 143 Mo, 108 s au plus), KS1 jamais déclenché (R:124-127) ; buffer libéré à l'it.
  17 440 (R:123). C4 : disp_share médiane 0,136, p95 0,264, 0 itération > 0,8 (R:121).
- Bancs A9 (aedf0ca, 2026-11-21) : disp_share médiane 0,103 à c_sil 0,1, buffer plein ; RAM minimum 2 354 Mo (A9:249-258).

## Décision (proposée)

### D1. Levier unique : date de début du fondu
- `sil.anneal_start_timesteps` 500 000 000 → **800 000 000** ; `anneal_end_timesteps` reste 900 000 000 (`configs/int_sil.yaml:77-78`).
  c_sil = 0,1 de 0 à 0,8B, linéaire de 0,1 à 0 entre 0,8B et 0,9B, 0 de 0,9B à 1B (calcul SIL sauté, buffer libéré, `src/training/sil.py:147-155`,
  `:526`). Aucun changement dans `src/` : le calendrier est déjà paramétré (`sil.py:91-92`).
- Tout le reste identique à v1 : β, buffer 250 000, fenêtre 150, lot 4 096, mode joint, priorités, alertes (`int_sil.yaml:74-94`).
- Effet sur l'intégrale de c_sil de 0 à 0,9B : 0,085 contre 0,070 en coefficient × B (+21 %, calcul exact sur le calendrier).

### D2. Conformité C1–C4 (`GCO_SPEC.md:80-88`)
- **C2** (`GCO_SPEC.md:84`, « coefficient annealé à 0 avant la phase finale ») : la phase finale est lue comme le dernier 0,1B de
  PPO pur (≈ 1 938 itérations), **même lecture que v1** (A9:104-108), validée par l'utilisateur le 2026-11-21 (A9:220-222). La spec ne fixe
  ni la durée de cette phase ni la pente du fondu : un fondu sur 0,1B au lieu de 0,4B ne change pas la lecture. Les deux termes
  (politique et valeur) sont annealés (A9:140). Le critère (ii) de D6 (≤ 0,2B) reste à plein coefficient, et le (i) mesure à 1B une
  politique qui a fini en GCO pur.
- **C1, C3, C4** : inchangés (A9 D6) ; `sil.enabled` reste l'interrupteur ; journal `sil/*` et alerte disp_share > 0,5 sur 50 itérations.

### D3. Prédiction pré-enregistrée (consignée avant le run)
- Critère **ADR-006 D6 inchangé** (A9:170-172) : succès si (i) à 1B, 20 000 épisodes `--seed 0`, borne basse Wilson ≥ 6,28 % et
  SPRT +100 H1, ou (ii) 1re paire H1 au plus tard à 0,2B. Référence std s0 : 3,14 % (R:74). Comparaison à std s0 seulement.
- Diagnostic de persistance **A9 D4 inchangé** : « gain dépendant de la SIL » si la borne haute Wilson à 1,0B (20 000 épisodes) est sous
  la borne basse à 0,9B (2 048 épisodes), `--seed 0`, descriptif.
- Lectures fixées d'avance : (1) **D4 déclenché** : le gain ne persiste pas sans la SIL, même construit jusqu'à 0,8B ; piste SIL
  **close pour la ligne principale** (reste l'ablation GCO-ext « sans libération », étiquetée, nouvel ADR ; mécanisme non établi).
  (2) **D6 SUCCÈS** : variante candidate ; graines 1 et 2 (K1, G3) sur décision de l'utilisateur. (3) **D6 NEUTRE, D4 non déclenché** :
  le gain persiste sous le seuil ; non promue, aucun run automatique. Attendu a priori : non chiffré ; une graine (A9:210-211).

### D4. Kill criteria
K2 à K6 (ADR-005 D6, amendés par ADR-006) et KS1 à KS4 (A9:178-186) inchangés, mêmes seuils (`int_sil.yaml:89-95`).
- **KS1 (RAM < 1 Go pendant > 10 min)** : buffer plein jusqu'à 0,9B et libéré à 0,9B, comme en v1 : même empreinte, même durée (estimé,
  non mesuré). **Correction de l'avis du matin** : la RAM n'est pas plus exposée ; risque KS1 = celui de v1 (R:124-126), cause non établie.
- **Plus exposés : alerte C4 et KS2.** Le régime c_sil = 0,1 dure 0,3B de plus ; en v1, la médiane de disp_share montait encore
  (0,129 → 0,178 de 0,3 à 0,5B, puis 0,19 à 0,5-0,7B alors que c_sil baissait, R:111-114). Un dépassement durable de 0,5 (alerte) ou
  de 0,8 (KS2) n'est pas exclu (estimé, non mesuré). K4 : KL de tranche ≤ 8,6·10⁻⁴ en v1 (R:108-117), loin de 0,05.
- KS4 en alerte attendu jusqu'à 0,9B (31 à 34k/s, R:128) : au rapport, pas un arrêt.

### D5. Préalables (Definition of Done, avant le GO)
1. `configs/int_sil_late.yaml` : copie de `int_sil.yaml` ; seuls `name` (`int_sil_late`) et `sil.anneal_start_timesteps` diffèrent.
   Test qui le vérifie sur les YAML chargés, comparaison à `int_sil.yaml` (`SIL_LATE_DIFF`, YAML brut et `RunConfig.to_dict()`)
   dans `tests/unit/test_run_4a_interventions.py`. Test de c_sil(t) aux bornes : 0,1 à 0,8B, 0,05 à 0,85B, 0 à 0,9B.
2. `--variant int_sil_late` (`scripts/run_4a.py:72-74`), run `int_sil_late_s0`, éval `run_s1v2_eval.yaml` (test comme
   `test_run_4a_interventions.py:136-137`). 3. θ₀ identique à std s0 : SHA-256 7fab4c74…cb8a, méthode de la l. 414 du même test (R:57).
4. Revue `code-reviewer` ; smoke SIL **sur le commit du run** : `scripts/train.py configs/smoke_sil.yaml --smoke`, puis
   `scripts/sil_bench_summary.py --smoke` PASS (commande d'A9:230-232). Le code a changé depuis le banc (fusion d'ADR-010, 70c64ed).
5. **Pas de nouveau banc** : les bancs A9 restent valides (même c_sil 0,1, buffer, fenêtre, lot, mode). Ils ont mesuré le régime
   c_sil = 0,1 à buffer plein, celui qui s'allonge ici ; la règle de calibrage A9 D5 ne dépend pas du calendrier ; v1 a fourni RAM et
   débit sur 0,9B de SIL (R:124-130). Validité perdue si le smoke s'écarte de v1, ou si `src/training/sil.py` a changé hors revue.
   Vérifié le 2026-11-22 : `git log aedf0ca..ce980f0 -- src/training/sil.py` est vide. Seul `ppo.py` a changé (ea9cbf1, code aux
   inactif hors `aux`, digests des configs existantes inchangés, revue APPROVE).

### D6. Protocole, sorties et place dans le calendrier
- Commande (après GO) : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant int_sil_late --seed 0 --max-hours 0`.
- Évals hors ligne : commandes d'A9 D8 (A9:167-169), aux itérations de std s0 (0, 1 938, …, 19 376) :
  `scripts/eval.py --a <instantané> --b random --episodes 2048 --n-envs 256 --seed 0 --sprt-elo 100 --sprt-max-decisive 1200
  --out runs/int_sil_late_offline_eval/s0_b<0.k>_it<it>.json`, puis 20 000 épisodes à 1B (`s0_final20k_it19376.json`).
  Comparaisons descriptives : SIL v1 (même graine ; jusqu'à 0,5B, la configuration effective est la même, mais l'identité des
  trajectoires XPU n'est pas vérifiée), std s0, ADR-007.
- Coût : ≈ 8,5 h (estimé, non mesuré : 0,9B à ≈ 31,5k/s ≈ 7,9 h, plus 0,1B à 51k/s ≈ 0,5 h, débits R:128), plus les évals hors
  ligne (≈ 6 min en v1, 17:27-17:33 UTC, R:44). v1 : 8,14 h mesurées (R:24).
- Ordre : **après ADR-007** (lancé le 2026-11-22 à 05 h 05, ≈ 5,5 h estimé, `production/state.md:57-58`) et ses évals. Les préalables
  GPU d'ADR-010 (smoke `smoke_aux`, bancs λ, A10:163-166, automatisés) tournent juste après ADR-007, quel que soit son verdict. Si ADR-007
  est **NEUTRE** : ce run, puis ADR-010. Si ADR-007 est **SUCCÈS** : l'utilisateur tranche d'abord sur la base ; cet ADR reste proposé
  (une variante sur γ 0,998 mêlerait deux leviers). Amende l'ordre validé le 2026-11-21 (ADR-007:97-99 ; A10:206-211), qui plaçait la
  variante SIL après ADR-010 ; ADR-010 est retardé d'environ 8,5 h (estimé).

## Conséquences
- Plus facile : on répond à la question de persistance pour le coût d'une ligne de config, sans toucher `src/`. Un « D4 déclenché »
  clôt proprement la piste SIL pour la ligne principale.
- Plus difficile : on retarde ADR-010. Si le run est SUCCÈS, la confirmation sur 3 graines coûte ≈ 25 h (R:216, estimé, non mesuré).
- Menaces : une graine ; date du fondu choisie après avoir vu v1 (chemins multiples, limités par les critères inchangés et ce
  pré-enregistrement) ; le fondu raide (pente ×4) peut créer son propre choc ; mécanisme de la chute non séparé (oubli ou recalage de V).
- Révision si : alerte C4 ou KS2 pendant le plateau prolongé ; smoke ou débit hors de v1 ; ADR-007 SUCCÈS ; publication des codes
  Terminal ou Pulsar 2.5 (`GCO_SPEC.md` §8).

## Points soumis à l'utilisateur
1. Levier unique : `anneal_start_timesteps` passe à 0,8B (fin à 0,9B, PPO pur de 0,9B à 1B). Lecture de C2 identique à v1.
2. Critère ADR-006 D6 et diagnostic A9 D4 inchangés, avec les trois lectures de D3. Si D4 est déclenché, la piste SIL est close pour la
   ligne principale.
3. Ordre : après ADR-007 et seulement s'il est NEUTRE, avant ADR-010 (amende l'ordre du 2026-11-21). Les préalables GPU d'ADR-010
   passent juste après ADR-007.
4. Préalables : config, test, variante et smoke SIL sur le commit du run. Pas de nouveau banc (D5-5).
5. Kill criteria A9 inchangés. Le risque principal est l'alerte C4 ou KS2, pas KS1 (correction de l'avis du matin).
