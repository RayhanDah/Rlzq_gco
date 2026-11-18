# ADR-008 — Intervention (1) : annealing de `p_kick` plus lent, 0,2 → 0,8 sur 1B (GCO-std, graine 0, 1B)

- Statut : **proposé, conditionnel au résultat de GCO-S graine 0 (ADR-006 D5)** : s'applique si GCO-S s0 est
  « S < std » (borne haute du Wilson 95 % des victoires par épisode < 1,60 %), sinon c'est ADR-007. Validation utilisateur
  requise avant tout lancement.
- Date : 2026-11-18
- Décideurs : rl-algo-lead (proposition) ; research-director ; utilisateur (validation, lancement).

## Contexte
- `docs/GCO_SPEC.md` §8 classe « distribution d'états initiaux génériques annealée » première intervention ; ADR-005 D6 et
  `docs/reports/run_4a.md` §8 la formulent « calendrier d'états génériques (p. ex. annealing plus lent) » sans la définir.
  ADR-006 D5 la pré-enregistre comme suite de GCO-S si « S < std », c'est-à-dire si les états génériques ont aidé
  (l'éval n'utilise que des kickoffs, ce qui avantage S : ADR-006 menace 5) ; elle cite « p. ex. 0,2 → 0,8 sur 1B ».
- Calendrier actuel (ADR-005 D4, `configs/run_4a_std.yaml`) : `p_kick` linéaire 0,2 → 0,8 sur les 0,5B premiers timesteps
  appris (9 688 itérations), puis 0,8 jusqu'à 1B. `p_kick` = probabilité qu'une réinitialisation d'épisode parte d'un
  kickoff ; 1 − `p_kick` = états génériques non ciblés (`GCO_SPEC.md` §2.3).
- Diagnostic (run 4a, `run_4a.md` §4 bis, §8) : le goulot est le **premier contact** (81 à 86 % des épisodes d'éval sans
  touche, l'agent touche dans 9 à 14 %). Les états génériques placent la balle près d'une voiture (§2.3) : c'est la voie,
  sans récompense ajoutée, par laquelle l'agent voit des touches et des buts. Après 0,5B, ils ne sont plus que 20 % des
  réinitialisations.

## Options
| Option | Pour | Contre |
|---|---|---|
| **A. 0,2 → 0,8 linéaire sur 1B** (`anneal_timesteps` 0,5B → 1B) | un seul champ ; même départ (0,2) et même arrivée (0,8 à la dernière itération) que std ; formule de `p_kick_at` inchangée (version de calendrier inchangée) ; citée par ADR-006 D5 | moins de kickoffs vus tôt, alors que l'éval est 100 % kickoff |
| B. Plateau bas : 0,2 jusqu'à 0,5B, puis 0,2 → 0,8 sur 0,5B | encore plus d'états génériques tôt | nouvelle formule et nouveau champ (code, version de calendrier) ; deux paramètres changés ; `p_kick` moyen 0,35, encore plus loin de l'éval |
| C. 0,2 → 0,8 sur 2B (au-delà du run) | pente encore plus faible | `p_kick` = 0,5 à 1B : la cible ≥ 0,8 de `GCO_SPEC.md` §2.3 n'est pas atteinte dans le run, et l'éval finale (kickoff) s'éloigne de la distribution d'entraînement |
| D. Cible plus basse (0,2 → 0,5) | idem | ce n'est pas un annealing « plus lent » mais une autre cible ; contraire au §2.3 |

## Mesures
Aucune mesure d'entraînement : décision réversible, fondée sur `GCO_SPEC.md` §2.3 et §8, ADR-005 D4, ADR-006 D5 et le
diagnostic de `docs/reports/run_4a.md` §4 bis et §8. Valeurs de `p_kick` calculées par `RunConfig.p_kick_at` sur les deux
configs (commande dans le test `tests/unit/test_run_4a_interventions.py`, 2026-11-18) :

| Appris | 0 | 0,1B | 0,2B | 0,3B | 0,4B | 0,5B | 0,6B | 0,7B | 0,8B | 0,9B | 1B (dernière it.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| std (ADR-005 D4) | 0,20 | 0,32 | 0,44 | 0,56 | 0,68 | 0,80 | 0,80 | 0,80 | 0,80 | 0,80 | 0,80 |
| **lent (option A)** | 0,20 | 0,26 | 0,32 | 0,38 | 0,44 | 0,50 | 0,56 | 0,62 | 0,68 | 0,74 | 0,80 |

`p_kick` moyen sur le run : 0,65 (std) contre 0,50 (lent) ; part moyenne d'états génériques 0,35 → 0,50 (×1,43), en
proportion des réinitialisations (pas des timesteps : la durée d'épisode dépend de l'origine ; journalisée par I4).

## Décision

### D1. Définition opérationnelle unique de « annealing plus lent »
**Option A** : `p_kick` linéaire de 0,2 à 0,8 sur **1B** timesteps appris, soit toute la durée du run ; 0,8 exactement à la
dernière itération (19 375, `anneal_iterations` = `total_iterations` = 19 376). Pente divisée par 2 ; `p_kick` ≤ celui de std
à chaque itération ; égal aux deux extrémités.
`configs/int_slow_anneal.yaml` = `configs/run_4a_std.yaml` avec **`initial_states.anneal_timesteps: 1000000000`** (hors
`name` et commentaires ; vérifié par `tests/unit/test_run_4a_interventions.py`). Aucun code changé : la formule
`RunConfig.p_kick_at` et `P_KICK_SCHEDULE_VERSION` sont inchangées.

### D2. Justification
- **Spec** : `GCO_SPEC.md` §2.3 n'impose que le départ (0,2), la forme (linéaire) et la cible (≥ 0,8 avant toute promotion
  finale) : A garde les trois ; B change la forme, C et D la cible. États génériques non ciblés uniquement (H-GCO-std) ;
  aucune récompense, aucun signal ajouté ; C1–C4 non sollicités.
- **Diagnostic** : si GCO-S fait moins bien que std (D5), le levier plausible est la masse d'états génériques ; A l'augmente
  à chaque point du run (×1,43 en moyenne) au moment où std la réduisait (après 0,5B, 20 % seulement), sans changer la
  distribution finale (0,8 à 1B, comme std, donc éval finale comparable).
- **Un seul levier** (`.claude/rules/training.md`) : A est le seul candidat qui tient en un champ.

### D3. Exécution (après GO utilisateur seulement)
- Prérequis (CLAUDE.md règle 5) : smoke test vert **sur le commit du run** (`scripts/train.py configs/smoke.yaml --smoke`).
- Commande : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant int_slow_anneal --seed 0 --max-hours 0`
  (run `int_slow_anneal_s0` ; éval en cours de run par défaut `configs/run_s1v2_eval.yaml`, protocole ADR-006 D2, D3).
  Pas de `--std-s1-observed` : c'est un GCO-std. Mêmes garanties que `--variant std` : sessions enfants, reprise après
  `DEVICE_LOST`, refus après STOP, arbre propre.
- Coût : ≈ 5,5 h (durée std mesurée, transposée ; estimé, non mesuré ; la part d'états génériques peut changer le débit).
- Diagnostic obligatoire (ADR-006 D7 I4, déjà implémenté) : `env/goals_kickoff`, `env/goals_generic`, `env/ends_*` et
  `p_kick` par itération dans `metrics.jsonl`, rapportés par tranche de 0,1B à côté de std s0.

### D4. Comparaison à std s0 et critère de succès
- Comparaison **uniquement** par la commande hors ligne d'ADR-006 D4, identique à `runs/run_4a_offline_eval/s0_*` : pour
  chaque instantané à 0,1B (`checkpoints/int_slow_anneal_s0/eval_snapshots/ckpt_*_it<it>.pt`,
  aux mêmes itérations que std s0 : 0, 1 938, 3 876, 5 813, 7 751, 9 688, 11 626, 13 564, 15 501,
  17 439, 19 376)
  `scripts/eval.py --a <instantané> --b random --episodes 2048 --n-envs 256 --seed 0 --sprt-elo 100 --sprt-max-decisive 1200
  --out runs/int_slow_anneal_offline_eval/s0_b<0.k>_it<it>.json`, puis à 1B la même commande avec `--episodes 20000`
  (`s0_final20k_it19376.json`). Les évals en cours de run (v2) sont décrites au rapport, jamais comparées entre runs.
- **Succès (ADR-006 D6)** si l'un des deux :
  (i) à 1B, 20 000 épisodes contre l'aléatoire `--seed 0` : borne basse du Wilson bilatéral 95 % de la part de décisives
  ≥ **6,28 %** **et** SPRT +100 Elo accepte H1 sur la même éval ;
  (ii) 1re paire de points consécutifs avec H1 sur les évals hors ligne à 2 048 épisodes `--seed 0` au plus tard à **0,2B**
  (paire complète ≤ 0,3B).
  Référence std s0 : 3,14 % de décisives à 1B, 1re paire à 0,5B (ni (i) ni (ii)).
- **Neutre** sinon : calendrier d'ADR-005 D4 conservé, intervention suivante de `GCO_SPEC.md` §8 (ADR-007, γ 0,998), un ADR
  chacune.

## Conséquences
- Plus facile : un seul champ, aucun code, distribution finale identique à std.
- **Risques et biais** :
  1. **biais contre l'intervention sur (ii)** : à 0,2B, `p_kick` vaut 0,32 contre 0,44 pour std ; l'éval (100 % kickoff)
     mesure une compétence moins entraînée. Un échec de (ii) est donc peu informatif ; (i) à 1B l'est davantage ;
  2. à 1B, l'agent a vu moins de kickoffs au total (≈ 0,50 contre 0,65 des réinitialisations) : (i) aussi est biaisé
     contre A, plus faiblement ;
  3. une graine : la variance entre graines std (×1,5 sur les victoires à 1B) domine ;
  4. GCO-S « < std » ne prouve pas que les génériques aident (une graine, ADR-006 D5 « défaut assumé ») : l'intervention
     teste aussi cette lecture.
- Révision si : buts d'origine générique (I4) ne montent pas sur le run (la masse ajoutée ne produit pas de buts), ou succès
  sur (i) seulement avec part de décisives sans SPRT ; option B ou C alors, par un nouvel ADR.

## Points à valider par l'utilisateur
1. Application conditionnelle : seulement si GCO-S s0 est « S < std » (ADR-006 D5).
2. Définition unique : 0,2 → 0,8 linéaire sur 1B (option A), seul champ `anneal_timesteps` modifié.
3. Commande de lancement D3, smoke test sur le commit du run d'abord.
