# Rapport : `int_sil_s0` (ADR-009, GCO-std + SIL, graine 0, 1B)

- Date : 2026-11-21. Rédigé par `experiment-analyst` (brouillon pour `research-director` et l'utilisateur). Analyse de fichiers
  existants uniquement ; aucun entraînement ni évaluation lancés pour ce rapport. Les scripts de calcul sont jetables (hors dépôt).
- Run : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant int_sil --seed 0 --max-hours 0`
  (`runs/int_sil_s0/run_meta.json`, `--eval-config run_s1v2_eval.yaml`), commit a1a57c5, laptop Core Ultra 5 125H + Arc (XPU),
  torch 2.15.0+xpu. Début 2026-11-21 08:51:35 UTC (09:51 locale), dernière ligne de `metrics.jsonl` 17:26:22 UTC (18:26 locale)
  (`t` de `metrics.jsonl`, convertis).
- Conventions : « B » = milliard de timesteps appris ; m = `runs/int_sil_s0/metrics.jsonl` (19 376 lignes, sans trou),
  e = `evals.jsonl`, g = `segments.jsonl`, v = `events.jsonl`. Wilson 95 % bilatéral calculé par moi à partir des comptes des JSON.

## 1. Résumé
- **Verdict ADR-006 D6 : NEUTRE** (appliqué mécaniquement, §3). (i) échoue : 867 décisives / 20 000 = 4,33 % [Wilson 4,06 ; 4,63]
  < 6,28 % ; (ii) échoue : 1re paire H1 consécutive à 0,3B et 0,4B, pas ≤ 0,2B.
- Trajectoire hors ligne en cloche (2 048 épisodes par point) : décisives 1,32 % (0B) → 8,79 % (0,7B) → 3,71 % (1B). Pic
  aligné sur le calendrier de c_sil, retour vers le niveau de std s0 à 1B (post hoc, une graine, sans causalité).
- **Diagnostic de persistance pré-enregistré (ADR-009 D4) : déclenché** (« gain dépendant de la SIL », §5), descriptif, marge faible.
- Run sain : 1 segment `completed`, 0 restauration, 0 NaN, 0 alerte C4, KS1 à KS3 non déclenchés, **KS4 en alerte** (débit appris
  31 à 34k/s jusqu'à 0,9B).
- Base std conservée ; SIL v1 non promue (§8, propositions à valider).

## 2. Déroulement
- g : `status` completed, `restores` 0, `nan_detected` 0, `end_iteration` 19 376, `loop_s` 29 294,7 s (8,14 h), `mean_steps_per_s`
  37 930 (pas **collectés**/s, 1 111 154 688 collectés ; 1 000 014 736 appris). Durée entre 1re et dernière ligne de m : 8,58 h.
  Comparaison : std s0 5,51 h (`docs/reports/run_4a.md` §1).
- v : 81 lignes = 1 `start`, 79 `checkpoint` (tous `healthy: true`, dernier it. 19 376), 1 `s1_reached` à l'itération 6 001
  (mesure en cours de run, ADR-006 D2, décrite en §7, jamais utilisée pour le critère). Aucun événement `guard`, aucune alerte
  C4/KS dans v.
- m : `train/restores_total` 0 ; entropie minimale 4,166 (K2 non déclenché) ; KL approx max 4,3·10⁻³ (K4, seuil 0,05) ; EV minimale
  après l'itération 200 : 0,803.
- Aucune perte du GPU, aucune reprise (`resumed_from` null).

## 3. Critère ADR-006 D6 (verdict)
Texte : `docs/adr/006-protocole-s1-v2.md` l. 142-150 ; reprise à l'identique par ADR-009 D8.

| Volet | Règle | Mesure | Résultat |
|---|---|---|---|
| (i) | 1B, 20 000 épisodes `--seed 0` : borne basse Wilson de la part de décisives ≥ 6,28 % **et** SPRT +100 Elo H1 | 867 / 20 000 = 4,33 % [4,06 ; 4,63] ; SPRT H1 à 382 épisodes (14 V / 1 D sur 15 décisives) | **non rempli** (le SPRT est rempli, la part non) |
| (ii) | 1re paire consécutive H1 au plus tard à 0,2B (2 048 épisodes) | H1 à 0,3B (1 048 ép.) et 0,4B (1 340 ép.) ; 0,1B et 0,2B plafond sans décision | **non rempli** |

Source : `runs/int_sil_offline_eval/s0_final20k_it19376.json` et `s0_b*_it*.json`. Commande :
`.venv/Scripts/python scripts/eval.py --a <instantané> --b random --episodes N --n-envs 256 --seed 0 --sprt-elo 100
--sprt-max-decisive 1200` (N = 2 048, 20 000 à 1B), `.venv` CPU (torch 2.15.0+cpu), commit 042b335, arbre propre
(`git_dirty` false), 2026-11-21 17:27-17:33 UTC, thermique non contrôlé (`provenance` des JSON). Les instantanés viennent de
`checkpoints/int_sil_s0/eval_snapshots/` ; le commit d'entraînement des politiques est a1a57c5 (`policies.a.git_commit`).

**Verdict : NEUTRE** (ADR-006 D6 : configuration de base conservée, intervention suivante, un ADR chacune). Autres mesures du
20 000 : Elo décisif +248 [+220 ; +276] (bootstrap par paire), 699 V / 168 D / 19 133 nuls, buts pour 0,102 / min, contre 0,0245 / min,
A touche la balle dans 21,7 % des épisodes, aucune touche 73,7 %.
À 2 048 épisodes, seuls 0,7B et 0,8B ont une borne basse > 6,28 % (§4) : D6 mesure à 1B, ce ne sont pas des critères.

## 4. Trajectoire hors ligne : SIL contre std s0 (descriptif)
Même commande, `--seed 0`, 2 048 épisodes par point. SIL : commit 042b335 (2026-11-21). Std s0 : `runs/run_4a_offline_eval/s0_b*.json`,
commit 6db79a4 (2026-11-18). Les commits d'évaluation diffèrent : `git diff --stat 6db79a4 042b335 -- scripts/eval.py src/eval`
touche `scripts/eval.py` (21 lignes) et `src/eval/match.py` (55 lignes) : option `--p-kick` (15d1b12, défaut 1 « identique bit à
bit », digest figé d'avant l'option) et garde HORS SPEC hors évaluation (ff81e9b). Contrôle empirique (2026-11-21) : θ₀ identique
(SHA-256 des poids de politique 7fab4c747a778f6a… pour `run_4a_std_s0` et `int_sil_s0`, méthode de
`tests/unit/test_run_4a_interventions.py:414`) et JSON de l'itération 0 identiques entre les deux commits (`summary` et `touches`
égaux : 12 V / 15 D, 1,32 %, touche 6,6 %).

| B | Décisives SIL % [Wilson] | SPRT +100 SIL | Touche A SIL % | Décisives std % [Wilson] | SPRT std | Touche A std % |
|---:|---|---|---:|---|---|---:|
| 0,0 | 1,32 [0,91 ; 1,91] | plafond | 6,6 | 1,32 [0,91 ; 1,91] | plafond | 6,6 |
| 0,1 | 1,81 [1,31 ; 2,48] | plafond | 7,3 | 1,76 [1,27 ; 2,42] | H1 (1 884) | 7,2 |
| 0,2 | 1,95 [1,44 ; 2,65] | plafond | 8,5 | 1,46 [1,03 ; 2,08] | plafond | 7,3 |
| 0,3 | 2,54 [1,94 ; 3,31] | **H1** (1 048) | 9,6 | 1,51 [1,07 ; 2,14] | plafond | 7,0 |
| 0,4 | 3,08 [2,41 ; 3,92] | **H1** (1 340) | 13,3 | 1,81 [1,31 ; 2,48] | plafond | 9,6 |
| 0,5 | 6,01 [5,06 ; 7,12] | H1 (358) | 25,4 | 2,00 [1,48 ; 2,70] | H1 (1 742) | 8,1 |
| 0,6 | 6,93 [5,91 ; 8,12] | H1 (186) | 34,7 | 1,71 [1,23 ; 2,37] | H1 (584) | 9,0 |
| 0,7 | 8,79 [7,64 ; 10,09] | H1 (188) | 41,7 | 2,34 [1,77 ; 3,09] | H1 (712) | 10,2 |
| 0,8 | 8,25 [7,14 ; 9,52] | H1 (276) | 41,8 | 2,69 [2,07 ; 3,48] | H1 (1 260) | 10,8 |
| 0,9 | 5,71 [4,79 ; 6,80] | H1 (290) | 34,8 | 2,59 [1,98 ; 3,37] | H1 (500) | 11,3 |
| 1,0 | 3,71 [2,98 ; 4,62] | H1 (382) | 21,7 | 2,64 [2,03 ; 3,42] | H1 (652) | 13,9 |
| 1,0 (20 000) | **4,33 [4,06 ; 4,63]** | H1 (382) | 21,7 | 3,14 [2,91 ; 3,39] (`s0_final20k`, 6db79a4) | H1 (652) | 13,9 |

Valeurs complémentaires (SIL) : Elo décisif à 2 048 : −39 (0B), +202 (0,4B), +295 (0,5B), +585 (0,7B, [472 ; 792]), +432 (0,8B),
+377 (0,9B), +349 (1B). Durée moyenne d'épisode 20,5 à 20,9 s (troncature « 20 s sans touche »). « Aucune touche » : 88,5 % (0B),
55,2 % (0,7B), 74,3 % (1B).

Observations post hoc, une graine, sans causalité : la hausse suit c_sil à plein coefficient (0 → 0,5B : décisives 1,32 → 6,01 % ;
l'écart avec std devient visible dès 0,3-0,5B) ; le pic est à 0,7B (c = 0,044 en moyenne sur la tranche) ; la baisse commence
**avant** la libération (0,8B → 0,9B : [7,14 ; 9,52] contre [4,79 ; 6,80], intervalles séparés de justesse) et se poursuit à
1B. À 1B, la part de décisives (4,33 %) est au-dessus de std s0 (3,14 %, IC disjoints) mais sous le seuil de 6,28 %. Pour mémoire,
GCO-S s0 : 6,16 % [5,84 ; 6,50] à 1B (`docs/reports/gcos_s0.md`, commit f2218a6, autre arbre ; pas comparé par test).

## 5. Diagnostic de persistance (ADR-009 D4, pré-enregistré)
Formulation exacte (`docs/adr/009-intervention-sil.md` D4, dernière puce) : « Diagnostic de persistance : éval hors ligne à 0,9B et
1,0B (déjà dans D8) ; "gain dépendant de la SIL" si la borne haute (Wilson 95 %) de la part de décisives à 1,0B est sous la borne
basse à 0,9B (descriptif, au rapport). »

| Point | Décisives | Wilson 95 % | Source |
|---|---|---|---|
| 0,9B (it. 17 439), 2 048 ép. | 117 (105 V / 12 D) | [4,79 ; 6,80] % | `s0_b0.9_it17439.json` |
| 1,0B, 20 000 ép. | 867 | [4,06 ; 4,63] % | `s0_final20k_it19376.json` |
| 1,0B, 2 048 ép. (contrôle) | 76 | [2,98 ; 4,62] % | `s0_b1.0_it19376.json` |

Borne haute à 1B (4,63 %) < borne basse à 0,9B (4,79 %) : **la règle conclut « gain dépendant de la SIL »**. Même conclusion avec
l'éval à 2 048 (4,62 < 4,79). Statut : descriptif, une graine, marge de 0,16 point (avec 2 048 : 0,17). Cela ne change pas le
verdict D6 et ne prouve ni la cause ni la généralité.

## 6. Diagnostics SIL (m, tranches de 0,1B)
Provenance : calculs sur m, commit a1a57c5. `sil/coef` : moyenne ; `disp_share` (C4) et `grad_share` : médiane et max ; entropie, KL
approx (×10⁻⁴) : moyennes ; RAM : minimum de `sil/ram_available_mb` ; débit appris : 51 611 × (itérations − 1) ÷ (Δ `t`), checkpoints
et évaluations inclus (même méthode que `docs/reports/run_4a.md` annexe A).

| B | coef | disp_share méd / max | grad_share méd / max | Entropie | KL (×10⁻⁴) | RAM min (Mo) | Débit appris (k/s) | it. disp_share > 0,5 |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0,0-0,1 | 0,1000 | 0,138 / 0,589 | 0,068 / 0,203 | 4,452 | 8,55 | 3 356 | 31,7 | 10 |
| 0,1-0,2 | 0,1000 | 0,077 / 0,187 | 0,047 / 0,087 | 4,475 | 3,10 | 3 463 | 31,7 | 0 |
| 0,2-0,3 | 0,1000 | 0,093 / 0,320 | 0,060 / 0,150 | 4,475 | 2,26 | 3 363 | 31,7 | 0 |
| 0,3-0,4 | 0,1000 | 0,129 / 0,303 | 0,078 / 0,170 | 4,467 | 2,52 | 3 490 | 31,6 | 0 |
| 0,4-0,5 | 0,1000 | 0,178 / 0,414 | 0,115 / 0,281 | 4,447 | 2,79 | 3 560 | 31,5 | 0 |
| 0,5-0,6 | 0,0875 | 0,189 / 0,418 | 0,127 / 0,300 | 4,401 | 3,61 | 3 410 | 31,4 | 0 |
| 0,6-0,7 | 0,0625 | 0,190 / 0,403 | 0,112 / 0,262 | 4,392 | 3,08 | 3 733 | 31,5 | 0 |
| 0,7-0,8 | 0,0375 | 0,150 / 0,336 | 0,076 / 0,214 | 4,366 | 2,83 | 521 | 32,9 | 0 |
| 0,8-0,9 | 0,0125 | 0,081 / 0,279 | 0,036 / 0,149 | 4,337 | 2,67 | 2 765 | 34,4 | 0 |
| 0,9-1,0 | 0 | 0 / 0 | 0 / 0 | 4,273 | 2,41 | 143 | 51,0 | 0 |

- **C4** : 10 itérations avec disp_share > 0,5 sur 19 376 (it. 12 à 19 et 33, 34 ; max 0,589), la plus longue série **8** (< 50) :
  **pas d'alerte C4**. Elles surviennent pendant le remplissage du buffer (buffer 4 096 transitions environ à l'it. 12, 250 000
  atteint à l'it. 333). Sur les 17 439 itérations à c_sil > 0 : disp_share médiane 0,136, p95 0,264. **KS2** (> 0,8) : 0 itération.
  `sil/frac_pos` médiane 0,98 à 1,00 par tranche (SIL active) ; `buffer_generic_frac` moyenne 0,93 (0-0,1B) → 0,19 (0,8-0,9B).
- **Libération** : `sil/released` = 1 dès l'itération 17 440 (900,1M appris), buffer à 0 ; c_sil plein jusqu'à l'it. 9 688.
- **RAM / KS1** : 25 épisodes consécutifs sous 1 Go, le plus long **108 s** (< 10 min) : KS1 non déclenché. Un seul pendant la SIL
  (it. 14 736, 23 s, min 521 Mo, c_sil 0,035 ; c'est l'épisode signalé par l'utilisateur, attribué par lui à Chrome ≈ 5 Go :
  non vérifiable dans les journaux). Les 24 autres sont **après** la libération (0,9-1,0B), dont le minimum du run **143 Mo à
  l'it. 18 087** (it. 18 052 : 71 s, 18 753 : 108 s). Cause non établie. RAM disponible moyenne sur le run : 3 724 Mo.
- **KS4** (alerte seulement) : débit appris 31,4 à 34,4k/s de 0 à 0,9B (< 40k), 51,0k/s à 0,9-1,0B (SIL arrêtée). Pire fenêtre
  glissante de 30 min : 29 829 pas/s (it. 9 485 à 10 525), au-dessus du seuil K6 de 25k (non déclenché). Débit moyen sur le run
  32,4k/s appris. Comptage des fenêtres < 40k : non fait (toutes les tranches ≤ 0,9B sont sous 40k en moyenne). Banc ADR-009 : 32,6k/s.
- K2 (entropie min 4,166, loin du plancher), K4 (KL max 4,3·10⁻³), K5 (0 restauration), KS3 : non déclenchés (bornes hautes de
  l'Elo décisif > 0 à tous les points ; en cours de run, protocole v2 passé de 0,2B à 1B sauf k = 1).

## 7. Analyse post hoc 0,6B à 1B (fichiers existants, tranches de 0,05B)
Question : les buts d'entraînement suivent-ils c_sil (lecture « oubli ») ou tiennent-ils (lecture « spécialisation contre le pool »),
et y a-t-il un saut de KL à 0,9B ? Source m, commit a1a57c5 ; buts et fins par somme sur les itérations de la tranche (`env/goals`,
`env/ends_*`, tous les deux camps, tous les adversaires).

| B | coef | buts / it. | buts kickoff / it. | buts génériques / it. | buts / fins d'ép. | fins « aucune touche » % | disp_share méd | frac_pos méd | KL (×10⁻⁴) | Entropie | ‖g‖ avant clip |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,60 | 0,0688 | 10,98 | 9,06 | 1,92 | 11,72 % | 88,3 | 0,191 | 0,985 | 3,14 | 4,397 | 0,091 |
| 0,65 | 0,0563 | 11,70 | 9,82 | 1,88 | 12,46 % | 87,5 | 0,190 | 0,984 | 3,03 | 4,388 | 0,097 |
| 0,70 | 0,0438 | 11,77 | 9,87 | 1,90 | 12,53 % | 87,5 | 0,161 | 0,984 | 2,76 | 4,379 | 0,103 |
| 0,75 | 0,0313 | 11,93 | 10,03 | 1,90 | 12,69 % | 87,3 | 0,139 | 0,985 | 2,90 | 4,353 | 0,123 |
| 0,80 | 0,0188 | 11,95 | 10,00 | 1,95 | 12,70 % | 87,3 | 0,111 | 0,984 | 2,49 | 4,357 | 0,117 |
| 0,85 | 0,0063 | 11,20 | 9,27 | 1,93 | 11,94 % | 88,1 | 0,054 | 0,985 | 2,85 | 4,317 | 0,128 |
| 0,90 | 0 | 9,17 | 7,28 | 1,89 | 9,77 % | 90,2 | 0 | 0 | 2,53 | 4,276 | 0,154 |
| 0,95 | 0 | 8,72 | 6,83 | 1,89 | 9,27 % | 90,7 | 0 | 0 | 2,29 | 4,271 | 0,148 |

Les fins par origine (kickoff 80 %, génériques 20 %) sont constantes (`p_kick` 0,8). Fins « touchées » (`ends_touched`, comptées par
itération, pas en % ici) : non utilisées.

- **Buts d'entraînement.** Entre 0,6B et 0,8B, c_sil passe de 0,069 à 0,019 et les buts / itération **montent** (10,98 → 11,95) : ils
  ne suivent pas c_sil dans cette plage. Ils baissent ensuite de 0,8B à 0,95B (11,95 → 8,72, −27 %), quand c_sil passe de 0,019 à 0 :
  baisse concentrée sur les buts depuis kickoff (10,00 → 6,83), les buts depuis états génériques ne bougent pas (1,95 → 1,89).
  Les 2 observations coexistent : lecture « spécialisation » seule (buts qui tiennent) n'est pas soutenue après 0,8B ; la lecture
  « oubli » n'est soutenue que pour la fin de l'annealing et la libération (c ≲ 0,02). La politique d'entraînement marque moins
  de buts au kickoff, contre soi-même et le pool, en même temps que l'éval hors ligne contre l'aléatoire baisse (8,25 → 5,71 → 3,71 %).
  `env/goals` mélange buts marqués et encaissés dans les arènes : ce n'est pas une mesure de compétence contre un adversaire donné.
  Les buts par camp ou par type d'adversaire (pool contre auto-jeu) ne sont pas journalisés ; le taux de victoire contre le pool n'a
  pas été exploité (compteurs `pool/*` non traités). Conclusion : **compatible avec une SIL qui porte le niveau**, non établi.
- **KL / entropie à 0,9B.** Pas de saut de KL moyen à la libération (it. 17 440) : 3,07 (it. 17 240-17 339), 4,01 (17 340-17 439),
  3,36 (17 440-17 539), 3,33 (17 540-17 639), 3,08 (17 640-17 839), ×10⁻⁴ ; KL max sur les it. 17 400-17 500 : 1,05·10⁻³ (max du
  run 4,3·10⁻³). La fenêtre de 100 it. avant la libération est légèrement plus haute (4,01) ; l'entropie baisse de 4,299 à 4,243
  dans cette même fenêtre, **avant** la libération, puis 4,241, 4,203, 4,223 ensuite. Pas de choc Adam visible aux moyennes sur
  100 itérations (une fenêtre plus fine n'a pas été regardée). La norme de gradient avant clip monte de façon continue de 0,091 à
  0,148 de 0,6 à 0,95B (PPO seul à la fin) ; la clip frac baisse de 21,2 à 11,8 (×10⁻⁴).
- **Évaluations en cours de run (e, v2, 3 072 épisodes par évaluation, contre l'aléatoire ; autre échantillon, protocole ADR-006 D3,
  commit a1a57c5 ; décrites, jamais comparées aux hors ligne)** :

| k | It. | B | Décisives (V / D) | Décisives % | Elo décisif | Touche A | (a) / (b) |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 1 938 | 0,1 | 38 (18 / 20) | 1,24 | −18 | 5,4 % | non / non |
| 2 | 3 876 | 0,2 | 56 (40 / 16) | 1,82 | +159 | 7,9 % | oui / oui |
| 3 | 5 813 | 0,3 | 57 (36 / 21) | 1,86 | +94 | 8,9 % | oui / oui |
| 4 | 7 751 | 0,4 | 92 (71 / 21) | 2,99 | +212 | 14,3 % | oui / oui |
| 5 | 9 688 | 0,5 | 178 (155 / 23) | 5,79 | +331 | 27,1 % | oui / oui |
| 6 | 11 626 | 0,6 | 196 (174 / 22) | 6,38 | +359 | 34,9 % | oui / oui |
| 7 | 13 564 | 0,7 | 280 (253 / 27) | 9,11 | +389 | 41,9 % | oui / oui |
| 8 | 15 501 | 0,8 | 226 (203 / 23) | 7,36 | +378 | 41,6 % | oui / oui |
| 9 | 17 439 | 0,9 | 206 (182 / 24) | 6,71 | +352 | 35,1 % | oui / oui |
| 10 | 19 376 | 1,0 | 131 (105 / 26) | 4,26 | +242 | 21,4 % | oui / oui |

  La forme est la même en cloche (pic à 0,7B, retour à 4,26 % à 1B) sur un autre échantillon d'épisodes : la bosse ne vient donc pas
  du seul jeu d'épisodes hors ligne. `s1_reached` (v) : à l'itération 6 001, série (0,2B ; 0,3B) ; pour mémoire, hors ligne la 1re paire
  H1 est (0,3B ; 0,4B). Les différences d'échantillon et de protocole (3 072 contre 2 048 épisodes, aléatoire contre θ₀ pour les buts)
  interdisent de comparer ces deux mesures. Buts contre θ₀ (e `frozen`, k = 10) : 115 pour, 20 contre sur 1 049,8 min.

## 7 bis. Évaluations post hoc descriptives (2026-11-22, après la phase A d'ADR-012)
Ces évaluations ont été suggérées en §9 et lancées par `post_phaseA_pipeline.py` (scratchpad) sur le commit 86aa57a, en CPU
`.venv`, avec `scripts/eval.py --episodes 20000 --n-envs 256 --sprt-elo 100 --sprt-max-decisive 1200`. Les sorties sont dans
`runs/int_sil_offline_eval/posthoc_*.json`. Elles ne servent jamais de critère.

| Mesure | Décisives [Wilson] | V / D | Élo décisif [IC bootstrap] | Touche A |
|---|---|---|---|---|
| Tête-à-tête 0,8B (A) contre 1,0B (B), `--seed 0` | 11,33 % [10,90 ; 11,78] | 1 551 / 715 | **+135 [119 ; 149]**, SPRT H1 | 49,98 % (B : 34,69 %) |
| 0,8B contre l'aléatoire, `--seed 1` | 8,25 % [7,88 ; 8,64] | 1 493 / 158 | +390 | 41,17 % |
| 0,9B contre l'aléatoire, `--seed 1` | 6,49 % [6,16 ; 6,84] | 1 159 / 140 | +367 | 34,88 % |

- **Tête-à-tête** : 1,0B perd nettement contre 0,8B. D'après la grille de lecture de §9 (« si 1,0B perd, oubli »), c'est la
  lecture « oubli » qui est retenue, pas la spécialisation ni la non-transitivité. L'hypothèse reste post hoc, sur une graine.
- **Graine d'éval 1** : la bosse n'est pas un artefact de l'échantillon `--seed 0`. À 0,8B, 8,25 % contre 8,25 % avec 2 048
  épisodes `--seed 0` (`s0_b0.8_it15501.json`).
- **Diagnostic D4 resserré** : à 0,9B, la borne basse passe à 6,16 % avec 20 000 épisodes. Elle dépasse la borne haute à 1,0B
  (4,63 %) de 1,53 point, au lieu de 0,16 avant. La conclusion « gain dépendant de la SIL » tient, toujours descriptive.

## 8. Limites
- **Une graine** (0) ; std a une variance entre graines de ×1,5 à 1B (`docs/reports/run_4a.md` §4 bis, 3,14 / 2,55 / 2,24 %). Le
  bruit de graine ne peut pas être exclu sans autre graine ; une bosse ×2 alignée sur c_sil est difficile à expliquer par le seul bruit
  (estimé, non mesuré).
- **Post hoc** : trajectoire, persistance (D4 est pré-enregistré mais descriptif), §7 : hypothèses, pas de test de cause. Pas de correction
  de multiplicité sur les 11 points par courbe.
- Évaluations hors ligne SIL et std faites sur des commits différents (042b335 contre 6db79a4), même graine 0 ; équivalence
  vérifiée sur un seul instantané (θ₀, résultats identiques, §4) et par le test de digest de 15d1b12, pas sur les autres points.
- Débit appris 31 à 34k/s contre 50,2k/s pour std s0 : le run SIL a coûté 8,1 à 8,6 h contre 5,5 h, ce qui compte pour tout passage
  à 3 graines (estimé : ≈ 25 h pour 3 graines, non mesuré).
- RAM : un épisode à 521 Mo pendant la SIL (it. 14 736) et 24 après la libération (min 143 Mo) ; cause non établie, aucune action
  du run (KS1 non déclenché, durée max 108 s).
- Le tête-à-tête entre instantanés a été mesuré après coup (§7 bis : 0,8B bat 1,0B) ; la lecture « SIL porteuse » reste une hypothèse. Lecture concurrente
  (research-director) : la SIL peut agir comme un pas effectif plus grand (disp_share ≈ 0,1 à 0,2) ; la phase A d'ADR-012 (LR ×3)
  y répondra, hors de ce rapport.

## 9. Suite (propositions à valider par l'utilisateur ; rien n'est décidé ici)
- **ADR-006 D6, verdict neutre** : base std conservée ; SIL v1 non promue ; intervention suivante de `docs/GCO_SPEC.md` §8, un ADR
  chacune. ADR-009 Conséquences : un neutre ne réfute pas la SIL (une graine).
- **Ordre proposé par `research-director`** (avis du 2026-11-21) : (1) analyse post hoc de ce rapport (0 run, faite en §7) ;
  (2) fin de la phase A d'ADR-012 (en cours) et sa conclusion : si LR ×3 reproduit le gain, la piste SIL perd sa raison d'être ;
  (3) ADR-007 (γ 0,998, statut « proposé », à valider) ; (4) ADR-010 ; (5) variante SIL éventuelle, après. Exception :
  si l'analyse attribue nettement la chute à l'oubli et si LR ×3 ne reproduit pas la bosse, la variante serait remontée avant ADR-007,
  par décision de l'utilisateur. §7 n'attribue pas « nettement » ; l'exception n'est donc pas armée par ce rapport.
- **Variante SIL = nouvel ADR, validation utilisateur.** « Sans libération » (c_sil > 0 jusqu'à 1B) viole C2
  (`docs/GCO_SPEC.md:84` : « coefficient annealé à 0 avant la phase finale », vérifié) : elle ne serait qu'une **ablation GCO-ext
  étiquetée**, jamais la ligne principale, pour mesurer le plafond de la SIL. Variantes conformes C2 (research-director) : fondu plus
  lent et plus tôt (0,3 → 0,9B), ou SIL jusqu'à 1,8B sur un run de 2B (coût doublé, estimé).
- **Évaluations descriptives suggérées**, sur fichiers existants, **après la phase A** (CPU pris), jamais critères :
  tête-à-tête 0,8B contre 1,0B (`scripts/eval.py --a <it. 15 501> --b <it. 19 376>` : si 1,0B perd, oubli ; si 1,0B gagne ou fait nul,
  spécialisation ou non-transitivité) ; 20 000 épisodes `--seed 1` à 0,8B et 0,9B pour resserrer le diagnostic D4 (marge actuelle 0,16
  point). Les instantanés sont dans `checkpoints/int_sil_s0/eval_snapshots/`.

## Annexe. Fichiers sources
`runs/int_sil_s0/{metrics,evals,events,segments}.jsonl`, `runs/int_sil_s0/run_meta.json` ; `runs/int_sil_offline_eval/s0_b*_it*.json`
(12 fichiers dont `s0_final20k_it19376.json`) ; `runs/run_4a_offline_eval/s0_*.json` (std) ; `docs/adr/006-protocole-s1-v2.md` l. 142-150 ;
`docs/adr/009-intervention-sil.md` (D4, D5, D8, kill criteria) ; `docs/GCO_SPEC.md:84`.
