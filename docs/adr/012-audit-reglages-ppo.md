# ADR-012 — Audit des réglages PPO, un levier : taux d'apprentissage ×3 (témoin dense d'abord, GCO-std ensuite si succès)

- Statut : **accepté** (validation de l'utilisateur le 2026-11-21, les 5 points en fin d'ADR ; suite prévue par ADR-011 D3
  pour le verdict AMBIGU). **Phase A faite : verdict D3 NEUTRE (2026-11-22), audit clos, pas de phase B.**
- Date : 2026-11-21
- Décideurs : research-director (proposition) ; experiment-analyst (P1) ; rl-algo-lead (config, variante du lanceur, tests) ;
  eval-lead (évals hors ligne) ; code-reviewer ; utilisateur (validation, lancement)

## Contexte
- Verdict D3 du témoin dense `diag_dense_s0` (ADR-011, HORS SPEC) : **AMBIGU**. À 1B, A touche dans 50,44 % des épisodes
  [Wilson 49,75 ; 51,13], SPRT +100 Elo H1 ; à 0,5B, 23,0 % [22,4 ; 23,6] (mesures ci-dessous). Le pipeline apprend le contact
  quand le signal existe, mais lentement : ×2,2 entre 0,5B et 1B, et 1B pour franchir à peine 50 %.
- ADR-011 D3 prévoit alors un « ADR d'audit des réglages PPO (un levier), avant ou en parallèle de la SIL ». L'utilisateur a
  décidé le 2026-11-21 que le run SIL (ADR-009) passe ensuite « quoi qu'il arrive » : cet audit vient **après** la SIL.
- Le soupçon d'ADR-011 (« KL 10⁻⁴, clip < 0,1 % en 4a ») est confirmé et précisé par les métriques du témoin : les mises à jour
  restent très en deçà de la région de confiance que PPO autorise (clip jamais actif), et l'entropie ne baisse vraiment qu'en
  seconde moitié de run, quand **LR et coefficient d'entropie décroissent ensemble**. Les journaux ne peuvent pas séparer ces
  deux causes (calendriers linéaires sur la même progression, `src/training/ppo.py:87-93`). Seule une intervention le peut.
- Coût d'une erreur : chaque run GCO coûte ≈ 5,5 h (mesuré en 4a). Si l'optimisation bride l'apprentissage même avec un signal
  dense, elle bride a fortiori la SIL et les interventions suivantes.

## Mesures
Réglages communs (identiques dans `configs/run_4a_std.yaml` et `configs/diag_dense.yaml`, seule différence `reward.variant`) :
LR 2e-4 → 5e-5 et coefficient d'entropie 0,01 → 0,002, tous deux linéaires sur 19 376 itérations (`ppo.py:87-93`) ; clip 0,2 ;
2 epochs × 3 minibatchs = **6 pas d'Adam par itération** de 51 611 pas appris (`ppo/n_minibatches` = 3) ; avantages normalisés
par minibatch (`ppo.py:326`) ; π et V séparés (π 3×256, `src/models/policy.py:17` ; V 3×512, `src/models/value.py:15`), **un seul
Adam et un clip global 0,5 communs** (`ppo.py:334-341`) ; tête de π à gain 0,01 (`policy.py:19`).

### M1. Témoin dense `diag_dense_s0` : valeurs d'**une itération** (pas des moyennes de tranche)
Source : `runs/diag_dense_s0/metrics.jsonl`, ligne N = itération N (vérifié sur les 10 lignes lues) ; run sur e04871e
(`production/state.md`), XPU, laptop Core Ultra 5 125H, 2026-11-20/21. Touche : éval en cours de run contre l'aléatoire,
3 072 épisodes (`runs/diag_dense_s0/evals.jsonl`, 2e bloc `touches` de chaque ligne, le 1er étant contre θ₀ ; 17,7 % à 0,4B
recoupe `state.md`). ln 90 = 4,4998.

| B | itér. | LR | ent_coef | entropie | ln 90 − H | KL approx | clip frac | ‖g‖ avant clip | EV | adv_std brut | A touche |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0,1 | 1 938 | 1,85e-4 | 0,0092 | 4,455 | 0,045 | 3,2e-4 | 0 | 0,053 | 0,866 | 0,024 | 7,1 % |
| 0,2 | 3 876 | 1,70e-4 | 0,0084 | 4,438 | 0,061 | 4,1e-4 | 6,5e-4 | 0,062 | 0,946 | 0,021 | 12,2 % |
| 0,3 | 5 813 | 1,55e-4 | 0,0076 | 4,375 | 0,125 | 4,9e-4 | 1,5e-4 | 0,070 | 0,971 | 0,020 | 11,8 % |
| 0,4 | 7 751 | 1,40e-4 | 0,0068 | 4,286 | 0,214 | 4,2e-4 | 8,1e-4 | 0,079 | 0,955 | 0,025 | 17,7 % |
| 0,5 | 9 688 | 1,25e-4 | 0,0060 | 4,277 | 0,223 | 1,04e-3 | 7,4e-3 | 0,103 | 0,953 | 0,023 | 23,2 % |
| 0,6 | 11 626 | 1,10e-4 | 0,0052 | 4,175 | 0,324 | 1,13e-3 | 6,8e-3 | 0,121 | 0,974 | 0,021 | 26,1 % |
| 0,7 | 13 564 | 9,5e-5 | 0,0044 | 4,038 | 0,462 | 1,30e-3 | 8,7e-3 | 0,163 | 0,943 | 0,031 | 30,1 % |
| 0,8 | 15 501 | 8,0e-5 | 0,0036 | 3,998 | 0,501 | 5,1e-4 | 1,2e-3 | 0,151 | 0,952 | 0,029 | 39,8 % |
| 0,9 | 17 439 | 6,5e-5 | 0,0028 | 3,871 | 0,629 | 8,6e-4 | 3,5e-3 | 0,181 | 0,945 | 0,034 | 40,3 % |
| 1,0 | 19 376 | 5,0e-5 | 0,0020 | 3,591 | 0,909 | 1,39e-3 | 6,3e-3 | 0,215 | 0,935 | 0,035 | 51,5 % |

### M2. GCO-std s0 (référence 4a)
- Moyennes par tranche (`docs/reports/run_4a.md` annexe A, `runs/run_4a_std_s0/metrics.jsonl`) : entropie 4,457 → 4,357 ;
  KL 1,3 à 6,6·10⁻⁴ ; clip frac 0,3 à 10,3·10⁻⁴ ; ‖g‖ avant clip 0,046 à 0,092 ; EV 0,82 → 0,96.
- Valeurs d'une itération, même fichier : it. 9 688 (l. 9 688) entropie 4,478, KL 2,0e-5, clip 0, ‖g‖ 0,049, EV 0,918 ;
  it. 19 376 (l. 19 376) entropie 4,334, KL 4,2e-5, clip 0, ‖g‖ 0,093, EV 0,957, adv_std brut 0,020. **Une valeur
  d'itération peut s'écarter d'un ordre de grandeur de la moyenne de tranche** (KL 2,0e-5 contre 2,34e-4 sur 0,4–0,5B) : M1
  est une trajectoire indicative, pas une statistique (d'où P1).
- Éval hors ligne à 1B, 20 000 épisodes (`run_4a.md` §4 bis, 6db79a4) : A touche 13,9 %, aucune touche 81,0 %, Elo décisif +203.
  Touche à 1B sur les 3 graines std : 8,7 / 10,3 / 13,9 % (rapport max/min **1,60**, seule mesure de variance entre graines du
  taux de touche).

### M3. Verdict D3 du témoin
Commande : `scripts/eval.py --a <instantané> --b random --episodes 20000 --n-envs 256 --seed 0 --sprt-elo 100
--sprt-max-decisive 1200` ; commit 6ae65d6, `.venv` CPU, laptop Core Ultra 5 125H, 2026-11-21, `git_dirty` faux.
- 0,5B (it. 9 688) : A touche 23,0 % [22,4 ; 23,6], aucune touche 72,9 %, 593 V / 19 101 N / 306 D, SPRT H1, 351 s
  (`runs/diag_dense_offline_eval/dense_s0_final20k_it09688.json`).
- 1B (it. 19 376) : A touche 50,44 % [49,75 ; 51,13], aucune touche 46,97 %, 1 948 V / 17 775 N / 277 D, Elo décisif +339
  [+317 ; +362], SPRT H1 après 116 épisodes, 237 s (`…_it19376.json`).
- Trajectoire hors ligne de référence, 2 048 épisodes `--seed 0` : `runs/diag_dense_offline_eval/dense_s0_traj2048_it*.json`
  (22 à 26 s par point).

### M4. Recette publique (comparaison qualitative seulement)
- Guide RLGym (lu le 2026-11-18, ADR-011 Mesures) : LR 1e-4 constant, entropie 0,01 constante, lot 100 000, minibatch 50 000,
  2 epochs, réseaux 2048-2048-1024-1024 ; entropie qui « should settle around 2 ».
- `rlgym-ppo`, branche main lue le 2026-11-21 (commit non relevé) : `ppo/ppo_learner.py` accumule les gradients des minibatchs et
  fait **un pas d'optimiseur par lot** ; π et V ont chacun leur optimiseur et leur clip 0,5 ; **aucune normalisation des
  avantages** dans l'apprenant ; `ppo/discrete_policy.py` : entropie de Shannon en nats, non normalisée.
- Calcul : si le lot PPO couvre toute l'itération (non vérifié), la recette fait 2 pas d'Adam par 100 000 échantillons à 1e-4,
  contre ≈ 11,6 pour nous (6 / 51 611) à 2e-4 → 5e-5, mais sur une π 8× plus large par couche. Sans normalisation des avantages
  ni réseau comparable, **on ne peut pas en déduire si notre LR est trop haut ou trop bas**. Rien n'est transféré.

### Lecture (calculée sur M1, descriptive)
1. **Amplitude** : KL ≤ 1,4·10⁻³ et clip frac ≤ 0,9 % dans le témoin dense, ≤ 10⁻³ en std. La région de confiance (clip 0,2)
   n'est jamais atteinte ; l'alarme usuelle est à 0,02 (`docs/01_papers.md` §1.1, 37 details), K4 à 0,05.
2. **KL en hausse pendant que la LR baisse 3,7×** (3,2e-4 → 1,4e-3) ; ‖g‖ ×4 (0,053 → 0,215) : le gradient devient plus
   cohérent quand l'agent touche ; rien ne sature.
3. **Confusion LR / entropie** : de 0,6B à 1B, (ln 90 − H) × ent_coef reste dans 1,7–2,0·10⁻³ et (ln 90 − H) × LR dans
   3,6–4,5·10⁻⁵ (5 points d'itération). Les deux lectures « l'entropie suit l'équilibre fixé par ent_coef » et « la politique
   avance au rythme de la LR » collent aux données ; LR et ent_coef sont presque proportionnels (rapport 0,020 → 0,025).
4. **GCO** : à 1B, std s0 garde H = 4,33 (96 % de ln 90) avec ent_coef 0,002 ; la politique reste près de l'uniforme, où le
   gradient de l'entropie s'annule.

## Options
| Option (un levier) | Pour | Contre |
|---|---|---|
| **LR ×3 (6e-4 → 1,5e-4, même forme)** | agit directement sur l'amplitude, le soupçon nommé par ADR-011 D3 ; coût de calcul nul ; une seule valeur de config ; in-spec partout ; agit aussi en GCO, où la politique est quasi uniforme | hors de la plage d'ablation de `01_papers.md` §2 ({1e-4, 3e-4}) ; amplifie aussi le bruit (marche aléatoire en GCO, estimé) ; risque d'instabilité (s0 a déjà eu un NaN, cause inconnue) |
| LR ×1,5 (3e-4, valeur pré-listée §2) | dans la plage prévue, sûr | KL ×2,25 attendu (estimé) : effet probablement trop faible pour être vu sur une graine |
| ent_coef ÷2 (0,005 → 0,001) | valeur 0,005 pré-listée (§2) ; lecture 3 compatible ; coût nul | en GCO la politique est quasi uniforme, là où le gradient de l'entropie s'annule : levier presque inerte dans la ligne principale (estimé) ; contre l'avis « entropie fondamentale en GCO » (§2) ; Five App. O.1 : 0,01 meilleur tôt |
| Epochs 2 → 6 ou minibatchs 3 → 9 (×3 pas d'Adam) | même effet attendu que LR ×3 | coût d'apprenant non mesuré (recouvrement collecte/apprentissage) ; réutilisation : Five §4.4, 2–3 ⇒ ≈ 2× plus lent ; plus hors politique |
| Clip 0,2 → 0,3 | trivial | inerte : le clip ne mord jamais (≤ 0,9 %, M1) |
| Retirer la normalisation des avantages | rapproche de `rlgym-ppo` | sens inverse : multiplie l'échelle du gradient de politique par l'adv_std brut ≈ 0,02–0,035, donc l'entropie domine 30 à 50 fois plus ; `01_papers.md` §1.1 la juge indispensable en GCO |
| Lot plus grand (`steps_per_batch` 7 → 14) | moins de bruit par pas | moitié moins de pas par échantillon (amplitude par échantillon en baisse) ; RAM (ADR-001 M8 : 2,73 Go disponibles au minimum) ; change aussi la fenêtre GAE (T = 7) : deux effets |
| Deux Adam et deux clips π / V (comme `rlgym-ppo`) | découple V de π | changement de code ; Adam est presque invariant à l'échelle du gradient et ‖g‖ < 0,5 en moyenne : effet attendu faible (estimé) |
| π plus large (3×256 → 2048-…) | rapproche de la recette | architecture, pas réglage ; débit de collecte CPU en baisse (non mesuré) ; plusieurs leviers (init, LR effective) |

## Décision (proposée)

### D1. Levier unique : facteur multiplicatif ×3 sur le calendrier de LR
- `ppo.lr_start` 2e-4 → **6e-4**, `ppo.lr_end` 5e-5 → **1,5e-4** ; forme linéaire et longueur (19 376 itérations) inchangées.
  Tout le reste identique à la base comparée (test exigé : seules ces deux clés diffèrent).
- Pourquoi ×3 : sous l'approximation du second ordre (KL ≈ ½ Δθᵀ F Δθ, Δθ ∝ LR sous Adam), KL ∝ LR², donc ×9 (**estimé, non
  mesuré**) : le pic d'itération du témoin (1,4e-3) passerait à ≈ 1,3e-2, sous l'alarme 0,02 et sous K4. ×10 donnerait ≈ 0,14 et
  déclencherait K4. ×1,5 (§2) serait sans doute invisible sur une graine.
- **Règle de repli, fixée avant le run** : si P1 trouve une moyenne de tranche de 0,1B de `ppo/approx_kl` du témoin telle que
  9 × moyenne > 0,02, le facteur passe à **×2** (4e-4 → 1e-4), consigné ici avant le GO.
- Écart à `01_papers.md` §2 (ablation {1e-4, 3e-4}) assumé : justifié par une mesure (KL ≥ 10× sous l'alarme), comme l'exige
  `.claude/rules/training.md` (« une source ou une mesure »).

### D2. Phase A : diagnostic sur le témoin dense (HORS SPEC)
- Pourquoi le témoin et pas le GCO d'abord : c'est là que l'AMBIGU est apparu ; le taux de touche y répond vite et se mesure
  à ± 0,7 point sur 20 000 épisodes ; en GCO le signal est trop rare pour qu'un réglage d'optimisation se lise sur une graine.
- Run `diag_dense_lr3_s0` : `configs/diag_dense.yaml` + D1, **graine 0**, récompense `dense-diag` inchangée (hash
  275f3f883e786a28), isolation d'ADR-011 D2 (HORS SPEC, `in_spec=False`, jamais promu, aucun checkpoint vers un run GCO).
- Budget : **0,5B appris**, arrêt à l'itération 9 688. La config garde 1B pour que la LR à l'itération t soit exactement 3× celle
  de la base (pas de compression du calendrier). Commande : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant
  diag_dense_lr3 --seed 0 --max-hours 0 --max-iterations <9 688 ou 9 689>` (`--max-iterations` = plafond absolu du compteur,
  `scripts/run_4a.py:356` ; rl-algo-lead vérifie au banc que l'instantané d'éval de l'itération 9 688 est bien écrit).
- Comparaison **hors ligne seulement**, commande D3 d'ADR-011 sur l'instantané de l'itération 9 688 : `scripts/eval.py --a
  <instantané> --b random --episodes 20000 --n-envs 256 --seed 0 --sprt-elo 100 --sprt-max-decisive 1200`, contre
  `dense_s0_final20k_it09688.json`. Trajectoire descriptive : 2 048 épisodes `--seed 0` aux itérations 1 938 … 9 688, contre
  `dense_s0_traj2048_it*.json`. Les évals en cours de run sont décrites, jamais comparées.

### D3. Critère pré-enregistré (Phase A, à 0,5B, 20 000 épisodes)
| Verdict | Condition | Lecture | Suite |
|---|---|---|---|
| **SUCCÈS** | borne basse du Wilson 95 % de « A touche » ≥ **46,0 %** (2 × 23,0 %) **et** SPRT +100 Elo H1 | la LR bridait l'apprentissage : ×3 atteint à 0,5B le niveau de la base à 1B | Phase B (D4), sur GO |
| **NUISIBLE** | borne haute < **22,4 %** (borne basse de la base), **ou** moyenne de `ppo/approx_kl` > 0,02 sur une tranche de 0,1B, **ou** STOP K2 / K5 | instabilité ou perte nette | LR inchangée ; audit clos |
| **NEUTRE** | sinon | la LR n'est pas le goulot à cette échelle | audit clos ; ent_coef devient l'hypothèse de tête, sans ADR automatique |
- Pourquoi 2× : une graine par bras ; la seule variance entre graines mesurée pour la touche est un rapport de 1,60 (M2). Un
  seuil au-dessus évite qu'un succès s'explique par cette dispersion seule. Limite : la variance entre graines du témoin dense
  n'est pas mesurée.
- Diagnostics joints (descriptifs) : tableau par tranche (KL, clip, entropie, ‖g‖, EV, adv_std) contre P1 ; rapport des KL des
  deux runs (test de l'hypothèse KL ∝ LR², attendu ≈ 9) ; (ln 90 − H) à 0,5B. Prédictions : si la LR limite, touche ≥ 2× et
  entropie plus basse ; si l'équilibre de l'entropie limite, touche ≈ 23 % et (ln 90 − H) ≈ 0,22.
- Un SUCCÈS ne prouve rien pour le GCO (signal ≈ 4 à 7 buts par itération, gradient dominé par le bruit) : d'où D4.

### D4. Phase B (seulement si SUCCÈS et GO de l'utilisateur) : GCO-std, in-spec
- Run `int_lr3_s0` : `configs/run_4a_std.yaml` + D1 (seule différence, testée), GCO-std, graine 0, 1B, éval en cours de run
  `run_s1v2_eval.yaml`. Critère **ADR-006 D6 inchangé**, comparaison hors ligne à std s0 par la commande d'ADR-009 D8
  (2 048 épisodes par 0,1B, 20 000 à 1B, `--seed 0`). Kill criteria K2 à K6 (ADR-005 D6, amendés par ADR-006), plus la règle KL
  de D3 au rapport. Coût ≈ 5,5 h (durée 4a mesurée, transposée, estimé).
- Si succès : LR ×3 devient candidate pour la configuration de base (nouvel ADR) ; toute combinaison avec la SIL est un autre
  levier (un ADR, une comparaison à la fois).

### D5. Kill criteria (Phase A)
K2 à K6 d'ADR-005 D6 inchangés (K4 : KL > 0,05 pendant 5 itérations ⇒ RESTORE ; K5 : > 3 restaurations ⇒ STOP), plus la règle
KL de D3 (lue après le run, sans code). Pas d'arrêt anticipé pour succès. Une perte du GPU est reprise automatiquement (ADR-005,
2026-11-15) : run « repris ».

### D6. Conformité GCO
- La LR est un réglage d'optimisation : ni récompense, ni signal injecté dans les avantages. Aucun terme ajouté ; C1–C4 non
  sollicités (même statut que γ et `p_kick` dans ADR-006 D5 ; `01_papers.md` §1.1 : « optimisation pure »). Valable en GCO-S,
  GCO-std et GCO-ext.
- Phase A est **hors spec à cause de la récompense dense** (ADR-011), pas à cause du levier : diagnostic, jamais promu. Phase B
  est in-spec (GCO-std).

### D7. Préalables (aucun calcul long)
- **P1** (experiment-analyst) : tableau par tranche de 0,1B de `runs/diag_dense_s0/metrics.jsonl` au format de l'annexe A de
  4a, avec le maximum par tranche de `ppo/approx_kl` ; applique la règle de repli de D1.
- **P2** (rl-algo-lead) : `configs/diag_dense_lr3.yaml`, variante `--variant diag_dense_lr3` ; test « seules `ppo.lr_start` et
  `ppo.lr_end` diffèrent de `diag_dense.yaml` » ; hash de récompense inchangé ; θ₀ identique au témoin (SHA-256 des poids de
  l'itération 0, comme ADR-006).
- **P3** : revue `code-reviewer` ; smoke test vert sur le commit du run (CLAUDE.md règle 5).
- **Résultat P1 (2026-11-21, experiment-analyst, lecture seule de `runs/diag_dense_s0/metrics.jsonl`, 19 376 itérations
  contiguës, HEAD a1a57c5)** : moyenne de tranche de `ppo/approx_kl` maximale **1,20·10⁻³** (tranche 0–0,1B ; autres tranches
  6,4·10⁻⁴ à 1,03·10⁻³) ; 9 × 1,20·10⁻³ = 0,0108 < 0,02, donc **×3 confirmé**, pas de repli. Maximum d'une itération : 4,43·10⁻²
  dans la tranche 0–0,1B (itérations de démarrage), ≤ 5,25·10⁻³ ailleurs. Une graine : constat descriptif.

### D8. Place dans le calendrier
- **Après le run SIL** (≈ 5,5 h, estimé) et ses évals hors ligne. Phase A ≈ **3,3 h** : 0,5B ≈ 3,1 h (session 1 du témoin :
  0,478B en 3 h, mesuré, `state.md`) + 20 000 épisodes ≈ 6 min (351 s mesurés à 0,5B) + 5 points de 2 048 ≈ 2 min. Ensuite, si
  SUCCÈS, Phase B (≈ 5,5 h) **avant** ADR-007, car elle peut changer la base de toutes les interventions suivantes (amende l'ordre
  d'ADR-009 D9 : validation utilisateur). Si NEUTRE ou NUISIBLE : retour à l'ordre D9 (ADR-007, puis ADR-010).
- En parallèle de la SIL sur Kaggle : **non recommandé**. Le relais `scripts/kaggle_run.py` n'a jamais tourné (`state.md`) et
  un matériel différent (T4, CUDA) s'ajouterait au levier.

## Conséquences
- Plus facile : tranche, pour ≈ 3,3 h et sans code dans `src/`, la question « l'optimisation bride-t-elle l'apprentissage
  quand le signal existe ? » ; protège la SIL et les interventions suivantes d'un réglage bridé.
- Plus difficile : un run hors spec de plus ; si SUCCÈS, la SIL (tournée à LR ×1) devra peut-être être rejouée sur la nouvelle
  base ; la LR modifie aussi la vitesse du pool de self-play (effet inclus dans le levier, assumé).
- Menaces : une graine par bras ; variance entre graines du témoin non mesurée ; levier choisi après avoir vu les données
  (limité par cet enregistrement avant le run) ; KL ∝ LR² est une approximation ; transfert dense → GCO non garanti (D4) ;
  le résultat ne dit rien d'une LR constante ni d'un ent_coef différent ; reprises XPU non bit à bit.
- Révision si : P1 contredit M1 (KL moyen de tranche ≥ 2,2·10⁻³, d'où le repli ×2) ; KL de la Phase A > 0,02 dès 0,1B ;
  publication d'un jalon public (pas jusqu'à la 1re touche) pour la recette RLGym.

## Résultat de la phase A (2026-11-22)
- Run `diag_dense_lr3_s0`, rapport `docs/reports/diag_dense_lr3_s0.md`. Éval D2 de l'instantané de l'itération 9 688 (commit
  86aa57a, `.venv` CPU) : « A touche » **29,10 % [28,48 ; 29,74]** (20 000 épisodes, SPRT H1). Témoin au même point : 23,04 %
  [22,46 ; 23,62]. KL moyenne de tranche maximale 2,10·10⁻³ ; aucun STOP.
- **Verdict D3 : NEUTRE**, appliqué par script (`runs/diag_dense_lr3_offline_eval/verdict_d3.json`). Gain ×1,26 sur la touche,
  sous le ×2 requis. Suite validée (point 5) : LR inchangée, audit clos, pas de phase B, retour à l'ordre ADR-009 D9 (ADR-007
  lancé le 2026-11-22).
- Diagnostic : le rapport des KL ×3 / base mesuré est de 1,4 à 2,1, et non ≈ 9. L'hypothèse KL ∝ LR² est réfutée pour ce run.

## Points validés par l'utilisateur (2026-11-21, « Je valide l'ADR-012 »)
1. Levier unique : LR ×3 (6e-4 → 1,5e-4, même forme), repli ×2 selon P1 ; ent_coef, epochs et clip écartés pour cet audit.
2. Phase A sur le témoin dense HORS SPEC, graine 0, arrêt à 0,5B ; succès si la borne basse du Wilson de « A touche » ≥ 46,0 %
   avec SPRT H1 ; nuisible si la borne haute < 22,4 %, KL de tranche > 0,02 ou STOP.
3. Phase B conditionnelle : GCO-std graine 0, 1B, critère ADR-006 D6, placée avant ADR-007 (amende ADR-009 D9).
4. Calendrier : après le run SIL ; pas d'exécution en parallèle sur Kaggle tant que le relais n'est pas validé.
5. Si NEUTRE : audit clos, aucun ADR ent_coef sans demande explicite.
