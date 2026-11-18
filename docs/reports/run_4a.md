# Rapport run 4a : GCO-std, trois graines, 1B timesteps appris

- Date : 2026-11-18. Rédigé par `experiment-analyst` (brouillon pour `research-director` et l'utilisateur). Tâche A7, ADR-005 §D7.
- Périmètre : analyse de fichiers existants (`runs/run_4a_std_s{0,1,2}/`, `checkpoints/run_4a_std_s*/eval_snapshots/`). Aucun
  entraînement, banc ni évaluation lancé pour ce rapport. Les scripts d'analyse sont jetables (hors dépôt).
- **Mise à jour 2026-11-18 (§4 bis)** : mesure hors ligne faite. SPRT +100 Elo contre l'aléatoire accepté sur l'instantané
  final des 3 graines ; (b) recalculée vraie sur les graines 0 et 1 ; l'agent ne touche la balle que dans 9 à 14 % des épisodes.
- **Verdict en une ligne** : aucune graine n'atteint S1-léger à 1B ; (a) n'est jamais remplie ; (b) est vraie à au moins une
  évaluation sur chaque graine ; GCO-S n'est pas lancé (ADR-005 D3). Le protocole ne tranche pas K1 dans ce cas (§6).
- Convention : « B » = milliard de timesteps appris ; `k` = index d'évaluation (tous les 0,05B, 969 itérations) ; les chiffres
  viennent de `metrics.jsonl` (m), `evals.jsonl` (e), `segments.jsonl` (g), `events.jsonl` (v), `run_meta*.json` (r) du run.

## 1. Provenance

| Graine | Segment | Commit | Début (UTC) | Reprise depuis | Fin du segment |
|---|---|---|---|---|---|
| 0 | 1 | 3e96158 | 2026-11-15 00:43:30 | | itération 2 283, `UR_RESULT_ERROR_DEVICE_LOST` (v, g), 1 947 s |
| 0 | 2 | 3e96158 | 2026-11-15 01:17:22 | ckpt 2 250 (33 lignes retirées) | 19 376, `completed`, 17 895 s ; **restauration K4** à 7 902 |
| 1 | 1 | 3e96158 | 2026-11-15 06:16:12 | | itération 7 572, `DEVICE_LOST`, 7 510 s |
| 1 | 2 | a76cd14 | 2026-11-15 22:44:29 | ckpt 7 500 (72 lignes retirées) | 19 376, `completed`, 12 058 s |
| 2 | 1 | a76cd14 | 2026-11-16 02:06:02 | | itération 3 282, `DEVICE_LOST`, 3 267 s |
| 2 | 2 | a76cd14 | 2026-11-16 03:01:43 | ckpt 3 250 (32 lignes retirées) | **arrêt demandé par l'utilisateur** à l'itération 14 493 (dernière ligne de `metrics_dropped.jsonl`, 06:09:35) ; ≈ 3,13 h |
| 2 | 3 | **e289c30** | 2026-11-17 23:53:32 | ckpt 14 250 (243 lignes retirées) | 19 376, `completed`, 5 060 s, fin 2026-11-18 01:17 |

- Le segment 2 de la graine 2 n'a **aucune** entrée dans `segments.jsonl` ni `events.jsonl` (processus tué) : sa durée vient des
  horodatages de `metrics.jsonl` et `metrics_dropped.jsonl`. La graine 2 a donc 3 `run_meta*` pour 2 lignes de `segments.jsonl`.
- Correction du cadrage de la tâche : la graine 2 a un 3e segment sur e289c30, pas seulement a76cd14. Vérifié :
  `git diff a76cd14 e289c30 --stat` ne touche que `production/state.md` ; `git diff 3e96158 a76cd14 --stat` touche
  `scripts/run_4a.py`, `tests/unit/test_run_4a_supervisor.py`, `docs/adr/005-run-4a.md`, le sprint et `state.md` (aucun fichier
  de `src/`).
- Configuration : les trois `run_meta.json` ont la même config aux seuls champs `name` et `seed` près (comparaison champ par champ).
  Hash de trajectoire / hash complet : s0 `8b7445a480e3` / `ceeb2cf586d8` ; s1 `14e9911e384d` / `3ac7d53769d7` ; s2 `8938f30a77e7` /
  `b65626ca17c8` (ils diffèrent par la graine). Hash de la récompense (`reward_config_hash`) identique : `d531d5919a4d`.
  Variante `GCO-std`, `max_episode_s` 120, `no_touch_timeout_s` 20, `p_kick` 0,2 → 0,8 sur 0,5B (`initial_states`), pool 80/20, PPO
  `gamma` 0,995, `lr` 2·10⁻⁴ → 5·10⁻⁵, `ent_coef` 0,01 → 0,002, 4 096 arènes, `tick_skip` 8 (r).
  Évaluation : `configs/run_4a_eval.yaml` (512 + 512 épisodes, `n_envs` 256, `R0` = 0,027 but/min par camp, P0 complet, ADR-005).
- Logiciel et matériel (r) : RocketGPU f0508a1, torch 2.15.0+xpu, Python 3.12.6, numpy 2.5.4, Windows 11 10.0.26200,
  Intel64 Family 6 Model 170 (Core Ultra 5 125H), Arc iGPU, 16 Go ; pilote 32.0.101.8860 (journal du sprint, pas dans `run_meta`).
- Durées d'exécution (somme des `wall_s` des segments, plus le segment non journalisé de la graine 2) : s0 5,51 h, s1 5,44 h,
  s2 5,44 h. Les trois ont exactement 1 000 014 736 timesteps appris (g).
- **Reprises** : chaque graine a eu 1 perte du GPU (à 117,8M, 390,8M et 169,4M timesteps appris), toutes dans le **premier**
  segment ; les ≈ 12,9 h cumulées de segments repris n'en ont montré aucune. Cause non établie (pas de test d'hypothèse ici).
  Les trois runs sont « repris » ; s0 est en plus « restauré » (une restauration K4, 8 716 288 timesteps écartés, `reseed`
  1 000 003). Reprise bit à bit sur XPU non prouvée (ADR-005 D8) : aucune reproductibilité bit à bit n'est prétendue.
- Évaluation de s0 : `k` = 8 figure deux fois dans `evals.jsonl` (21 lignes) : avant et après la restauration (7 751 est rejoué). Les
  résultats diffèrent (aléatoire 9/4 puis 8/4 victoires/défaites ; gelé 7/5 puis 8/6), ce qui donne une idée du bruit d'évaluation.
  Les tableaux de l'annexe gardent la ligne postérieure à la restauration.

## 2. Courbes par graine (D7 : optimisation, collecte, débit)

Tableaux par tranche de 0,1B en **annexe A** (moyennes de `metrics.jsonl`, 19 376 itérations par graine, sans trou). Lecture :
- **Entropie** (m `ppo/entropy`, maximum ln 90 = 4,4998) : tranche finale 4,357 (s0), 4,235 (s1), 4,374 (s2), soit 94 à 97 % du
  maximum ; minima 4,215 (it. 18 281), 4,020 (it. 17 470), 4,239 (it. 18 896). La décroissance s'accélère en fin de run sur s1
  (4,427, 4,414, 4,305, 4,235 par tranche de 0,1B à partir de 0,6B). Aucune itération sous 2,25 : l'alerte K2 n'est jamais
  émise (le seul événement `guard` de `events.jsonl` est la restauration de s0).
- **KL approx** : 1 à 2·10⁻⁴ en moyenne, maximum du run 4,8·10⁻³ (s0), 4,9·10⁻³ (s1), 4,6·10⁻³ (s2), contre un seuil K4 de 0,05.
  **Clip frac** < 0,1 % en moyenne par tranche. **Normes de gradient avant clip** moyennes 0,05 à 0,12, sous `max_grad_norm` 0,5 :
  les mises à jour sont petites et le clip global n'est pas actif en moyenne (maximum par itération non calculé).
- **EV** : −0,08 à la 1re itération de s0, ≈ 0,82 sur la 1re tranche, 0,96 de 0,4B à 1B ; minimum après l'itération 200 : 0,668 (s0),
  0,634 (s1), 0,657 (s2) ; aucune itération < 0 après l'itération 50 sur aucune graine.
- **`p_kick`** : 0,26 → 0,80 de 0 à 0,5B puis 0,80 constant (conforme D4). Les buts d'entraînement par itération (m `env/goals`)
  baissent de 7,4 à ≈ 3,8 quand `p_kick` monte (les états génériques donnent plus de buts : piège du smoke test, ADR-005), puis
  remontent à `p_kick` = 0,8 constant.
- **Buts à l'entraînement par origine : non journalisés** (seuls `env/goals`, `env/episode_ends`, `env/goals_total` existent ; D7
  l'exigeait). Le taux « buts / fins d'épisode » (annexe A) mélange kickoffs, états génériques et arènes contre le passé.
- **Débit appris** (annexe A, `train/trained_timesteps` ÷ temps entre lignes de `metrics.jsonl`, trous > 300 s exclus) : voir K6.
- **RAM : non journalisée** dans `metrics.jsonl` (aucune clé mémoire). Seul le banc M8 (2 h, hors de ces runs) l'a mesurée
  (journal du sprint). Pas de chiffre pour ces trois runs.
- **Restaurations** : s0 : 1 (it. 7 902 → 7 750, « valeurs non finies : policy_loss, value_loss, entropy, approx_kl, grad_norm,
  params, grads »), s1 : 0, s2 : 0 (m `train/restores_total`). Cause du NaN inconnue.

## 3. Évaluations (D7 : par évaluation)

Détail des 60 évaluations en **annexe B** (score contre l'aléatoire avec Wilson unilatéral sur les parties décisives, buts/min pour
et contre le gelé avec Garwood bilatéral 95 %). `R0` = 0,027 but/min par camp (P0 : 140 buts en 10 240 épisodes, `scripts/eval.py
--a random --b random --episodes 10240 --seed 0 --n-envs 256`, commit c83198e, ADR-005 « Mesure P0 complète »). Le gelé est
l'instantané de 0,1B plus tôt (θ₀ pour k = 1, 2).

| Graine | (a) remplie | (a) : décisives max sur 512 | Évaluations où (b) est vraie (itération) | (b) 3 de suite |
|---|---|---|---|---|
| 0 | jamais | 23 (k = 16, 13/10, verdict `failed`) | k = 14, 18, 19, 20 (13 564, 17 439, 18 407, 19 376) | **oui** (k = 18, 19, 20) |
| 1 | jamais | 17 (k = 17) | k = 12 (11 626) | non |
| 2 | jamais | 15 (k = 12) | k = 13, 16 (12 595, 15 501) | non |

(b) est vraie 7 fois sur 60 évaluations (e `condition_b`). Sous H0 (taux de but = `R0`), un Garwood bilatéral à 95 % dépasse `R0` dans
≈ 2,5 % des cas, soit ≈ 1,5 évaluation attendue sur 60 (calcul, indépendance supposée, ce qui est faux entre instantanés voisins) :
7 observées, dont 4 sur s0 en fin de run. Seuil pratique de (b) à ≈ 176 min de jeu : ≥ 10 buts marqués (10 buts → borne basse
0,0274 à k = 14 de s0 ; 9 buts → 0,0234 à k = 11).

**Contre l'aléatoire, (a) échoue d'abord par manque de puissance, mais le signal de jeu est faible.** Chiffres (e `random`) :
- Parties décisives : 2,1 %, 2,2 %, 2,1 % des 512 épisodes en moyenne sur les 20 évaluations (10,8 ; 11,0 ; 10,5 par évaluation ;
  s0/s1/s2). Aux évaluations k = 16 à 20 : 15,8 ; 13,6 ; 11,6 par évaluation. Une seule évaluation sur 60 atteint 20 décisives
  (s0 k = 16) et elle échoue (13/23, Wilson bas 0,397). Plus de 97 % de nuls partout.
- Durée moyenne d'un épisode d'évaluation (e `minutes` × 60 / épisodes) : 20,45 s (s0), 20,53 s (s1), 20,54 s (s2) à k = 16 à 20,
  contre 20,85 s pour aléatoire contre aléatoire (P0 : 3 557,7 min / 10 240). En entraînement, `env/episode_len_est` ≈ 305 à 309
  pas, soit ≈ 20,3 à 20,6 s à 15 décisions/s (120 Hz ÷ `tick_skip` 8, estimé). Ces durées sont proches de la troncature « 20 s sans
  touche » : presque tous les épisodes se terminent sans contact prolongé avec la balle. Le taux de touche n'est pas journalisé :
  lecture indirecte, estimée, non mesurée.
- Le signal existe : buts de l'agent contre l'aléatoire 0,057/min contre 0,021/min pour l'aléatoire sur k = 16 à 20 (149 contre 56
  buts, 2 625 min, trois graines), contre 0,019 et 0,027 sur k = 1 à 5 (51 contre 71). Part de victoires sur décisives k = 16 à 20 :
  0,709 (s0, 56/79), 0,750 (s1, 51/68), 0,724 (s2, 42/58) ; agrégé 149/205 = 0,727, Wilson bas unilatéral 0,673, p binomial
  unilatéral 3·10⁻¹¹ (**post hoc**, 15 évaluations d'instantanés voisins non indépendants : indicatif, pas un test de S1).
  Équivalent « Elo décisif » 400·log₁₀(V/D) ≈ +170 (+155, +191, +168 par graine) contre −57 sur k = 1 à 5.
- Puissance (calcul binomial exact) : avec 20 décisives, il faut ≥ 14 victoires pour que le Wilson bas dépasse 0,5 ; probabilité
  d'y arriver à 20 décisives : 0,61 si le vrai taux est 0,70, 0,42 à 0,65, 0,25 à 0,60. Probabilité d'obtenir ≥ 20 décisives sur 512
  épisodes (Poisson, moyenne k = 16 à 20) : 17 % (s0), 6 % (s1), 1,6 % (s2).
- Épisodes nécessaires au taux observé (k = 16 à 20) : 20 décisives en moyenne = 648 / 753 / 883 épisodes (s0 / s1 / s2) ; 60
  décisives (puissance 0,94 à taux vrai 0,70 ; 0,75 à 0,65) = 1 944 / 2 259 / 2 648 épisodes, soit 3,8 à 5,2 fois les 512 actuels.
  Temps : l'évaluation contre l'aléatoire de 512 épisodes dure 7,2 s en moyenne dans le run (e `timing.random_s`) ; ×4 à ×5 ≈ 30 à 40 s
  (estimé, non mesuré). Les 20 évaluations de s0/s1/s2 ont coûté 373 / 352 / 361 s au total (e `timing.total_s`), 1,8 à 1,9 % du temps.
- Contrefactuel sans le plancher de 20 décisives : (a) (Wilson bas > 0,5) serait vraie à 5 évaluations sur chaque graine, mais jamais
  avec (b) trois fois de suite ; le meilleur cas est s0, k = 19 et 20 (2 de suite). Le plancher n'est donc pas la seule cause de
  l'échec de S1-léger.
- Lecture de (b) : contre le gelé, buts pour et buts contre montent ensemble. k = 18 à 20, 3 graines (9 évaluations, 1 555 min) :
  77 buts pour, 96 contre, soit 0,111 but/min au total contre 0,043 sur k = 1 à 3 (69 buts, 1 598 min) et 0,039 pour deux politiques
  aléatoires (P0 : 140 buts / 3 557,7 min). (b) mesure que le taux de but du duel dépasse `R0`, pas que la politique courante domine
  le gelé : k = 16 à 20, buts pour/contre = 54/50 (s0), 38/51 (s1), 31/37 (s2). Aucune graine ne montre de supériorité sur son
  instantané de 0,1B plus tôt.

## 4. Elo ancré et SPRT : non fait

Ni l'Elo ancré tous les 0,1B (IC bootstrap) ni le SPRT à S1-seed et à 1B ne figurent dans les journaux (aucune occurrence de
« elo » ou « sprt » dans les `.jsonl` et `.json` des trois runs, vérifié par `grep -il`). S1-seed ne peut pas être calculé selon
D2 : il part de « l'instantané de la 1re évaluation de la série S1-léger », série inexistante. Le seul « Elo » du rapport est
l'équivalent décisif ci-dessus, tiré de `evals.jsonl`, non ancré et sans bootstrap.

Ce qu'il faudrait (estimé, non mesuré ; non lancé) : 10 instantanés par graine (k pairs), poids déjà conservés (`eval_snapshots`,
21 à 22 fichiers par graine, 16 Mo pour s0) :
`.venv/Scripts/python scripts/eval.py --a <instantané> --b random --episodes 2048 --n-envs 256 --seed 0 --out <fichier>.json`, soit
≈ 30 s par instantané (4 × 7,2 s), ≈ 5 min par graine. SPRT (`--sprt-elo 100 --sprt-max-decisive 1200`, plafond 20 000 épisodes) :
≈ 4,7 min au pire par instantané (39 × 7,2 s) ; à taux vrai 0,72 il s'arrêterait vers 34 décisives (≈ 1 400 épisodes, ≈ 20 s).
Indicatif : le rapport de vraisemblance du SPRT (H0 : p = 0,5, H1 : p = 0,64 soit +100 Elo) calculé sur les décomptes cumulés
k = 16 à 20 vaut 6,3 (s0), 7,0 (s1), 5,1 (s2) contre un seuil d'acceptation ln 19 = 2,94 ; non officiel (instantanés non indépendants,
décomptes agrégés), et D2 ancre S1-seed à la série S1-léger : décision à prendre (§9).

## 4 bis. Mesure hors ligne (2026-11-18, ajoutée après le rapport, à la demande de l'utilisateur)

Provenance : commit 6db79a4 (arbre propre, `git_dirty: false` dans chaque JSON), laptop Zenbook 14 (Core Ultra 5 125H), politiques
sur CPU, thermique non contrôlée, 2026-11-18 ≈ 18:10–18:45 UTC. Commande par instantané :
`scripts/eval.py --a <instantané> --b random --episodes 2048 --n-envs 256 --seed 0 --sprt-elo 100 --sprt-max-decisive 1200`
(20 000 épisodes pour l'instantané final). Sorties : `runs/run_4a_offline_eval/*.json` (hors git). Le comptage des touches
(`touches`, diagnostic D6, jamais dans la perte) est ajouté par 6db79a4 : « touche » = au moins un step de politique avec
contact balle dans l'épisode. Le SPRT est rejoué dans l'ordre des paires ; même graine 0 partout, donc les 2 048 premiers
épisodes de l'éval à 20 000 sont ceux de l'éval à 1,0B (même décision SPRT).

**Contre l'aléatoire, instantané final (1B, 20 000 épisodes)**

| Graine | V / N / D | Elo décisif [IC 95 % bootstrap] | SPRT +100 Elo | A touche la balle | l'aléatoire touche | aucune touche |
|---|---|---|---|---:|---:|---:|
| 0 | 479 / 19 372 / 149 | +203 [+172 ; +235] | accepte H1 (652 épisodes) | 13,9 % | 6,4 % | 81,0 % |
| 1 | 366 / 19 491 / 143 | +163 [+130 ; +199] | accepte H1 (654) | 10,3 % | 6,3 % | 84,2 % |
| 2 | 320 / 19 552 / 128 | +159 [+126 ; +196] | accepte H1 (1 416) | 8,7 % | 6,1 % | 86,1 % |

**Trajectoire (2 048 épisodes par point) : SPRT +100 Elo contre l'aléatoire**
- θ₀ (0B) : Elo décisif −39, +20, 0 ; plafond atteint sans décision ; aucune touche dans 88 à 89 % des épisodes.
- Graine 0 : H1 acceptée à 0,1B (isolé), puis de 0,5B à 1,0B à chaque point ; Elo 172 à 300.
- Graine 1 : H1 acceptée de 0,5B à 1,0B à chaque point ; Elo 178 à 275.
- Graine 2 : H1 acceptée à 0,3, 0,4, 0,6 et 1,0B ; plafond sans décision à 0,5, 0,7, 0,8, 0,9B ; Elo 114 à 196.
- Taux « A touche » : 6–7 % à θ₀, 9–14 % à 1B ; « aucune touche » : 88 % → 81–86 %. Durée moyenne 20,3 à 20,9 s (troncature
  « 20 s sans touche » dominante). 11 points par graine sans correction de multiplicité : la tendance compte, pas un point isolé.

**Condition (b) recalculée sur 2 048 épisodes** (instantané 1B contre gelé à 0,9B, it. 17 439 ; Garwood bilatéral 95 %,
`eval.rating.garwood_interval`, buts de A / minutes jouées ; `R0` = 0,027)

| Graine | Buts pour / contre | Minutes | Borne basse buts/min | > R0 ? | Elo décisif contre le gelé |
|---|---|---:|---:|---|---|
| 0 | 50 / 46 | 690,2 | 0,0538 | oui | +14 [−58 ; +89] |
| 1 | 50 / 41 | 686,3 | 0,0541 | oui | +34 [−34 ; +115] |
| 2 | 27 / 15 | 697,1 | 0,0255 | non (de peu) | +102 [−8 ; +228] |

**Lecture**
- Hors ligne, les deux volets de S1-seed (D2 : SPRT +100 Elo accepté ; (b) sur ≈ 2 000 épisodes) sont remplis par l'instantané
  final des graines 0 et 1, pas par la graine 2 ((b) manquée de 0,0015 but/min). Ce n'est **pas** S1-seed au sens strict de D2,
  qui ancre la mesure à la 1re évaluation d'une série S1-léger, inexistante ici. Compter cette mesure comme S1-seed est un
  changement de critère **après avoir vu les données** : il exige un amendement d'ADR-005 et doit être signalé comme tel.
- L'échec de S1-léger (a) tient au plancher de 20 parties décisives sur 512 : à 20 000 épisodes, l'avantage contre l'aléatoire
  est net (IC de l'Elo décisif entièrement au-dessus de +100 pour les 3 graines).
- Le niveau de jeu reste très faible en absolu : l'agent n'atteint la balle que dans 9 à 14 % des épisodes kickoff
  (6 % au départ), et 81 à 86 % des épisodes finissent sans aucune touche. Le goulot « premier contact » du §8 est confirmé
  par mesure directe. « Battre l'aléatoire » ici veut dire : quand quelqu'un touche la balle, c'est plus souvent l'agent, et
  il convertit mieux.

## 5. Par variante : GCO-std, médiane [min ; max] sur 3 graines (GCO-S non lancé)

Aucune moyenne ± écart type (3 graines). Sources : m, e, g, calculs de ce rapport.

| Mesure | s0 | s1 | s2 | Médiane [min ; max] |
|---|---|---|---|---|
| Steps / heures jusqu'à S1-seed | non atteint à 1B | non atteint | non atteint | non atteint (3/3) |
| Évaluations avec (b) vraie (sur 20) | 4 | 1 | 2 | 2 [1 ; 4] |
| Décisives max sur 512 contre l'aléatoire | 23 | 17 | 15 | 17 [15 ; 23] |
| Part de victoires décisives, k = 16 à 20 | 0,709 | 0,750 | 0,724 | 0,724 [0,709 ; 0,750] |
| Entropie, dernière tranche de 0,1B | 4,357 | 4,235 | 4,374 | 4,357 [4,235 ; 4,374] |
| Entropie minimale du run | 4,215 | 4,020 | 4,239 | 4,215 [4,020 ; 4,239] |
| EV, dernière tranche | 0,959 | 0,958 | 0,965 | 0,959 [0,958 ; 0,965] |
| Buts / fins d'épisode, dernière tranche | 5,32 % | 4,80 % | 4,60 % | 4,80 % [4,60 ; 5,32] |
| Débit appris soutenu (pas/s) | 50,2k | 51,3k | 51,4k | 51,3k [50,2k ; 51,4k] |
| Restaurations | 1 | 0 | 0 | 0 [0 ; 1] |
| Durée d'exécution (h) | 5,51 | 5,44 | 5,44 | 5,44 [5,44 ; 5,51] |

H2 (GCO-S contre GCO-std) : aucune paire disponible.

## 6. Kill criteria (D6)

| # | Statut | Chiffres |
|---|---|---|
| K1 | **ambigu, voir ci-dessous** | (b) vraie sur 3 graines sur 3 à au moins une évaluation ; 3 de suite sur s0 seulement ; S1-léger ((a) et (b)) sur 0 graine |
| K2 | non déclenché | entropie minimale 4,020 (89 % de ln 90) ; seuil 0,450 ; alerte 2,25 jamais franchie |
| K3 | non armé | `s1_reached` jamais vrai (m `train/s1_reached` = 0) ; à titre descriptif, aucune itération EV < 0 après l'itération 50 |
| K4 | déclenché une fois, s0 | valeurs non finies à l'itération 7 902, restauration automatique ; KL maximale 4,9·10⁻³ < 0,05 ; 0 sur s1 et s2 |
| K5 | non déclenché | 1 restauration cumulée sur s0 (seuil > 3), 1 consécutive (seuil > 2) |
| K6 | non déclenché | débit appris soutenu 50,2k / 51,3k / 51,4k pas/s (s0/s1/s2, ÷ temps entre lignes de `metrics.jsonl`, trous exclus), pire fenêtre glissante de 30 min : 44,9k (it. 7 721 à 9 286, contient la restauration de s0) / 48,9k / 46,6k ; seuil 25k. Débit collecté moyen des segments : 58,4k à 61,1k (g `mean_steps_per_s`) |
| K7 | non armé | `s1_reached` jamais vrai ; `k7_bad` faux à toutes les évaluations des trois runs |

**K1, les deux lectures (à trancher par l'utilisateur et le `research-director`, non tranchées ici).**
- Lecture stricte (texte D6 : « aucune seed ne remplit S1-léger (b) ») : (b) est remplie sur 3 graines (au moins une évaluation chacune ;
  avec la règle des 3 évaluations de suite, sur s0 seulement). K1 n'est **pas** déclenché. Le critère de `docs/PLAN.md` (« aucun but
  contre l'adversaire gelé après 1B sur 3 seeds ») ne l'est pas non plus : de 5 à 31 buts (pour plus contre) par évaluation contre le
  gelé, jamais zéro. Mais H1 n'est pas atteinte non plus, et la condition de GO de D7 (S1-seed sur ≥ 1 graine) n'est pas remplie.
- Lecture par l'intention (S1-léger complet = (a) et (b) sur 3 évaluations de suite) : 0 graine sur 3, K1 serait déclenché : fin de la
  ligne std, intervention R&D (D6, D7).
- Cas intermédiaire non prévu par le protocole : (b) sans (a). Le désaccord vient de ce que (a) ne peut presque pas être remplie
  à 512 épisodes (§3) alors que (b) l'est parfois. Conséquence pour GCO-S : D3 le conditionne à « ≥ 1 graine std a S1-léger » :
  non rempli sous les deux lectures, donc GCO-S non lancé.

## 7. Hypothèses (D1) et menaces (D8)

- **H1** (S1-seed sur ≥ 1 graine sur 3 avant 1B) : **non atteinte**. Au sens littéral de la colonne « Réfutée si » (0 graine sur 3),
  elle est réfutée ; mais S1-seed n'a jamais été mesuré (pas de série S1-léger ni de SPRT hors ligne), donc la réfutation porte sur le
  protocole tel qu'exécuté. Indicatif seulement : les décomptes agrégés contre l'aléatoire (§3, §4) sont compatibles avec un
  avantage décisif ≈ +170 Elo sur k = 16 à 20 sur les 3 graines.
- **H2** : non testable (GCO-S non lancé, 0 paire). Pas d'escalade « H2 réfutée » (D7).
- **H3** (EV ≥ 0 et score sans régression après S1) : sans objet, car conditionnelle à S1 ; K3 et K7 non armés. Descriptivement,
  EV ≥ 0,63 après l'itération 200 et pas de régression visible du score contre l'aléatoire (part de victoires décisives de 0,42 à 0,73).
- **Menaces D8 applicables** : (i) 3 graines et variance entre graines : s0 a (b) 4 fois, s1 1 fois, s2 2 fois, sans test possible ;
  (ii) débit : mesuré 50,2k à 51,4k appris/s, durée 5,4 à 5,5 h par run, sans écart au plateau M8 (51 à 54k) autre qu'en bas de
  fourchette ; (iii) self-play non stationnaire : le gelé est une version à 0,1B qui diffère peu (KL de 10⁻⁴ par itération), d'où
  (b) peu informative sur la progression ; (iv) `R0` : mesuré (0,027), 1 passage, 1 graine ; contraste depuis états génériques
  non fait ; (v) écart sim/jeu non mesuré ; (vi) reprise bit à bit non prouvée : les 3 runs sont repris, s0 restauré ;
  (vii) évaluations : `n_envs` 256 fixe (précision 3), échantillon de kickoffs fixé par run, bruit d'évaluation visible sur s0 k = 8.
  Menace supplémentaire : 60 évaluations avec (b) à 95 % bilatéral, sans correction de multiplicité (§3).

## 8. Diagnostic sur les journaux existants (D6)

- **L'agent progresse, lentement.** Contre l'aléatoire : part de victoires décisives 0,42 (k = 1 à 5, agrégé) → 0,73 (k = 16 à 20) ;
  taux de but du duel contre le gelé ×2,6 (0,043 → 0,111 but/min). À `p_kick` constant 0,8 (tranches de 2 422 itérations à partir
  de l'itération 9 688), buts / fins d'épisode à l'entraînement : s0 4,15 → 4,50 → 4,96 → 5,31 % ; s1 4,24 → 4,53 → 4,76 → 4,82 % ;
  s2 4,14 → 4,30 → 4,30 → 4,54 %. Ce taux mélange les origines (non journalisées) et les adversaires, et n'est pas une mesure de S1
  (D4) ; la hausse est cohérente avec les évaluations mais n'est pas séparable des états génériques.
- **La politique reste presque uniforme.** Entropie 94 à 97 % de ln 90 en fin de run, KL ≈ 10⁻⁴, clip < 0,1 %, gradients avant clip
  < 0,12 en moyenne : peu de mouvement par itération. `lr` et `ent_coef` atteignent leurs planchers (5·10⁻⁵ et 0,002) en fin de run ;
  rien n'indique un effondrement (K2 loin), plutôt un apprentissage lent.
- **Les épisodes sont presque tous des non-contacts** (durée ≈ 20,5 s ≈ troncature « 20 s sans touche », §3) : le goulot probable est
  le premier contact puis le tir, pas la stabilité numérique (indirect, taux de touche non journalisé ; le protocole permet de le
  mesurer « en diagnostic seulement », jamais dans la perte).
- **Critique** : EV 0,96 dès 0,4B, perte de valeur ≈ 1,3·10⁻⁴ ; pas de goulot visible côté critique. L'EV élevée est probablement
  portée par les épisodes depuis états génériques (hypothèse, non mesurée).
- **Données manquantes pour aller plus loin** : buts par origine, taux de touche, Elo ancré, SPRT, RAM, cause du `DEVICE_LOST` et
  du NaN.
- **Interventions candidates, dans l'ordre de `docs/GCO_SPEC.md` §8 (aucune choisie ; une seule à la fois, un ADR chacune,
  GCO-std graine 0 d'abord, comparée à la graine 0 de ce run)** : (1) calendrier d'états génériques (annealing plus lent) ;
  (2) γ 0,995 → 0,998 ; (3) self-imitation ; (4) tâche auxiliaire « prochain but » ; (5) composition critic ; (6) world model.
  Hors §8 et hors entraînement : puissance de l'évaluation (plus d'épisodes ou plancher de décisives adapté), question de protocole.

## 9. Décision de suite : options et coûts (estimé, non mesuré, sauf mention ; aucune recommandation)

Base de coût mesurée : un run de 1B = 5,44 à 5,51 h de calcul (§1), ≈ 5,5 h par graine.
1. **Mesure hors ligne sur les instantanés existants** (Elo ancré, SPRT +100 Elo, éventuellement taux de touche) : ≈ 5 à 10 min par
   graine, sans entraînement (§4). Ne change pas le statut de S1-léger du protocole ; informe H1 et le choix entre les lectures de K1.
   Exige de décider si un SPRT sur un instantané final compte pour S1-seed (écart à D2 : à consigner par ADR).
2. **Lire K1 comme déclenché et lancer l'intervention 1 de §8 (annealing plus lent)** : ADR, puis 1 run GCO-std graine 0 ≈ 5,5 h ;
   3 graines ≈ 16,5 h.
3. **Lire K1 comme non déclenché et prolonger** les runs au-delà de 1B (checkpoint final `ckpt_000078_it00019376.pt` présent pour
   les 3 graines) : ADR de budget (le calendrier `lr` / entropie est terminé à 1B, ce qui change la trajectoire) ; ≈ 5,5 h par 1B et
   par graine.
4. **Lancer GCO-S graine 0 malgré D3** (≈ 5,5 h) : exige un amendement de l'ADR-005 ; n'informe pas H1, H2 reste descriptive sur
   une paire.
5. **Renforcer l'évaluation pour les prochains runs** (≥ 2 000 épisodes contre l'aléatoire) : ≈ 4 à 5 fois le coût actuel de 1,9 %,
   soit ≈ 8 % d'un run ; précision 3 de l'ADR-005 à réviser.
6. Le gate G3 n'est pas franchissable : la condition de GO de D7 (S1-seed sur ≥ 1 graine) n'est pas remplie.
Ces options ne sont pas exclusives (1 est peu coûteuse et précède utilement 2 à 4).

## Annexe A. Tableaux par graine : optimisation, collecte, débit

Source : `runs/run_4a_std_s{0,1,2}/metrics.jsonl`, moyennes par tranche de `train/trained_timesteps` (0,1B ≈ 1 937 itérations ;
dernière tranche jusqu'à 1,0B inclus). Débit : 51 611 × itérations ÷ somme des temps entre lignes (< 300 s), évaluations incluses.

#### Graine 0 : optimisation (moyenne par tranche de 0,1B appris)

| Tranche (B) | Entropie | KL approx (×10⁻⁴) | Clip frac (×10⁻⁴) | EV | Perte π (×10⁻⁴) | Perte V (×10⁻⁴) | ‖g‖ avant clip | ‖g_π‖ après clip | ‖g_V‖ après clip |
|---|---|---|---|---|---|---|---|---|---|
| 0.0–0.1 | 4.457 | 6.57 | 10.3 | 0.821 | -2.15 | 3.94 | 0.082 | 0.045 | 0.0267 |
| 0.1–0.2 | 4.479 | 2.32 | 1.1 | 0.871 | -1.45 | 2.81 | 0.054 | 0.034 | 0.0121 |
| 0.2–0.3 | 4.485 | 1.28 | 0.3 | 0.942 | -1.04 | 1.89 | 0.046 | 0.029 | 0.0100 |
| 0.3–0.4 | 4.476 | 1.49 | 0.8 | 0.960 | -1.05 | 1.45 | 0.050 | 0.032 | 0.0080 |
| 0.4–0.5 | 4.455 | 2.34 | 5.6 | 0.961 | -1.14 | 1.29 | 0.057 | 0.038 | 0.0061 |
| 0.5–0.6 | 4.449 | 2.28 | 6.8 | 0.962 | -1.17 | 1.19 | 0.063 | 0.042 | 0.0048 |
| 0.6–0.7 | 4.434 | 2.29 | 7.9 | 0.962 | -1.15 | 1.28 | 0.070 | 0.046 | 0.0042 |
| 0.7–0.8 | 4.408 | 2.10 | 5.9 | 0.962 | -1.03 | 1.39 | 0.076 | 0.051 | 0.0038 |
| 0.8–0.9 | 4.388 | 1.78 | 4.1 | 0.960 | -0.91 | 1.53 | 0.083 | 0.055 | 0.0033 |
| 0.9–1.0 | 4.357 | 1.39 | 2.8 | 0.959 | -0.81 | 1.62 | 0.092 | 0.062 | 0.0032 |

#### Graine 0 : collecte et débit

| Tranche (B) | `p_kick` | Buts / itération | Buts / fins d'épisode | Durée d'épisode (pas) | Débit appris (pas/s) | Itérations |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 0.26 | 7.38 | 7.77 % | 456 | 61.4k | 1937 |
| 0.1–0.2 | 0.38 | 6.49 | 6.81 % | 304 | 48.4k | 1938 |
| 0.2–0.3 | 0.50 | 5.43 | 5.75 % | 307 | 50.3k | 1937 |
| 0.3–0.4 | 0.62 | 4.74 | 5.05 % | 309 | 50.4k | 1938 |
| 0.4–0.5 | 0.74 | 4.20 | 4.48 % | 309 | 46.1k | 1937 |
| 0.5–0.6 | 0.80 | 3.83 | 4.09 % | 309 | 49.0k | 1938 |
| 0.6–0.7 | 0.80 | 4.14 | 4.41 % | 309 | 50.0k | 1938 |
| 0.7–0.8 | 0.80 | 4.49 | 4.75 % | 306 | 49.8k | 1937 |
| 0.8–0.9 | 0.80 | 4.83 | 5.09 % | 306 | 49.6k | 1938 |
| 0.9–1.0 | 0.80 | 5.04 | 5.32 % | 306 | 49.5k | 1938 |

#### Graine 1 : optimisation (moyenne par tranche de 0,1B appris)

| Tranche (B) | Entropie | KL approx (×10⁻⁴) | Clip frac (×10⁻⁴) | EV | Perte π (×10⁻⁴) | Perte V (×10⁻⁴) | ‖g‖ avant clip | ‖g_π‖ après clip | ‖g_V‖ après clip |
|---|---|---|---|---|---|---|---|---|---|
| 0.0–0.1 | 4.473 | 4.09 | 2.5 | 0.824 | -1.76 | 3.78 | 0.076 | 0.038 | 0.0298 |
| 0.1–0.2 | 4.478 | 2.49 | 1.6 | 0.869 | -1.49 | 2.81 | 0.055 | 0.034 | 0.0118 |
| 0.2–0.3 | 4.475 | 2.18 | 1.6 | 0.944 | -1.40 | 1.85 | 0.057 | 0.037 | 0.0109 |
| 0.3–0.4 | 4.465 | 2.41 | 1.7 | 0.958 | -1.31 | 1.56 | 0.059 | 0.039 | 0.0084 |
| 0.4–0.5 | 4.456 | 2.20 | 2.8 | 0.959 | -1.22 | 1.32 | 0.058 | 0.039 | 0.0066 |
| 0.5–0.6 | 4.440 | 2.51 | 3.6 | 0.959 | -1.30 | 1.25 | 0.066 | 0.046 | 0.0053 |
| 0.6–0.7 | 4.427 | 2.22 | 4.1 | 0.961 | -1.21 | 1.32 | 0.071 | 0.048 | 0.0046 |
| 0.7–0.8 | 4.414 | 1.84 | 4.1 | 0.962 | -1.11 | 1.34 | 0.078 | 0.054 | 0.0042 |
| 0.8–0.9 | 4.305 | 2.31 | 4.5 | 0.958 | -1.25 | 1.48 | 0.103 | 0.073 | 0.0036 |
| 0.9–1.0 | 4.235 | 2.01 | 5.3 | 0.958 | -1.08 | 1.48 | 0.121 | 0.086 | 0.0033 |

#### Graine 1 : collecte et débit

| Tranche (B) | `p_kick` | Buts / itération | Buts / fins d'épisode | Durée d'épisode (pas) | Débit appris (pas/s) | Itérations |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 0.26 | 7.30 | 7.70 % | 449 | 52.2k | 1937 |
| 0.1–0.2 | 0.38 | 6.41 | 6.74 % | 304 | 52.1k | 1938 |
| 0.2–0.3 | 0.50 | 5.55 | 5.87 % | 306 | 52.0k | 1937 |
| 0.3–0.4 | 0.62 | 4.85 | 5.17 % | 309 | 52.8k | 1938 |
| 0.4–0.5 | 0.74 | 4.15 | 4.42 % | 309 | 54.0k | 1937 |
| 0.5–0.6 | 0.80 | 3.93 | 4.20 % | 310 | 50.2k | 1938 |
| 0.6–0.7 | 0.80 | 4.23 | 4.50 % | 309 | 50.2k | 1938 |
| 0.7–0.8 | 0.80 | 4.34 | 4.61 % | 307 | 50.4k | 1937 |
| 0.8–0.9 | 0.80 | 4.59 | 4.84 % | 306 | 50.0k | 1938 |
| 0.9–1.0 | 0.80 | 4.57 | 4.80 % | 305 | 49.8k | 1938 |

#### Graine 2 : optimisation (moyenne par tranche de 0,1B appris)

| Tranche (B) | Entropie | KL approx (×10⁻⁴) | Clip frac (×10⁻⁴) | EV | Perte π (×10⁻⁴) | Perte V (×10⁻⁴) | ‖g‖ avant clip | ‖g_π‖ après clip | ‖g_V‖ après clip |
|---|---|---|---|---|---|---|---|---|---|
| 0.0–0.1 | 4.454 | 7.21 | 11.3 | 0.821 | -2.22 | 3.95 | 0.081 | 0.045 | 0.0258 |
| 0.1–0.2 | 4.478 | 1.97 | 1.0 | 0.849 | -1.34 | 2.98 | 0.050 | 0.031 | 0.0102 |
| 0.2–0.3 | 4.477 | 1.68 | 0.8 | 0.910 | -1.22 | 2.18 | 0.050 | 0.031 | 0.0094 |
| 0.3–0.4 | 4.465 | 1.94 | 1.4 | 0.950 | -1.32 | 1.56 | 0.052 | 0.034 | 0.0083 |
| 0.4–0.5 | 4.453 | 2.48 | 4.4 | 0.962 | -1.33 | 1.27 | 0.054 | 0.036 | 0.0065 |
| 0.5–0.6 | 4.440 | 2.47 | 5.6 | 0.965 | -1.39 | 1.17 | 0.060 | 0.041 | 0.0054 |
| 0.6–0.7 | 4.427 | 2.23 | 4.6 | 0.964 | -1.27 | 1.20 | 0.066 | 0.045 | 0.0046 |
| 0.7–0.8 | 4.418 | 1.88 | 4.5 | 0.965 | -1.13 | 1.20 | 0.072 | 0.049 | 0.0039 |
| 0.8–0.9 | 4.405 | 1.42 | 2.6 | 0.965 | -0.99 | 1.20 | 0.078 | 0.054 | 0.0033 |
| 0.9–1.0 | 4.374 | 1.19 | 2.2 | 0.964 | -0.79 | 1.31 | 0.083 | 0.059 | 0.0027 |

#### Graine 2 : collecte et débit

| Tranche (B) | `p_kick` | Buts / itération | Buts / fins d'épisode | Durée d'épisode (pas) | Débit appris (pas/s) | Itérations |
|---|---|---|---|---|---|---|
| 0.0–0.1 | 0.26 | 7.35 | 7.75 % | 460 | 52.2k | 1937 |
| 0.1–0.2 | 0.38 | 6.51 | 6.84 % | 304 | 49.4k | 1938 |
| 0.2–0.3 | 0.50 | 5.48 | 5.80 % | 307 | 49.9k | 1937 |
| 0.3–0.4 | 0.62 | 4.79 | 5.09 % | 308 | 51.7k | 1938 |
| 0.4–0.5 | 0.74 | 4.18 | 4.45 % | 308 | 51.6k | 1937 |
| 0.5–0.6 | 0.80 | 3.90 | 4.16 % | 308 | 51.5k | 1938 |
| 0.6–0.7 | 0.80 | 4.00 | 4.25 % | 308 | 51.6k | 1938 |
| 0.7–0.8 | 0.80 | 4.04 | 4.28 % | 307 | 55.8k | 1937 |
| 0.8–0.9 | 0.80 | 4.09 | 4.32 % | 306 | 53.0k | 1938 |
| 0.9–1.0 | 0.80 | 4.35 | 4.60 % | 306 | 48.3k | 1938 |


## Annexe B. Évaluations par graine

Source : `runs/run_4a_std_s{0,1,2}/evals.jsonl`. V/D/N = victoires/défaites/nuls contre l'aléatoire sur 512 épisodes ; Wilson unilatéral 95 % sur les décisives (verdict `indeterminate` sous 20 décisives) ; Garwood bilatéral 95 % ; `R0` = 0,027 but/min par camp ; (b) = borne basse des buts pour > `R0`.

#### Graine 0 : évaluations

| k | Itér. | B | Aléatoire V/D/N (déc.) | Score (nul = ½) | Wilson 1 côté [bas ; haut] sur décisives | Verdict | Gelé : buts pour (n, /min [Garwood 95 %]) | Gelé : buts contre | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 969 | 0.05 | 3/4/505 (7) | 0.499 | [0.19 ; 0.71] | indeterminate | 5 (0.028 [0.009 ; 0.066]) | 4 (0.023 [0.006 ; 0.058]) | non | non |
| 2 | 1938 | 0.10 | 0/4/508 (4) | 0.496 | [0.00 ; 0.40] | indeterminate | 1 (0.006 [0.000 ; 0.031]) | 5 (0.028 [0.009 ; 0.065]) | non | non |
| 3 | 2907 | 0.15 | 2/5/505 (7) | 0.497 | [0.10 ; 0.59] | indeterminate | 4 (0.023 [0.006 ; 0.058]) | 1 (0.006 [0.000 ; 0.031]) | non | non |
| 4 | 3876 | 0.20 | 2/1/509 (3) | 0.501 | [0.25 ; 0.92] | indeterminate | 7 (0.039 [0.016 ; 0.081]) | 1 (0.006 [0.000 ; 0.031]) | non | non |
| 5 | 4844 | 0.25 | 5/2/505 (7) | 0.503 | [0.41 ; 0.90] | indeterminate | 5 (0.028 [0.009 ; 0.066]) | 11 (0.062 [0.031 ; 0.112]) | non | non |
| 6 | 5813 | 0.30 | 4/7/501 (11) | 0.497 | [0.18 ; 0.61] | indeterminate | 5 (0.029 [0.009 ; 0.067]) | 5 (0.029 [0.009 ; 0.067]) | non | non |
| 7 | 6782 | 0.35 | 8/5/499 (13) | 0.503 | [0.39 ; 0.80] | indeterminate | 9 (0.051 [0.023 ; 0.097]) | 5 (0.028 [0.009 ; 0.066]) | non | non |
| 8 | 7751 | 0.40 | 8/4/500 (12) | 0.504 | [0.43 ; 0.84] | indeterminate | 8 (0.045 [0.020 ; 0.089]) | 6 (0.034 [0.012 ; 0.074]) | non | non |
| 9 | 8720 | 0.45 | 6/3/503 (9) | 0.503 | [0.40 ; 0.86] | indeterminate | 6 (0.034 [0.012 ; 0.074]) | 3 (0.017 [0.004 ; 0.050]) | non | non |
| 10 | 9688 | 0.50 | 4/2/506 (6) | 0.502 | [0.35 ; 0.88] | indeterminate | 9 (0.050 [0.023 ; 0.096]) | 6 (0.034 [0.012 ; 0.073]) | non | non |
| 11 | 10657 | 0.55 | 12/6/494 (18) | 0.506 | [0.47 ; 0.82] | indeterminate | 9 (0.051 [0.023 ; 0.097]) | 8 (0.046 [0.020 ; 0.090]) | non | non |
| 12 | 11626 | 0.60 | 7/1/504 (8) | 0.506 | [0.59 ; 0.97] | indeterminate | 7 (0.039 [0.016 ; 0.081]) | 2 (0.011 [0.001 ; 0.041]) | non | non |
| 13 | 12595 | 0.65 | 7/2/503 (9) | 0.505 | [0.50 ; 0.92] | indeterminate | 4 (0.023 [0.006 ; 0.059]) | 6 (0.034 [0.013 ; 0.075]) | non | non |
| 14 | 13564 | 0.70 | 6/3/503 (9) | 0.503 | [0.40 ; 0.86] | indeterminate | 10 (0.057 [0.027 ; 0.105]) | 8 (0.046 [0.020 ; 0.090]) | non | **oui** |
| 15 | 14532 | 0.75 | 9/5/498 (14) | 0.504 | [0.43 ; 0.81] | indeterminate | 4 (0.023 [0.006 ; 0.059]) | 9 (0.052 [0.024 ; 0.098]) | non | non |
| 16 | 15501 | 0.80 | 13/10/489 (23) | 0.503 | [0.40 ; 0.72] | failed | 4 (0.023 [0.006 ; 0.058]) | 6 (0.034 [0.012 ; 0.074]) | non | non |
| 17 | 16470 | 0.85 | 10/1/501 (11) | 0.509 | [0.68 ; 0.98] | indeterminate | 9 (0.052 [0.024 ; 0.098]) | 4 (0.023 [0.006 ; 0.059]) | non | non |
| 18 | 17439 | 0.90 | 7/3/502 (10) | 0.504 | [0.44 ; 0.87] | indeterminate | 11 (0.063 [0.032 ; 0.113]) | 9 (0.052 [0.024 ; 0.098]) | non | **oui** |
| 19 | 18407 | 0.95 | 13/4/495 (17) | 0.509 | [0.57 ; 0.89] | indeterminate | 16 (0.094 [0.054 ; 0.152]) | 14 (0.082 [0.045 ; 0.138]) | non | **oui** |
| 20 | 19376 | 1.00 | 13/5/494 (18) | 0.508 | [0.53 ; 0.86] | indeterminate | 14 (0.082 [0.045 ; 0.137]) | 17 (0.099 [0.058 ; 0.159]) | non | **oui** |

#### Graine 1 : évaluations

| k | Itér. | B | Aléatoire V/D/N (déc.) | Score (nul = ½) | Wilson 1 côté [bas ; haut] sur décisives | Verdict | Gelé : buts pour (n, /min [Garwood 95 %]) | Gelé : buts contre | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 969 | 0.05 | 4/5/503 (9) | 0.499 | [0.22 ; 0.70] | indeterminate | 4 (0.022 [0.006 ; 0.058]) | 1 (0.006 [0.000 ; 0.031]) | non | non |
| 2 | 1938 | 0.10 | 2/5/505 (7) | 0.497 | [0.10 ; 0.59] | indeterminate | 5 (0.028 [0.009 ; 0.065]) | 4 (0.022 [0.006 ; 0.057]) | non | non |
| 3 | 2907 | 0.15 | 4/5/503 (9) | 0.499 | [0.22 ; 0.70] | indeterminate | 5 (0.028 [0.009 ; 0.065]) | 2 (0.011 [0.001 ; 0.040]) | non | non |
| 4 | 3876 | 0.20 | 6/8/498 (14) | 0.498 | [0.24 ; 0.64] | indeterminate | 3 (0.017 [0.003 ; 0.048]) | 2 (0.011 [0.001 ; 0.040]) | non | non |
| 5 | 4844 | 0.25 | 4/5/503 (9) | 0.499 | [0.22 ; 0.70] | indeterminate | 4 (0.023 [0.006 ; 0.058]) | 4 (0.023 [0.006 ; 0.058]) | non | non |
| 6 | 5813 | 0.30 | 4/3/505 (7) | 0.501 | [0.29 ; 0.81] | indeterminate | 5 (0.028 [0.009 ; 0.065]) | 6 (0.034 [0.012 ; 0.073]) | non | non |
| 7 | 6782 | 0.35 | 3/7/502 (10) | 0.496 | [0.13 ; 0.56] | indeterminate | 6 (0.034 [0.012 ; 0.073]) | 5 (0.028 [0.009 ; 0.065]) | non | non |
| 8 | 7751 | 0.40 | 7/7/498 (14) | 0.500 | [0.30 ; 0.70] | indeterminate | 7 (0.040 [0.016 ; 0.081]) | 3 (0.017 [0.003 ; 0.050]) | non | non |
| 9 | 8720 | 0.45 | 7/2/503 (9) | 0.505 | [0.50 ; 0.92] | indeterminate | 9 (0.051 [0.023 ; 0.097]) | 12 (0.068 [0.035 ; 0.119]) | non | non |
| 10 | 9688 | 0.50 | 8/4/500 (12) | 0.504 | [0.43 ; 0.84] | indeterminate | 7 (0.040 [0.016 ; 0.082]) | 4 (0.023 [0.006 ; 0.058]) | non | non |
| 11 | 10657 | 0.55 | 8/5/499 (13) | 0.503 | [0.39 ; 0.80] | indeterminate | 7 (0.040 [0.016 ; 0.083]) | 4 (0.023 [0.006 ; 0.059]) | non | non |
| 12 | 11626 | 0.60 | 9/3/500 (12) | 0.506 | [0.51 ; 0.90] | indeterminate | 12 (0.067 [0.035 ; 0.118]) | 5 (0.028 [0.009 ; 0.066]) | non | **oui** |
| 13 | 12595 | 0.65 | 5/3/504 (8) | 0.502 | [0.35 ; 0.84] | indeterminate | 7 (0.040 [0.016 ; 0.082]) | 10 (0.057 [0.027 ; 0.105]) | non | non |
| 14 | 13564 | 0.70 | 6/5/501 (11) | 0.501 | [0.32 ; 0.76] | indeterminate | 5 (0.029 [0.009 ; 0.067]) | 6 (0.035 [0.013 ; 0.075]) | non | non |
| 15 | 14532 | 0.75 | 5/3/504 (8) | 0.502 | [0.35 ; 0.84] | indeterminate | 9 (0.052 [0.024 ; 0.098]) | 8 (0.046 [0.020 ; 0.090]) | non | non |
| 16 | 15501 | 0.80 | 11/3/498 (14) | 0.508 | [0.57 ; 0.91] | indeterminate | 8 (0.046 [0.020 ; 0.092]) | 10 (0.058 [0.028 ; 0.107]) | non | non |
| 17 | 16470 | 0.85 | 12/5/495 (17) | 0.507 | [0.51 ; 0.85] | indeterminate | 9 (0.052 [0.024 ; 0.100]) | 11 (0.064 [0.032 ; 0.115]) | non | non |
| 18 | 17439 | 0.90 | 14/2/496 (16) | 0.512 | [0.68 ; 0.96] | indeterminate | 8 (0.046 [0.020 ; 0.091]) | 10 (0.058 [0.028 ; 0.107]) | non | non |
| 19 | 18407 | 0.95 | 8/4/500 (12) | 0.504 | [0.43 ; 0.84] | indeterminate | 6 (0.035 [0.013 ; 0.076]) | 13 (0.076 [0.040 ; 0.130]) | non | non |
| 20 | 19376 | 1.00 | 6/3/503 (9) | 0.503 | [0.40 ; 0.86] | indeterminate | 7 (0.040 [0.016 ; 0.083]) | 7 (0.040 [0.016 ; 0.083]) | non | non |

#### Graine 2 : évaluations

| k | Itér. | B | Aléatoire V/D/N (déc.) | Score (nul = ½) | Wilson 1 côté [bas ; haut] sur décisives | Verdict | Gelé : buts pour (n, /min [Garwood 95 %]) | Gelé : buts contre | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 969 | 0.05 | 4/2/506 (6) | 0.502 | [0.35 ; 0.88] | indeterminate | 4 (0.023 [0.006 ; 0.058]) | 7 (0.039 [0.016 ; 0.081]) | non | non |
| 2 | 1938 | 0.10 | 4/9/499 (13) | 0.495 | [0.15 ; 0.54] | indeterminate | 6 (0.034 [0.013 ; 0.074]) | 5 (0.028 [0.009 ; 0.066]) | non | non |
| 3 | 2907 | 0.15 | 4/6/502 (10) | 0.498 | [0.19 ; 0.65] | indeterminate | 3 (0.017 [0.003 ; 0.049]) | 3 (0.017 [0.003 ; 0.049]) | non | non |
| 4 | 3876 | 0.20 | 3/8/501 (11) | 0.495 | [0.11 ; 0.52] | indeterminate | 5 (0.028 [0.009 ; 0.065]) | 7 (0.039 [0.016 ; 0.081]) | non | non |
| 5 | 4844 | 0.25 | 4/2/506 (6) | 0.502 | [0.35 ; 0.88] | indeterminate | 5 (0.028 [0.009 ; 0.066]) | 2 (0.011 [0.001 ; 0.041]) | non | non |
| 6 | 5813 | 0.30 | 4/5/503 (9) | 0.499 | [0.22 ; 0.70] | indeterminate | 5 (0.028 [0.009 ; 0.064]) | 0 (0.000 [0.000 ; 0.020]) | non | non |
| 7 | 6782 | 0.35 | 6/5/501 (11) | 0.501 | [0.32 ; 0.76] | indeterminate | 4 (0.022 [0.006 ; 0.057]) | 4 (0.022 [0.006 ; 0.057]) | non | non |
| 8 | 7751 | 0.40 | 6/4/502 (10) | 0.502 | [0.35 ; 0.81] | indeterminate | 8 (0.045 [0.020 ; 0.089]) | 3 (0.017 [0.004 ; 0.050]) | non | non |
| 9 | 8720 | 0.45 | 8/2/502 (10) | 0.506 | [0.54 ; 0.93] | indeterminate | 4 (0.023 [0.006 ; 0.058]) | 5 (0.028 [0.009 ; 0.066]) | non | non |
| 10 | 9688 | 0.50 | 11/3/498 (14) | 0.508 | [0.57 ; 0.91] | indeterminate | 9 (0.051 [0.023 ; 0.096]) | 7 (0.039 [0.016 ; 0.081]) | non | non |
| 11 | 10657 | 0.55 | 4/1/507 (5) | 0.503 | [0.44 ; 0.95] | indeterminate | 7 (0.040 [0.016 ; 0.083]) | 7 (0.040 [0.016 ; 0.083]) | non | non |
| 12 | 11626 | 0.60 | 10/5/497 (15) | 0.505 | [0.46 ; 0.83] | indeterminate | 9 (0.051 [0.023 ; 0.098]) | 8 (0.046 [0.020 ; 0.090]) | non | non |
| 13 | 12595 | 0.65 | 8/4/500 (12) | 0.504 | [0.43 ; 0.84] | indeterminate | 10 (0.058 [0.028 ; 0.106]) | 15 (0.087 [0.049 ; 0.143]) | non | **oui** |
| 14 | 13564 | 0.70 | 4/4/504 (8) | 0.500 | [0.25 ; 0.75] | indeterminate | 6 (0.034 [0.013 ; 0.075]) | 6 (0.034 [0.013 ; 0.075]) | non | non |
| 15 | 14532 | 0.75 | 7/5/500 (12) | 0.502 | [0.36 ; 0.78] | indeterminate | 9 (0.052 [0.024 ; 0.098]) | 6 (0.034 [0.013 ; 0.075]) | non | non |
| 16 | 15501 | 0.80 | 6/4/502 (10) | 0.502 | [0.35 ; 0.81] | indeterminate | 10 (0.059 [0.028 ; 0.108]) | 5 (0.029 [0.010 ; 0.068]) | non | **oui** |
| 17 | 16470 | 0.85 | 9/2/501 (11) | 0.507 | [0.57 ; 0.94] | indeterminate | 6 (0.034 [0.013 ; 0.075]) | 6 (0.034 [0.013 ; 0.075]) | non | non |
| 18 | 17439 | 0.90 | 12/3/497 (15) | 0.509 | [0.59 ; 0.92] | indeterminate | 4 (0.023 [0.006 ; 0.059]) | 6 (0.034 [0.013 ; 0.075]) | non | non |
| 19 | 18407 | 0.95 | 6/4/502 (10) | 0.502 | [0.35 ; 0.81] | indeterminate | 8 (0.046 [0.020 ; 0.091]) | 7 (0.040 [0.016 ; 0.083]) | non | non |
| 20 | 19376 | 1.00 | 9/3/500 (12) | 0.506 | [0.51 ; 0.90] | indeterminate | 3 (0.017 [0.004 ; 0.051]) | 13 (0.075 [0.040 ; 0.128]) | non | non |
