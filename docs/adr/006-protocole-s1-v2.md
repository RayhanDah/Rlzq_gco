# ADR-006 — Protocole S1 v2 (amende ADR-005 après le run 4a)

- Statut : **proposé, en attente de validation utilisateur**
- Date : 2026-11-18
- Décideurs : research-director (proposition) ; utilisateur (validation). Amende `docs/adr/005-run-4a.md` (D1, D2, D3, D6, D7)
  sans le réécrire : tout ce qui n'est pas cité ici reste en vigueur.

## Contexte
Run 4a terminé : GCO-std, 3 graines, 1B appris chacune (`docs/reports/run_4a.md` §1).
- **Protocole tel qu'exécuté** (§3, §6) : S1-léger jamais atteint ; (a) jamais remplie ; (b) vraie 7 fois sur 60 évaluations
  (s0 : 4, dont 3 de suite ; s1 : 1 ; s2 : 2). K2 à K7 non déclenchés ou non armés ; une restauration K4 sur s0.
- **Défaut de conception de (a)** (§3) : plancher de 20 décisives sur 512 épisodes, alors que la part de décisives contre
  l'aléatoire est de 2,1 à 2,2 % en moyenne (≈ 11 par évaluation). Une seule évaluation sur 60 atteint 20 décisives.
  Probabilité d'en avoir ≥ 20 sur 512 en fin de run : 17 %, 6 %, 1,6 % (s0, s1, s2) ; puissance du Wilson à 20 décisives :
  0,61 si le vrai taux est 0,70. (a) était presque impossible à remplir. Ce n'est pas la seule cause : sans plancher, (a) et (b)
  ne sont jamais vraies 3 fois de suite (§3, contrefactuel).
- **Défaut de (b)** (§3, lecture de (b)) : contre le gelé à 0,1B, buts pour et contre montent ensemble ; (b) mesure le rythme
  du duel, pas une domination. Le gelé glissant n'est pas stationnaire (ADR-005 D8).
- **Mesure hors ligne** (§4 bis, commit 6db79a4, 2026-11-18) : SPRT +100 Elo contre l'aléatoire accepté sur l'instantané 1B des
  3 graines ; Elo décisif +203 [+172 ; +235], +163 [+130 ; +199], +159 [+126 ; +196] ; (b) recalculée vraie sur s0 et s1.
  L'agent touche la balle dans 9 à 14 % des épisodes ; 81 à 86 % des épisodes finissent sans touche (§4 bis, §8).

## Options
| Sujet | Option | Pour | Contre |
|---|---|---|---|
| Statut de 4a | compter le hors-ligne comme S1-seed | GO de D7 | critère changé après avoir vu les données |
| | **exploratoire, post hoc** | honnête | G3 reste bloqué |
| Épisodes par point | 2 048 | coût ×2 | puissance ≈ 0,69 (calcul ci-dessous) |
| | **3 072** | puissance ≈ 0,89 | coût ≈ 5,4 % du run (estimé) |
| (b) | gelé glissant (actuel) | déjà codé | non stationnaire, mesure le rythme |
| | retiré | coût nul | s'écarte de `GCO_SPEC.md` §4 (S1 = marque contre lui-même gelé **et** SPRT) |
| | **contre θ₀ fixe** | stationnaire, conforme au §4 | une éval de plus par point |
| S1-léger | conservé ou modifié | continuité | redondant avec la v2, double les tests |
| | **retiré** | un seul critère | `s1_reached` à réarmer (K3, K7) |

## Mesures
- Évaluation hors ligne des instantanés : `scripts/eval.py --a <instantané> --b random --episodes 2048 --n-envs 256 --seed 0
  --sprt-elo 100 --sprt-max-decisive 1200` (20 000 épisodes à 1B), commit 6db79a4, laptop Core Ultra 5 125H, politiques sur
  CPU, 2026-11-18 ≈ 18:10–18:45 UTC ; sorties `runs/run_4a_offline_eval/*.json` (`run_4a.md` §4 bis).
  - Durée (champ `throughput.wall_s`) : 14,1 à 19,1 s par éval de 2 048 épisodes contre l'aléatoire (33 fichiers) ; 18,6 à 23,9 s
    contre le gelé (3 fichiers) ; 131 à 154 s pour 20 000 épisodes.
  - À 1B, 20 000 épisodes (`s{0,1,2}_final20k_it19376.json`) : V / N / D = 479 / 19 372 / 149, 366 / 19 491 / 143,
    320 / 19 552 / 128. Part de décisives **3,14 %**, 2,55 %, 2,24 % ; victoires par épisode 2,40 %, 1,83 %, 1,60 % ;
    taux de victoire décisive 0,763, 0,719, 0,714 (calculé sur ces décomptes).
  - θ₀ contre l'aléatoire : Elo décisif −39, +20, 0, aucune décision SPRT (§4 bis).
- Coût de l'éval en cours de run actuelle (512 + 512 épisodes, 20 évaluations) : 373 / 352 / 361 s au total, soit 17 à 18 s par
  évaluation et 1,8 à 1,9 % du temps (`evals.jsonl` `timing.total_s`, `run_4a.md` §3). Run de 1B : 5,44 à 5,51 h (§1).
- `R0` = 0,027 but/min par camp (P0, commit c83198e, 2026-11-14, ADR-005 « Mesure P0 complète »).
- SPRT : incréments ln(p1/p0) = +0,2470 et ln((1−p1)/(1−p0)) = −0,3288 avec p1 = 0,6401 (+100 Elo), bornes ±ln 19 = ±2,944
  (`src/eval/sprt.py:76-89`) ; vérifié sur `s0_b1.0_it19376.json` : 18 V, 4 D → LLR 3,131 (calculé) = 3,1306 (journal).

### Calcul de puissance (calculé, approximation brownienne du premier passage ; non simulé)
Hypothèse de conception : vrai taux de victoire décisive **p = 0,70** (+147 Elo), sous la plus faible graine mesurée à 1B
(0,714, s2). Dérive par décisive μ = 0,70 × 0,2470 − 0,30 × 0,3288 = 0,0743 ; σ = 0,264. Probabilité d'atteindre +2,944 en
n décisives : Φ((μn − A)/(σ√n)) + e^{2μA/σ²} Φ((−A − μn)/(σ√n)), e^{2μA/σ²} = 536. Borne basse négligée (≈ 0,3 % à p = 0,70).
| Épisodes | part de décisives 2,0 % | 2,2 % (s2 à 1B) | 2,6 % (s1 à 1B) |
|---|---|---|---|
| 2 048 | 41 déc. → 0,63 | 45 → 0,69 | 53 → 0,79 |
| **3 072** | 61 → 0,86 | 68 → **0,89** | 80 → 0,94 |
- À p = 0,64 (+100 Elo exact), 3 072 épisodes, 2,2 % : ≈ 0,6. Le protocole n'est pas puissant à l'effet minimal de H1 ; il l'est
  à l'effet observé en fin de run. Limite assumée.
- Sous H0 (p = 0,5, θ₀) : erreur α ≤ 0,05 par point (Wald ; la troncature la réduit). Avec la règle « 2 points consécutifs » :
  ≤ 0,0025 par paire, ≤ 0,0225 sur les 9 paires d'un run (borne de l'union, indépendance supposée).
- Puissance de la règle des 2 consécutifs, à 0,85 par point : 0,72 pour une paire donnée, 0,94 sur 4 points, 0,97 sur 5.
- (b) contre θ₀, 3 072 épisodes ≈ 1 050 min (20,5 s par épisode, estimé) : il faut ≥ 40 buts (Garwood bas de 40 = 28,6 >
  0,027 × 1 050 = 28,4). Si le taux contre θ₀ vaut celui de s2 contre l'aléatoire à 1B (0,047/min ; θ₀ ≈ aléatoire,
  hypothèse) : 49 buts attendus, puissance ≈ 0,92 (approximation normale, calculé).
- Les approximations (passage continu, contrôle aux frontières de paires, part de décisives aléatoire) surestiment un peu la
  puissance. Vérification Monte Carlo recommandée (eval-lead, non bloquante).
- **Vérification Monte Carlo (2026-11-18, faite)** : 4 000 tirages par cas, graine 12345, `eval.sprt.sprt_scan` (décision
  aux frontières de paires, plafond = épisodes du point), épisodes indépendants (la corrélation par paire de kickoff est
  ignorée), commit 7d7859c, laptop. Puissance à p = 0,70 : 2 048 épisodes → 0,61 / 0,68 / 0,78 (part de décisives 2,0 /
  2,2 / 2,6 %) ; **3 072 → 0,84 / 0,89 / 0,92**. À p = 0,64 (+100 Elo), 3 072, 2,2 % : 0,58. Sous H0 (p = 0,5), 3 072,
  2,2 % : 2,6 % de H1 acceptée par point. L'approximation ci-dessus est confirmée à ± 0,02.

## Décision

### D1. Statut des résultats 4a
- **S1 « hors ligne, post hoc, exploratoire »** : la mesure du §4 bis n'est pas S1-seed au sens de D2 (critère fixé après les
  données). Elle sert de référence descriptive, jamais de verdict.
- **K1 non déclenché** (lecture littérale de D6 : (b) remplie sur 3 graines sur 3). **H1 non concluante** : protocole
  sous-puissant, ni atteinte ni réfutée. **GO de D7 non atteint** : G3 reste fermé.
- Lecture rétrospective, exploratoire, avec la règle v2 sur les évals hors ligne de 2 048 épisodes ((b) contre θ₀ non mesurée) :
  1re paire SPRT à 0,5B (s0), 0,5B (s1), 0,3B (s2).

### D2. S1-seed v2 (pré-enregistré, remplace S1-léger et S1-seed d'ADR-005 D2 pour tous les runs suivants)
- **Cadence** : tous les 0,1B appris (10 points par run), sur l'instantané du point (instantanés toujours gardés tous les 0,05B).
- **Épisodes** : 3 072 contre l'aléatoire **et** 3 072 contre θ₀ (instantané de l'itération 0 du run), `n_envs` 256 fixe,
  graine d'éval = graine du run, tous les épisodes joués (taille fixe, pour l'Elo et les touches).
- **Point réussi** si les deux :
  (a) SPRT H0 : 0 contre H1 : +100 Elo, α = β = 0,05, sur les décisives, nuls exclus, **sans plancher de décisives**, rejoué
  dans l'ordre des paires (comme `scripts/eval.py --sprt-elo 100`) : décision « accepte H1 ». Plafond atteint sans décision
  = point non réussi ;
  (b) voir D3.
- **`s1_reached`** : armé, de façon irréversible, au **2e de 2 points consécutifs réussis** ; événement journalisé. Il
  conditionne K2 (STOP avant S1), K3 et K7 comme dans ADR-005 D6.
- **S1-seed officiel** = 1er point de la paire ; « steps jusqu'à S1 » = steps de ce point (résolution 0,1B).
- **Confirmation hors ligne** (dès l'armement, hors du run) : `scripts/eval.py --a <1er point> --b random --episodes 20000
  --n-envs 256 --seed 1 --sprt-elo 100 --sprt-max-decisive 1200` (graine 1 : échantillon indépendant de l'éval en cours de run),
  plus (b) contre θ₀ sur 3 072 épisodes, graine 1. Confirmé si SPRT accepte H1 et (b) vraie. Non confirmé : S1-seed
  « non confirmé » au rapport ; `s1_reached` reste armé (gardes plus prudentes).
- **S1-léger (512 épisodes, 0,05B) retiré.** Le plancher de 20 décisives disparaît du critère.
- **K7** : après `s1_reached`, borne haute du Wilson unilatéral 95 % des victoires décisives contre l'aléatoire < 0,5 (≥ 20
  décisives) sur **2** points consécutifs (même étendue ≈ 0,2B que 3 × 0,05B) : alerte.
- **Elo ancré** : Elo décisif contre l'aléatoire avec IC bootstrap par paires (déjà calculé par l'éval) journalisé à chaque point.
- **Coût** : ≈ 18 s par 1 024 épisodes en cours de run (mesuré, §Mesures) ; 6 144 épisodes × 10 points ≈ 1 060 s ≈ **5,4 %**
  d'un run de 1B (estimé, non mesuré, extrapolation linéaire), contre 1,9 % aujourd'hui. Cible d'ADR-005 (≤ 5 %) relevée à
  ≤ 8 % ; révision si le coût mesuré dépasse 10 % (seuil existant).

### D3. Critère (b) : contre θ₀ fixe
- **Formule** : borne basse de l'intervalle de **Garwood bilatéral 95 %** des buts de A par minute jouée contre θ₀ > **`R0`
  = 0,027** but/min par camp, sur les 3 072 épisodes du point (`eval.rating.garwood_interval`).
- Justification : `GCO_SPEC.md` §4 définit S1 par « marque contre lui-même gelé » **et** « SPRT +100 Elo contre l'aléatoire » ;
  retirer (b) changerait la spec. θ₀ est fixe, donc la mesure est stationnaire (lève la menace D8 du gelé glissant), et (b)
  exige un taux de but absolu, ce qui écarte une politique passive qui gagnerait peu de duels rares. Prix : une éval de plus.
- θ₀ ≈ aléatoire (Elo −39, +20, 0) : (b) est proche d'un second ancrage, pas d'une mesure de progression en self-play. Assumé.
- **K1 reformulé** : une ligne (variante + config) à 3 graines dont aucune n'atteint S1-seed v2 à 1B. Non applicable aux runs à
  une graine de cet ADR.

### D4. Levée de la condition D3 d'ADR-005 pour GCO-S graine 0
- Motif : `GCO_SPEC.md` §10 H1 exige un témoin GCO-S en run court ; la condition « ≥ 1 graine std a S1-léger » portait sur un
  critère presque impossible à remplir (Contexte).
- **Run** : GCO-S graine 0, 1B appris, `configs/run_4a_s.yaml` inchangé hors protocole d'éval (D2, D3) et journal (D7). Run
  court de Phase 4a, pas un run long : G3 n'est pas en jeu. Coût ≈ 5,5 h (durée std mesurée, transposée, estimé).
- **Comparaison à std graine 0** (évaluée avec l'ancien protocole en cours de run) : uniquement par la **même commande hors
  ligne** que la référence `runs/run_4a_offline_eval/s0_*` : instantanés à chaque 0,1B, 2 048 épisodes, `--seed 0`, SPRT
  +100 Elo, plafond 1 200, et 20 000 épisodes `--seed 0` à 1B. Les évals en cours de run ne sont pas comparées entre runs.
- H2 reste **descriptive** (une paire) : date de la 1re paire SPRT hors ligne (référence std s0 : 0,5B), Elo décisif, touches.
- Vérification préalable : θ₀ identique entre std s0 et S s0 (hash des poids de l'itération 0).

### D5. Règle de décision après GCO-S (pré-enregistrée)
- Mesure : victoires de A par épisode contre l'aléatoire à 1B, 20 000 épisodes, `--seed 0` (commande de `s0_final20k`).
- **« S < std »** ⇔ borne haute du Wilson bilatéral 95 % de GCO-S s0 < **1,60 %**, la plus basse des 3 graines std à 1B
  (s2 : 320 / 20 000). Soit ≈ moins de 286 victoires sur 20 000 (calculé).
- Pourquoi l'étendue des 3 graines et pas s0 seule : la variance entre graines domine (1,60 à 2,40 %, rapport 1,5). Comparée à
  s0 seule (2,40 %), la graine std s1 (1,83 %, borne haute ≈ 2,02 %) serait elle-même « < std ».
- **S < std** → intervention (1) : calendrier d'états génériques plus lent (les états génériques ont aidé). **Sinon** →
  intervention (2) : γ 0,995 → 0,998 (rien ne montre que les états génériques aident). Les deux restent GCO-std (aucun signal
  ajouté, C1–C4 non sollicités) ; chacune a son ADR (paramètres exacts fixés là, p. ex. 0,2 → 0,8 sur 1B pour (1)).
- Défaut assumé : la règle est asymétrique ; en cas d'ambiguïté, elle choisit (2).

### D6. Critère de succès d'une intervention (GCO-std graine 0, 1B)
- **Succès** si l'un des deux :
  (i) à 1B, sur 20 000 épisodes contre l'aléatoire `--seed 0` : borne basse du Wilson bilatéral 95 % de la part de décisives
  ≥ **6,28 %** (2 × 3,14 %, s0) **et** SPRT +100 Elo accepte H1 sur la même éval (exclut « plus de chaos sans domination ») ;
  (ii) SPRT durable avant 0,3B : sur les évals hors ligne à 2 048 épisodes `--seed 0`, 1re paire de points consécutifs avec H1
  au plus tard à **0,2B** (paire complète ≤ 0,3B).
- Calibrage sur la ligne de base : 0 graine std sur 3 ne remplit (i) (3,14 / 2,55 / 2,24 %) ni (ii) (1res paires à 0,5 / 0,5 /
  0,3B).
- **Neutre** sinon : configuration de base conservée, intervention suivante de `GCO_SPEC.md` §8, un ADR chacune.
- Mesures en cours de run (D2) : décrites au rapport, jamais utilisées pour ce critère.

### D7. Instrumentation exigée (diagnostic seulement, jamais dans la perte ni dans un critère d'arrêt)
**Avant GCO-S (bloquant)** :
- **I1** Éval en cours de run S1-seed v2 (D2, D3) : 3 072 + 3 072 épisodes tous les 0,1B ; dans `evals.jsonl` par point :
  V/N/D, Elo décisif + IC bootstrap, SPRT (décision, LLR, épisodes et décisives à la décision), buts/min pour et contre θ₀ avec
  Garwood, (b), point réussi, `s1_reached`, K7, durée. Config d'éval dédiée ; `configs/run_4a_eval.yaml` inchangé.
- **I2** Touches à chaque éval (contre l'aléatoire et contre θ₀) : part d'épisodes où A touche, où l'adversaire touche, sans
  touche, et moyennes par épisode (champ `touches` de `scripts/eval.py`, 6db79a4, porté dans `evals.jsonl`).
- **I3** Cause de fin d'épisode à l'entraînement, par itération : but, no-touch 20 s, 120 s (compteurs dans `metrics.jsonl`).
- **Condition** : un test montre que I1 à I3 ne changent pas la trajectoire d'entraînement (même graine : mêmes métriques
  PPO qu'avant sur un run court), puis smoke test vert sur le commit du run (CLAUDE.md règle 5).

**Avant le run d'intervention (peut attendre GCO-S, où p_kick = 1 rend l'origine triviale)** :
- **I4** Buts d'entraînement par origine d'épisode (kickoff / état générique) par itération (exigé par ADR-005 D4, jamais fait).

**Optionnel** : RAM (disponible et privée du processus) toutes les N itérations.

## Conséquences
- 4a est clos comme exploratoire. Les runs suivants ont un critère S1 puissant (≈ 0,89 par point à l'effet de fin de run),
  pré-enregistré, conforme à `GCO_SPEC.md` §4. Le GO de D7 (vers G3) exige S1-seed v2 **confirmé** sur ≥ 1 run GCO-std conduit
  sous ce protocole ; avec une graine, mention « amorçage fragile ».
- Coût : ≈ 5,4 % de temps d'éval par run (estimé) ; GCO-S ≈ 5,5 h, puis une intervention ≈ 5,5 h.
- **Menaces** :
  1. une graine par variante et par intervention : la variance entre graines (×1,5 sur les victoires à 1B) domine ;
  2. critère conçu après avoir vu 4a (jardin des chemins) : limité par le statut exploratoire de 4a et l'enregistrement de
     cet ADR avant tout run ;
  3. comparaisons entre commits et protocoles : uniquement par la commande hors ligne identique ; I1 à I3 ne doivent pas
     changer la trajectoire ;
  4. puissance calculée par approximation, à la part de décisives de fin de run : plus tôt, elle est plus faible, d'où une
     date de S1 biaisée vers le tard (plus 0,1B de délai par la règle des 2 points) ;
  5. GCO-S s'entraîne à 100 % sur kickoffs et l'éval n'utilise que des kickoffs : avantage de distribution pour S, qui
     pousse la règle D5 vers (2) ;
  6. θ₀ ≈ aléatoire : (b) n'informe pas sur la progression en self-play ;
  7. reprises après `DEVICE_LOST` non bit à bit, restaurations K4 : runs « repris » au rapport ;
  8. multiplicité : 10 points par run, 2 tests par point ; règle des 2 points ≤ 2,25 % de faux S1 par run (borne) ;
  9. écart sim / jeu non mesuré (R13) : S1 vaut en sim seulement.
- **Révision** si : coût d'éval mesuré > 10 % ; part de décisives < 1,5 % aux points tardifs (puissance < 0,6) ; Monte Carlo
  de la puissance < 0,8 au point de conception ; publication des codes Terminal / Pulsar 2.5 (spec §8).

## Implémentation et vérifications (2026-11-18, avant validation)
- I1, I2 : `src/training/evalhook.py` (`protocol: s1_v2`), `configs/run_s1v2_eval.yaml`, `scripts/run_4a.py --eval-config` ;
  I3, I4 : `src/env/batch_env.py`, `src/training/rollout.py`, `src/training/logging.py` (scalaires `env/ends_*`,
  `env/goals_kickoff|generic`). Fusions 07bc6d3 et 07d51e6, revues `code-reviewer` APPROVE (majeur corrigé : test du
  plafond du SPRT, vu échouer). Tests : 601 unitaires, 142 d'intégration avec RocketGPU (2 xfail connus, R13).
- Smoke test : **PASS** sur 07d51e6 (`scripts/train.py configs/smoke.yaml --smoke`, `runs/smoke_07d51e6b/`, XPU, reprise
  exacte, 0 NaN, ≈ 58k pas/s sur la boucle, non soutenu) ; les scalaires I3/I4 sont présents dans `metrics.jsonl`.
- Essai réel du protocole v2 (07d51e6, 2026-11-18, laptop, XPU) : `scripts/run_4a.py --variant s --seed 0 --std-s1-observed
  --max-hours 0 --max-iterations 4 --eval-config <copie de run_s1v2_eval.yaml avec un point toutes les 2 itérations>
  --runs-root runs/trial_v2_07d51e6 --checkpoints-root checkpoints/trial_v2_07d51e6`. **Coût mesuré d'un point v2
  (2 × 3 072 épisodes) : 64,6 s et 63,7 s** (26,7 s contre l'aléatoire, 36,8 s contre θ₀) ; 10 points ≈ 11 min, ≈ 3,2 % d'un
  run de 5,5 h (extrapolé). Journal conforme (SPRT, (b) contre θ₀, touches, `s1_reached_before`).
- **θ₀ identique** entre std s0 (`checkpoints/run_4a_std_s0/eval_snapshots/ckpt_000000_it00000000.pt`) et GCO-S s0 :
  SHA-256 des poids de la politique 7fab4c74… pour les deux.
- Smoke test **PASS** aussi sur **1b98a0d** (commit final, après fusion des interventions ADR-007/008 et du contraste P0 :
  `runs/smoke_1b98a0d/`, 0 NaN, reprise exacte, ≈ 57k pas/s non soutenu) ; essai du lanceur GCO-S sur 1b98a0d
  (`--max-iterations 1`, `runs/trial_s_1b98a0d/`) : la session enfant reçoit `--eval-config run_s1v2_eval.yaml`.
- Commande de GCO-S graine 0 (après validation, sur 1b98a0d ou un commit sans changement de `src/`) :
  `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant s --seed 0 --max-hours 0 --eval-config run_s1v2_eval.yaml
  --std-s1-observed`. Le drapeau `--std-s1-observed` est le garde-fou d'ADR-005 D3 ; ici il vaut levée par D4 de cet ADR.

## Points à valider par l'utilisateur
1. Statut de 4a : exploratoire ; K1 non déclenché ; H1 non concluante ; G3 fermé.
2. S1-seed v2 : tous les 0,1B, 3 072 + 3 072 épisodes, SPRT sans plancher, 2 points consécutifs, confirmation 20 000 épisodes
   (graine 1) ; S1-léger retiré ; cible de coût relevée à ≤ 8 % (≈ 5,4 % estimé).
3. (b) contre θ₀ fixe, Garwood bilatéral 95 %, `R0` = 0,027 ; K1 reformulé sur 3 graines.
4. Levée de la condition D3 : GCO-S graine 0, 1B, `run_4a_s.yaml` hors éval ; comparaison à std s0 par la commande hors ligne.
5. Règle D5 : « S < std » si la borne haute de GCO-S est sous 1,60 % de victoires par épisode, alors (1), sinon (2).
6. Succès d'intervention : part de décisives ≥ 6,28 % (borne basse) avec SPRT H1 à 1B, ou 1re paire SPRT ≤ 0,2B ; neutre sinon.
7. Instrumentation I1 à I3 avant GCO-S, I4 avant l'intervention, RAM optionnelle.
