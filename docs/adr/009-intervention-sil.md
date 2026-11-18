# ADR-009 — Intervention (3) : self-imitation learning (SIL) sur nos buts (GCO-std, graine 0, 1B)

- Statut : **proposé** (validation utilisateur requise avant tout lancement ; D9 propose un changement d'ordre des
  interventions qui amende ADR-005 D6, **sans toucher ADR-006**)
- Date : 2026-11-18
- Décideurs : research-director (proposition) ; rl-algo-lead (implémentation, `configs/int_sil.yaml`, `--variant int_sil`,
  désactivée par défaut) ; utilisateur (validation, lancement)

## Contexte
- Intervention (3) de `docs/GCO_SPEC.md` §8 et d'ADR-005 D6. La spec l'autorise en GCO-std (§3, ligne « Self-imitation (SIL/UPGO
  sur nos propres épisodes avec but) » : interdit en GCO-S, **autorisé** en GCO-std, « n'utilise que ±1 ») sous C1–C4.
- Diagnostic du run 4a (`docs/reports/run_4a.md` §4 bis, §8 ; mesures hors ligne, commit 6db79a4, 2026-11-18) : l'agent bat
  l'aléatoire (Elo décisif +159 à +203 à 1B) mais ne touche la balle que dans 8,7 à 13,9 % des épisodes kickoff ; 81,0 à 86,1 %
  des épisodes finissent sans touche. La politique bouge à peine : entropie 94 à 97 % de ln 90, KL ≈ 1 à 2·10⁻⁴, clip < 0,1 %,
  normes de gradient 0,05 à 0,12 (annexe A). À l'entraînement, 3,8 à 7,4 buts par itération de 51 611 pas appris (annexe A,
  `env/goals`).
- Lecture : les rares trajectoires qui marquent existent (≈ 4 à 7 par itération), mais leur crédit est noyé dans un gradient
  d'avantages normalisés presque entièrement fait de bruit. La SIL (Oh et al. 2018) réutilise exactement ces trajectoires, sans
  autre signal que le ±1. Le seul point de données GCO à petit budget (bot communautaire, « commence à marquer à 350M »,
  fiabilité moyenne, spec §5) utilise SIL **et** des learned state setters : effet de la SIL seule inconnu.

## Options
| Option | Pour | Contre |
|---|---|---|
| **SIL de Oh et al. (perte −log π·(R−V)₊ + β·½·((R−V)₊)², buffer priorisé, retours MC des épisodes terminés)** | publiée, testée avec PPO (Oh §5.6, Tab. 4) ; n'utilise que ±1 ; une perte de plus, PPO inchangé | off-policy sans poids d'importance (Oh §5.5) ; argument de borne inférieure valable en environnement **déterministe** seulement (Oh §4.2) ; RAM du buffer ; code à écrire |
| UPGO (AlphaStar) | utilise V pour les retours | spec AlphaStar non relue ici ; plus de code (traces) ; aucun réglage publié pour Rocket League |
| PPO+D (Libardi et al. 2020) : rejouer ses meilleures trajectoires dans le lot PPO | réutilise la perte PPO | importance sampling sur des données anciennes, ratio de rejeu à régler ; non relu dans cette phase |
| SIL à coefficient constant sur tout le run | effet maximal | viole C2 (pas d'annealing à 0) ; gain non séparé de la présence de la SIL |

## Mesures
Aucune mesure d'entraînement SIL : décision réversible, fondée sur Oh et al. 2018 (arXiv 1806.05635, lu le 2026-11-18 via
ar5iv : §3 Éq. 1–3 et Algorithme 1, §4.2, §5.3–5.6, Annexe A Tab. 3–4), sur le rapport 4a et sur ADR-001 M8.
- **Oh et al., réglages publiés.** A2C+SIL (Tab. 3) : 4 mises à jour SIL par itération, lot 512, poids SIL 1, β 0,01, buffer
  10⁵, exposant de priorité 0,6, correction de biais 0,1 à 0,4. **PPO+SIL (Tab. 4)** : 10 mises à jour SIL par lot PPO de 2 048,
  lot SIL 512, **poids SIL 0,1**, **β ∈ {0,01 ; 0,05}**, buffer 50 000, exposant ∈ {0,6 ; 1,0}, correction 0,1, γ 0,99. Les
  épisodes n'entrent dans le buffer qu'à leur terminaison (Algorithme 1). Limite déclarée : exploitation excessive au début
  nuisible sur certains jeux (§5.4) ; inutile si aucune expérience positive n'est jamais vue (Venture, §5.3).
- **Données de 4a** (annexe A du rapport) : 3,8 à 7,4 buts par itération ; épisodes de ≈ 305 pas (≈ 20,3 s) ; observation de 109
  float32 (`src/obs/schema.py:37`) ; 4 096 arènes × 2 voitures.
- **Mémoire** (ADR-001 M8, commit 6998993, 2026-11-14, banc 2 h, pool plein) : mémoire privée ≈ 6,1 Go, **disponible minimale
  2,73 Go** sur 16 Go. M6 : chute de débit coïncidant avec moins de 1 Go disponible (cause non établie).
- **Calculs de cet ADR (exacts aux arrondis, non mesurés)** :
  - fenêtre de collecte : 8 192 agents × 300 pas × 109 × 2 octets (float16) = **0,54 Go** (1,07 Go en float32) ;
  - buffer : 250 000 transitions × ≈ 232 octets (obs float16 218, action 1, R 4, priorité 4, origine 1, itération 4) = **58 Mo** ;
    arbre de sommes ≈ 4 Mo ; total SIL ≈ **0,60 Go**, d'où une mémoire disponible minimale ≈ 2,1 Go (**estimé, non mesuré**) ;
  - entrées dans le buffer : ≤ 7,4 buts × 300 pas ≈ 2 200 transitions par itération (borne haute ; les buts depuis états
    génériques sont sans doute plus courts) : le buffer couvre ≥ 110 itérations (≥ 5,8M pas appris) ;
  - échantillons par itération : PPO 2 epochs × 51 611 = 103 222 ; SIL (D3) 2 × 8 192 = 16 384, soit **0,16** (Oh PPO+SIL :
    5 120 / 20 480 = 0,25 ; Oh A2C+SIL : 2 048 / 80 = 25,6). Pas d'optimiseur : PPO 6 par itération (3 minibatchs × 2 epochs,
    `src/training/ppo.py:256`), SIL 2.

## Décision

### D1. Données du buffer (C1 : seuls transitions, ±1 et drapeaux de l'environnement)
- **Épisodes admis** : uniquement ceux qui finissent par un **but** (`terminated`), et seulement côté **agent appris**
  (`learn_mask` ; le camp gelé des arènes contre le passé est exclu) dont l'équipe reçoit **+1** (csc compris, `GCO_SPEC.md` §2.1).
- **Retour** : Monte Carlo exact, R_t = γ^(T−t)·r_T avec r_T = +1 (la récompense est nulle ailleurs, §2.1), γ = `ppo.gamma`
  (0,995 ; mêmes valeurs que le critique).
- **Épisodes tronqués (no-touch 20 s, 120 s) : jamais admis.** Leur retour MC n'existe pas ; le compléter par V(s′) rendrait la
  cible circulaire (R dépendrait du critique qu'on compare à R), et le traiter comme 0 contredirait `GCO_SPEC.md` §2.2
  (« une troncature n'est jamais un terminal à 0 »).
- **Épisodes perdants (r_T = −1) : exclus** en v1. Écart à Oh (qui garde tout) : (R − V)₊ n'y est positif que si le critique est
  plus pessimiste que l'issue ; gain attendu faible pour une RAM doublée. Ablation possible plus tard (nouvel ADR).
- **Fenêtre** : seuls les **300 derniers pas** (20 s) avant le but sont gardés (γ³⁰⁰ = 0,22) ; un épisode plus long perd sa tête,
  sans biais sur R. Fenêtre circulaire par agent, obs en **float16** si le test D7-5 passe, sinon float32 avec 150 pas
  (0,54 Go, même budget).
- L'origine de l'épisode (kickoff / générique) est stockée pour le **diagnostic seulement** ; ni l'échantillonnage ni la perte
  ne la lisent (test D7-6).

### D2. Perte et priorités (Oh et al. §3, Éq. 1–3)
- L_sil = c_sil(t) · moyenne_lot[ w_i · ( −log π(a_i|s_i)·(R_i − V(s_i))₊ + β · ½·((R_i − V(s_i))₊)² ) ], (R − V)₊ sans
  gradient dans le terme de politique (poids d'échantillon, Oh §3).
- **β = 0,01** (Oh Tab. 3, et borne basse de Tab. 4). **c_sil = 0,1** au plateau (Oh Tab. 4, PPO+SIL).
- Priorité p_i = ((R_i − V(s_i))₊ + 10⁻³)^**0,6** (exposant Oh Tab. 3–4) ; transition neuve à la priorité maximale courante ;
  priorités mises à jour après chaque pas SIL avec le V du pas. Poids d'importance w_i = (N·P(i))^(−0,1) / max_j w_j :
  « correction de biais 0,1 » d'Oh Tab. 4, lue selon la convention de Schaul et al. (prioritized replay) ; Oh ne la définit pas
  (interprétation, signalée).
- Aucune normalisation de (R − V)₊ (contrairement aux avantages PPO, `ppo.py:276`) : c'est le poids d'Oh.

### D3. Mises à jour (une intervention, PPO inchangé)
- Après les 2 epochs PPO de chaque itération : **2 pas SIL**, lot de **8 192** transitions tirées du buffer, si le buffer contient
  ≥ 8 192 transitions (sinon aucun pas, journalisé).
- **Optimiseur Adam séparé** (mêmes β, ε et calendrier de LR 2·10⁻⁴ → 5·10⁻⁵ que PPO) ; clip global 0,5 propre au pas SIL.
  Ainsi l'état Adam de PPO n'est jamais modifié par la SIL, et `sil.enabled: false` laisse la trajectoire bit à bit identique
  à std (test D7-1).
- Pas de transfert exact possible des ratios d'Oh (0,25 à 25,6 selon le cadre) : 2 pas de 8 192 (ratio d'échantillons 0,16,
  D Mesures) est une **proposition dérivée** entre les deux cadres publiés. Le journal C4 (D5) dira si la SIL pèse.
- **Buffer : 250 000 transitions**, éviction FIFO ; plafond fixé par la RAM (D Mesures), pas par une source.

### D4. Calendrier de c_sil (C2)
- c_sil = **0,1 de 0 à 0,5B**, puis **linéaire de 0,1 à 0 entre 0,5B et 0,9B**, **0 de 0,9B à 1B** (≈ 1 938 itérations de PPO pur ;
  collecte et pas SIL arrêtés, buffer libéré).
- Pourquoi dans ce run : C2 exige un coefficient nul avant la phase finale ; un gain qui disparaît sans la SIL ne servirait pas la
  ligne principale. La fenêtre du critère (ii) d'ADR-006 D6 (≤ 0,3B) est entièrement à plein coefficient ; le critère (i) à 1B
  mesure une politique qui a fini en GCO pur.
- Diagnostic de persistance : éval hors ligne à 0,9B et 1,0B (déjà dans D8) ; « gain dépendant de la SIL » si la borne haute
  (Wilson 95 %) de la part de décisives à 1,0B est sous la borne basse à 0,9B (descriptif, au rapport).

### D5. Journal C4 (`metrics.jsonl`, chaque itération ; `sil/*`)
`coef`, `steps_done`, `buffer_size`, `buffer_new`, `buffer_age_mean` (itérations), `buffer_generic_frac` (diagnostic),
`loss_policy`, `loss_value`, `frac_pos` (part des transitions tirées avec (R−V)₊ > 0), `adv_pos_mean`, `adv_pos_max`,
`grad_norm_pi_sil` et `grad_norm_pi_ppo` (normes avant clip sur les **paramètres de la politique**, moyennes de l'itération),
`rho` = grad_norm_pi_sil / (grad_norm_pi_sil + grad_norm_pi_ppo), `ram_available_mb` et `ram_private_mb` (RAM rendue
**obligatoire** pour ce run).
- La SIL n'injecte rien dans les avantages : la métrique C4 « part de la variance des avantages » se transpose en **rho**.
  **Alerte C4** : rho > 0,5 pendant 50 itérations consécutives (journalisée, au rapport).

### D6. Conformité C1–C4 (`GCO_SPEC.md` §3)
- **C1** : aucune notion de Rocket League. Entrées : transitions (s, a), récompense ±1, drapeau `terminated`, `learn_mask`, γ, V
  appris. L'origine d'épisode n'entre ni dans la perte ni dans le tirage (test D7-6).
- **C2** : la SIL n'est pas une récompense, mais elle biaise la politique et le critique (exploitation de trajectoires chanceuses :
  l'environnement est stochastique, adversaire échantillonné et états génériques, hors du cadre déterministe d'Oh §4.2 ; et le
  terme de valeur pousse V vers le haut là où V < R). D'où l'annealing à 0 (D4), appliqué aux deux termes.
- **C3** : `sil.enabled` (défaut `false`) dans `configs/int_sil.yaml` ; désactivée, trajectoire identique à std (test D7-1).
  Ablation : std s0 (même config sans SIL) et GCO-S s0 existent déjà ; aucun run GCO-S supplémentaire pour cet ADR.
- **C4** : D5.

### D7. Tests exigés avant le run (test-engineer, rl-algo-lead ; Definition of Done)
1. `sil.enabled: false` : métriques PPO et poids identiques bit à bit à std sur N itérations (CPU, même graine).
2. Aucune transition d'épisode tronqué, perdant, ou du camp gelé n'entre dans le buffer (épisodes synthétiques).
3. Retours MC exacts sur épisodes synthétiques (γ^(T−t), fenêtre de 300, épisode plus long que la fenêtre).
4. Reprise exacte en CPU avec l'état SIL (buffer, priorités, fenêtres, Adam SIL, RNG) dans le checkpoint.
5. Parité float16 : |Δ log π| max ≤ 10⁻² et |ΔV| max ≤ 10⁻³ entre obs float32 et float16 sur un lot réel ; sinon float32 (D1).
6. La perte et le tirage sont invariants si l'on permute le champ « origine ».
7. Valeurs de c_sil(t) aux bornes du calendrier D4.
Puis : smoke test vert sur le commit du run (CLAUDE.md règle 5), et **banc de 30 min** (`scripts/bench_run.py`, config int_sil,
pool plein) mesurant débit appris et RAM disponible minimale, consignés ici avant le GO.

### D8. Protocole et critère de succès
- Run : `int_sil_s0`, GCO-std, **graine 0, 1B appris**, config = `configs/run_4a_std.yaml` + bloc `sil` (seule différence, vérifiée
  par test comme pour ADR-007/008). Éval en cours de run : protocole v2 (ADR-006 D2, D3), décrite au rapport, jamais comparée.
- Commande (après GO) : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant int_sil --seed 0 --max-hours 0`.
- Comparaison **uniquement** par la commande hors ligne d'ADR-006 D4, aux itérations de std s0 (0, 1 938, …, 19 376) :
  `scripts/eval.py --a <instantané> --b random --episodes 2048 --n-envs 256 --seed 0 --sprt-elo 100 --sprt-max-decisive 1200
  --out runs/int_sil_offline_eval/s0_b<0.k>_it<it>.json`, puis 20 000 épisodes à 1B.
- **Succès = ADR-006 D6, inchangé** : (i) à 1B, 20 000 épisodes `--seed 0`, borne basse du Wilson bilatéral 95 % de la part de
  décisives ≥ **6,28 %** **et** SPRT +100 Elo accepte H1 ; ou (ii) 1re paire de points consécutifs avec H1 (évals à 2 048)
  au plus tard à **0,2B**. Référence std s0 : 3,14 %, 1re paire à 0,5B. **Neutre** sinon.
- Comparaisons descriptives (aucun test) : std s0, GCO-S s0, et l'intervention ADR-007 ou ADR-008 si elle a tourné ; taux de
  touche (champ `touches`), Elo décisif par point, `sil/*`.
- Si succès : prochaines étapes proposées (décision utilisateur) : graines 1 et 2 de `int_sil` (K1 exige 3 graines ; G3 exige
  S1-seed v2 confirmé, ADR-006 Conséquences). Si neutre : intervention suivante (D9).

### Kill criteria (en plus de K2 à K6 d'ADR-005 D6 amendés par ADR-006, inchangés)
| # | Critère | Action |
|---|---|---|
| KS1 | Mémoire disponible < 1,0 Go pendant > 10 min | STOP propre ; buffer et fenêtre révisés par amendement |
| KS2 | rho > 0,8 pendant 500 itérations consécutives (≈ 26M appris) tant que c_sil > 0 | STOP : la SIL a remplacé PPO |
| KS3 | Après 0,3B, borne haute de l'IC bootstrap de l'Elo décisif contre l'aléatoire < 0 sur 2 points v2 consécutifs | STOP : pire que l'aléatoire |
| KS4 | Débit appris < 40k pas/s sur 30 min (−20 % de 4a) | alerte, au rapport (K6 à 25k inchangé) |
Les touches restent un diagnostic, jamais un critère d'arrêt (ADR-006 D7). Pas d'arrêt anticipé pour succès ni pour absence
de but avant 1B (ADR-005 D3).

### D9. Ordre des interventions : SIL avant ADR-007/008 (recommandation)
**Recommandation** : lancer la SIL **avant** l'intervention (1) ou (2), si elle est prête (fusionnée, revue, tests D7 verts, smoke
PASS, banc fait) quand le laptop se libère ; sinon lancer ADR-007/008 selon ADR-006 D5 pour ne pas laisser la machine inactive.
Le témoin dense (ADR-011), s'il est accepté, passe avant les deux.
- **Pour** :
  1. Le goulot mesuré (peu de touches, politique quasi uniforme, KL 10⁻⁴) est un défaut d'**amplification** des rares succès.
     (1) n'augmente la part moyenne d'états génériques que de 0,35 à 0,50 (ADR-008, ×1,43) alors que std en voyait déjà ;
     (2) agit sur la cible du critique (EV déjà 0,96) et laisse la fenêtre GAE à 1,22 → 1,28 s (ADR-007). Aucune des deux ne
     change le poids des ≈ 4 à 7 trajectoires gagnantes par itération ; la SIL le fait directement.
  2. Seul point GCO public à petit budget : SIL (+ LSS), buts à 350M (spec §5, fiabilité moyenne, confondu avec les LSS).
  3. Coût égal par run (≈ 5,5 h, durée mesurée de 4a transposée, estimé) ; l'utilisateur et le tech lead jugent (1)–(2) peu
     susceptibles de suffire (avis, pas une mesure).
  4. Aucun critère ne bouge : ADR-006 D5 désigne toujours (1) ou (2) d'après GCO-S s0, D6 juge chaque intervention contre std s0
     (pas de base cumulée). Seul le calendrier change, enregistré avant le run.
- **Contre** : l'ordre de §8 était pré-enregistré (ADR-005 D6) et on le change après avoir vu 4a (risque de chemins multiples,
  limité par les critères fixes) ; la SIL coûte du code, de la RAM et des risques (exploitation précoce, Oh §5.4) quand (1)–(2)
  sont prêts sans code ; si la SIL réussit, (1)–(2) deviendront des ablations sur une autre base.
- Cet ordre amende ADR-005 D6 (ordre de §8) : validation utilisateur. `GCO_SPEC.md` §8 reste un classement a priori, inchangé.

## Conséquences
- Plus facile : seule intervention qui vise directement le crédit des rares buts ; PPO, récompense et éval inchangés.
- Plus difficile : premier code d'entraînement hors PPO (buffer, fenêtre, second optimiseur, état de checkpoint ≈ 0,6 Go de plus
  par checkpoint, coût d'écriture à mesurer au banc) ; une graine (variance ×1,5 entre graines std à 1B) : un « neutre » ne réfute
  pas la SIL.
- Menaces : exploitation de buts chanceux (stochasticité) ; buffer dominé par des buts depuis états génériques (suivi par
  `buffer_generic_frac`), dont le transfert au kickoff n'est pas garanti ; c_sil emprunté à MuJoCo.
- Révision si : rho < 0,05 pendant tout le plateau (la SIL ne pèse pas : tester c_sil = 1, valeur d'Oh Tab. 3, par nouvel ADR) ;
  KS1 au banc ; `frac_pos` < 1 % durablement (critique déjà optimiste, buffer inutile) ; publication des codes Terminal /
  Pulsar 2.5 (spec §8).

## Points à valider par l'utilisateur
1. Réglages D1–D4 : buts gagnants seulement, tronqués exclus, fenêtre 300 pas, buffer 250 000, c_sil 0,1, β 0,01, priorités 0,6
   et correction 0,1, 2 pas SIL de 8 192 par itération, Adam séparé, annealing 0,5B → 0,9B puis 0.
2. Journal C4 (rho) et KS1 à KS4 ; RAM obligatoire pour ce run.
3. Critère de succès = ADR-006 D6, comparaison hors ligne à std s0.
4. Ordre D9 : (témoin dense si accepté) → SIL si prête → (1) ou (2) selon ADR-006 D5 → (4) tâche auxiliaire (ADR-010).
