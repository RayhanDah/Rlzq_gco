# ADR-011 — Run témoin à récompense dense, HORS SPEC, diagnostic du pipeline

- Statut : **accepté** : GO de l'utilisateur le 2026-11-19 (« GO témoin dense »), points 1 à 4 tels que proposés. Run hors
  spec, diagnostic, jamais promu (CLAUDE.md règle 9 : la ligne principale reste GCO-std). Lancement après GCO-S s0 et un smoke
  test vert sur le commit du run.
- Date : 2026-11-18
- Décideurs : research-director (proposition) ; reward-engineer (implémentation après GO) ; utilisateur (décision)

## Contexte
- Question de l'utilisateur : avec 1B pas, la communauté obtient un bot qui joue avec des récompenses façonnées ; nous, en GCO, ne
  touchons la balle que dans 8,7 à 13,9 % des épisodes kickoff, 81 à 86 % sans touche (`docs/reports/run_4a.md` §4 bis).
- Deux explications concurrentes, que nos données ne séparent pas :
  - **(G) difficulté propre au GCO** : un signal par but, ≈ 4 à 7 buts pour 51 611 pas appris par itération (annexe A de 4a),
    contre un signal à chaque pas en dense. Attendue : S1 « peut ne jamais arriver » sans aide (`GCO_SPEC.md` §6.1) ;
  - **(P) défaut du pipeline** (obs, table d'actions, simulateur RocketGPU et ses écarts R13, PPO et ses réglages) : par exemple, KL
    ≈ 10⁻⁴ et clip < 0,1 % en 4a (annexe A) indiquent des mises à jour minuscules ; le guide officiel RLGym dit d'une entropie
    saine qu'elle « should settle around 2 » (définition exacte non vérifiée), la nôtre reste à 4,24–4,37 sur ln 90 = 4,50.
- Si (P), chaque intervention GCO (≈ 5,5 h chacune) est gaspillée. Un témoin dense, identique à std s0 sauf la récompense, sépare
  (G) de (P) au prix d'un run.
- Ce n'est **pas** GCO-ext : GCO-ext garde la récompense §2.1 (`GCO_SPEC.md` §3, ligne « Récompense »). C'est un run **hors spec**,
  comme l'ablation « scorer-only » déjà prévue (`GCO_SPEC.md` §7.7 ; `src/rewards/config.py:115-118` : `in_spec=False`,
  `ablation_only=True`).

## Options
| Option | Pour | Contre |
|---|---|---|
| **Récompense de départ du guide officiel RLGym, à l'échelle de notre ±1** | recette publique, documentée, la plus utilisée pour un premier bot | pas notre obs, nos réseaux, ni leur normalisation des retours : le témoin teste **notre** pipeline, pas leur recette |
| Exemple `rlgym-ppo` (`example.py`) | public | timeout 10 s, terme démo ; plus éloigné de notre env |
| Récompense de touche pure | cible directe du goulot | critère de verdict circulaire (on récompense ce qu'on mesure) |
| Pas de témoin, audit du code seul | aucun calcul | un audit ne prouve pas qu'un signal dense est appris de bout en bout |
| Témoin court (0,1–0,2B) | ≈ 1 h | aucun repère public de pas jusqu'à la 1re touche : un échec précoce ne prouverait rien |

## Mesures
Aucune mesure du témoin. Sources lues le 2026-11-18 :
- Guide officiel RLGym « Training an agent » (https://rlgym.org/Rocket%20League/training_an_agent/) : `CombinedReward` de
  `InAirReward` 0,002, `SpeedTowardBallReward` 0,01, `VelocityBallToGoalReward` 0,1, `GoalReward` 10 ; terminaison
  `GoalCondition`, troncature no-touch 30 s et 300 s ; `RepeatAction(LookupTableAction(), 8)` ; PPO lot 100 000, minibatch 50 000,
  2 epochs, entropie 0,01, LR 1e-4, réseaux 2048-2048-1024-1024, `standardize_returns=True` ; budget `timestep_limit` 1B. Les
  deux termes de vitesse sont décrits « clamped at zero ». **Aucun jalon publié** (pas jusqu'à la 1re touche ou au 1er but).
- Exemple `rlgym-ppo` (https://github.com/AechPro/rlgym-ppo, `example.py`, dernier commit du fichier 1ddefdf, 2023-09-05) :
  `VelocityPlayerToBallReward` 0,01, `VelocityBallToGoalReward` 0,1, `EventReward(goal ±1, demo 0,1)` × 10, no-touch 10 s.
- Constantes `rlgym` v1.2.0 (`rlgym/utils/common_values.py:12-18`) : `ORANGE_GOAL_BACK` = (0, 6000, 321,3875),
  `BLUE_GOAL_BACK` = (0, −6000, 321,3875), `BALL_MAX_SPEED` = 6000, `CAR_MAX_SPEED` = 2300 ; formule de
  `VelocityBallToGoalReward` dans `ball_goal_rewards.py` (même tag).
- Coût : run de 1B = 5,44 à 5,51 h mesuré (`run_4a.md` §1, débit appris 50,2k à 51,4k pas/s, commits 3e96158 / a76cd14 /
  e289c30, laptop Core Ultra 5 125H + Arc). Pour ce témoin : **estimé, non mesuré** (épisodes plus longs si l'agent touche).
- Repères de touche (`run_4a.md` §4 bis, commit 6db79a4) : θ₀ 6–7 % ; std à 1B 8,7 / 10,3 / 13,9 % ; l'aléatoire adverse 6,1–6,4 %.

## Décision (si l'utilisateur dit GO)

### D1. Récompense exacte (variante `dense-diag`, hors spec)
Pour l'agent i au pas de décision t (état en fin de pas, après les 8 ticks) :
```
r_t(i) = g_t(i)                                   # §2.1 exactement : ±1 par équipe, csc selon le but touché, 0 sinon
       + 0,001  · max(0, v_voiture(i) · û(voiture(i) → balle)) / 2300
       + 0,01   · max(0, v_balle · û(balle → fond du but adverse de i)) / 6000
       + 0,0002 · 1[voiture(i) pas au sol]
```
- Poids = ceux du guide officiel **divisés par 10**, pour garder le but à ±1 : même rapport entre termes, échelle des retours
  comparable à 4a (notre PPO n'a pas de normalisation des retours, ADR-007 D2 ; un ±10 changerait l'échelle des pertes et le clip
  global commun π + V). Façonnage par pas ≤ 0,0112, soit ≤ 2,24 sur l'horizon 1/(1−γ) = 200 pas (calculé).
- Cible du but : `ORANGE_GOAL_BACK` / `BLUE_GOAL_BACK` de `rlgym` v1.2.0 selon l'équipe. Pas de terme démo, pas de touche.
- Tout le reste = std s0 : env (troncatures 20 s / 120 s, `p_kick` 0,2 → 0,8 sur 0,5B), obs, actions, PPO, pool, graine 0,
  budget 1B (mêmes calendriers de LR et d'entropie). **Un seul levier : la récompense.**

### D2. Isolation (jamais promu)
- Variante de récompense `dense-diag` : `in_spec=False`, `ablation_only=True` ; nom de run `diag_dense_s0`, racines `runs/` et
  `checkpoints/` propres. Le hash de récompense diffère : aucun checkpoint ne peut initialiser un run GCO, entrer dans un pool,
  servir d'adversaire d'entraînement, ni être exporté vers RLBot (refus testés). Étiquette « HORS SPEC » dans chaque journal et au
  rapport.
- Après GO seulement : ajout de la variante à `GCO_SPEC.md` §7.7 et exception explicite dans `.claude/rules/rewards.md` (« aucun
  shaping »), comme l'exige §7.8 (ADR = celui-ci). Les tests §7 restent verts pour les variantes in-spec ; la variante a ses tests
  (formule, symétrie bleu/orange, zéro façonnage quand la voiture est immobile au sol et la balle immobile).

### D3. Critère de verdict (pré-enregistré)
Mesure : `scripts/eval.py --a <instantané> --b random --episodes 20000 --n-envs 256 --seed 0 --sprt-elo 100
--sprt-max-decisive 1200` (épisodes kickoff, comme `s0_final20k` de 4a) ; « touche » = champ `touches` (part d'épisodes où A
touche, 6db79a4). Trajectoire descriptive à chaque 0,1B avec 2 048 épisodes.
| Verdict | Condition | Lecture | Suite |
|---|---|---|---|
| **PASS** (pipeline capable) | borne basse du Wilson 95 % de « A touche » ≥ **50 %** **et** SPRT +100 Elo accepte H1 | obs, actions, sim et PPO apprennent le contact quand le signal existe : le goulot est (G) | interventions GCO, SIL en tête (ADR-009 D9) |
| **FAIL** (pipeline suspect) | à 1B, borne haute du Wilson 95 % de « A touche » < **25 %** | même un signal dense ne fait pas toucher : (P) | interventions GCO suspendues ; ADR d'audit (obs, table d'actions, R13, amplitude des mises à jour PPO) |
| **AMBIGU** | sinon | le dense apprend, lentement : soupçon sur les réglages d'optimisation (KL 10⁻⁴, clip < 0,1 % en 4a) | ADR d'audit des réglages PPO (un levier), avant ou en parallèle de la SIL |
- Seuils : 50 % est clément pour un poursuivant de balle face à un aléatoire qui ne touche qu'à 6 % (estimé, non mesuré) ; 25 %
  ≈ 2× le meilleur std à 1B (13,9 %). Diagnostics joints : KL, clip, entropie, normes de gradient du témoin contre 4a s0.
- Un PASS ne valide pas les réglages pour le signal clairsemé ; un FAIL ne localise pas le défaut. Le témoin ne dit rien du GCO
  lui-même.

### D4. Coût et déroulé
- Session 1 : `scripts/run_4a.py --variant diag_dense --seed 0 --max-hours 3` (arrêt propre à une frontière d'itération,
  `scripts/run_4a.py:6`, au-delà de 0,5B à 51k pas/s, estimé), puis l'éval D3 de l'instantané 0,5B (itération 9 688 ; ≈ 2,5 min pour
  20 000 épisodes en 4a, mesuré ; plus long si les épisodes s'allongent). PASS à 0,5B ⇒ fin du témoin (≈ 3 h). Sinon, session 2
  jusqu'à 1B (≈ 2,7 h de plus, estimé) et verdict à 1B. Deux regards (0,5B, 1B) : multiplicité assumée pour un contrôle binaire.
- Kill criteria : K2 à K6 (ADR-005 D6, amendés par ADR-006) ; aucun lien avec S1 ou G3.
- **Place dans le calendrier** : juste après GCO-S s0 (en cours), **avant** la SIL et (1)/(2) : il qualifie tous les runs GCO
  suivants pour ≈ 3 à 5,6 h. Si son code n'est pas prêt, la SIL ou (1)/(2) passent d'abord (ADR-009 D9).

## Conséquences
- Plus facile : répond à la question de l'utilisateur par une mesure, pas une opinion ; protège ≥ 3 runs GCO de 5,5 h.
- Plus difficile : introduit du façonnage dans `src/rewards/` (isolé, testé, jamais promu) ; un PASS peut être mal lu comme
  « le dense marche, abandonnons le GCO » : hors sujet, la ligne principale reste GCO-std (G1 H1, règle 9).
- Révision si : verdict AMBIGU (audit des réglages) ; ou si un jalon public (pas jusqu'à la 1re touche avec cette recette) est
  trouvé, pour recalibrer les seuils avant le run.

## Points à décider par l'utilisateur
1. GO ou non pour un run hors spec, étiqueté, jamais promu (règle 9 de CLAUDE.md).
2. Récompense D1 (guide officiel RLGym ÷ 10, but ±1 de §2.1) ; un seul levier contre std s0.
3. Verdict D3 (PASS ≥ 50 % de touche + SPRT, FAIL < 25 % à 1B) ; arrêt à 0,5B si PASS.
4. Place : après GCO-S s0, avant la SIL ; coût ≈ 3 à 5,6 h (estimé).

## Implémentation et vérifications (2026-11-19)
- Code : variante `dense-diag` (`src/rewards/dense.py`), `configs/diag_dense.yaml` (seuls `name` et `reward.variant`
  diffèrent de `run_4a_std.yaml`, testé), `scripts/run_4a.py --variant diag_dense` (run `diag_dense_s<seed>`), refus de
  checkpoint hors spec (`training.checkpoint.require_in_spec`, `purpose` de `load_checkpoint_policy`). Commit ff81e9b,
  revue `code-reviewer` APPROVE (aucun majeur) ; fusion dans main 015c65d avec la SIL (ADR-009, désactivée par défaut).
- Hashs de récompense inchangés pour les variantes in-spec (GCO-S 2b6485236c99aa29, GCO-std d531d5919a4d9181, GCO-ext
  fb907ee1d40cc464, scorer-only b8051d9a95dbdec9) ; `dense-diag` 275f3f883e786a28.
- Tests sur 015c65d : ruff, format, mypy verts ; 679 tests unitaires ; 144 d'intégration avec RocketGPU (2 xfail connus, R13).
- Smoke tests **PASS** sur 015c65d, XPU, laptop Core Ultra 5 125H, 2026-11-19 : `scripts/train.py configs/smoke.yaml --smoke`
  (`runs/smoke_015c65d/`) et la même config avec `reward.variant: dense-diag` (`runs/smoke_dense_015c65d/`, récompense
  moyenne par pas 3,8e-4 > 0 : le façonnage est actif) ; 0 NaN, reprise exacte. Essai du lanceur (`--max-iterations 1`,
  `runs/trial_dense_015c65d/`) : `run_meta` porte `HORS SPEC`, `reward_variant: dense-diag`, éval `run_s1v2_eval.yaml`.
- Errata D3 : l'éval en cours de run est celle de `run_s1v2_eval.yaml` (3 072 épisodes contre l'aléatoire et 3 072 contre θ₀
  tous les 0,1B, touches comprises) au lieu de 2 048 : sur-ensemble de la trajectoire descriptive prévue. Le verdict reste
  celui de D3, hors ligne, 20 000 épisodes. `s1_reached` et K7 du hook sont sans objet pour ce run.
- À retenir pour la Phase 5/6 : un futur pool d'adversaires externes ou un export RLBot devra appeler
  `load_checkpoint_policy(..., purpose="training_opponent" | "export")` ; la garde n'est pas automatique (revue, mineur).
- Commande, session 1 (sur GO de l'utilisateur) : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant diag_dense
  --seed 0 --max-hours 3`.
- **Verdict D3 (2026-11-21) : AMBIGU.** Commande D3 (`scripts/eval.py --a <instantané> --b random --episodes 20000 --n-envs 256
  --seed 0 --sprt-elo 100 --sprt-max-decisive 1200`), commits 8fdf740 (0,5B) et 6ae65d6 (1B), code identique, `.venv` CPU, laptop Core Ultra 5 125H, `git_dirty` faux.
  - 0,5B (it. 9 688) : A touche **23,0 %** [Wilson 95 % 22,4 ; 23,6], aucune touche 72,9 %, 593 V / 19 101 N / 306 D (4,5 % de
    décisives), SPRT H1 ; pas de PASS à 0,5B, donc session 2 jusqu'à 1B (D4)
    (`runs/diag_dense_offline_eval/dense_s0_final20k_it09688.json`).
  - 1B (it. 19 376) : A touche **50,44 %** [**49,75** ; 51,13], aucune touche 46,97 %, 1 948 V / 17 775 N / 277 D (11,1 % de
    décisives, 9,7 % de victoires par épisode), Elo décisif +339 [+317 ; +362], SPRT H1 après 116 épisodes
    (`…/dense_s0_final20k_it19376.json`). Repère std s0 à 1B (§4 bis de 4a) : 13,9 %, 81,0 % sans touche, Elo +203.
  - Application mécanique : PASS non atteint (borne basse 49,75 % < 50 %, bien que l'estimation ponctuelle dépasse 50 %) ;
    FAIL exclu (borne haute 51,13 % ≫ 25 %). Le seuil n'est pas relu après coup.
  - Lecture : obs, actions, simulateur et PPO apprennent le contact quand le signal existe (×3,6 la touche de std s0 à 1B) ; une
    panne franche du pipeline (P) est écartée, mais l'apprentissage est lent (23 % à 0,5B, 1B pour atteindre ≈ 50 %).
    Diagnostics (`runs/diag_dense_s0/metrics.jsonl`, valeurs d'itération) : KL 3·10⁻⁴ à 1,4·10⁻³ et clip frac ≤ 0,9 %, soit un clip
    jamais actif ; entropie 4,45 (0,1B) → 4,28 (0,5B) → 3,59 (1B, 80 % de ln 90), contre 4,33 pour std s0 à 1B. L'entropie baisse
    surtout en seconde moitié, quand LR et ent_coef décroissent ensemble ; les journaux ne séparent pas les deux causes.
  - Suite (D3, ligne AMBIGU) : **ADR-012** (proposé), audit d'un levier (LR ×3), **après** le run SIL (« quoi qu'il arrive »,
    décision de l'utilisateur du 2026-11-21, ADR-009). Run « repris » : au moins une perte du GPU en session 2 (it. 10 689,
    `runs/diag_dense_s0/segments.jsonl` l. 1), reprise automatique.
  - Trajectoire descriptive (2 048 épisodes, graine 0, même commande, `…/dense_s0_traj2048_it*.json`) : A touche 7,2 % (0,1B),
    10,8, 12,8, 18,1, 23,9, 28,8, 27,7, 39,4, 41,1, 50,5 % (1B) ; SPRT H1 à partir de 0,4B. Progression sans plateau à 1B.
