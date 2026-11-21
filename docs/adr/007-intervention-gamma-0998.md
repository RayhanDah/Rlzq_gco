# ADR-007 — Intervention (2) : γ 0,995 → 0,998 (GCO-std, graine 0, 1B)

- Statut : **accepté pour exécution par délégation** : l'utilisateur a validé le 2026-11-21 l'ordre « analyse post hoc SIL →
  ADR-012 → ADR-007 → ADR-010 → variante SIL » et délégué la gestion des runs (« je te laisse gérer le projet et les runs ») ;
  les 3 points en fin d'ADR n'ont pas été relus un par un par l'utilisateur. Condition ADR-006 D5 levée : borne haute du
  Wilson des victoires par épisode de GCO-S s0 4,60 % ≥ 1,60 % (`docs/reports/gcos_s0.md`). Lancement seulement si la
  phase A d'ADR-012 n'est pas SUCCÈS (sinon phase B d'abord, sur GO, ADR-012 D8). (Proposé le 2026-11-18, conditionnel.)
- Date : 2026-11-18
- Décideurs : rl-algo-lead (proposition) ; research-director ; utilisateur (validation, lancement).

## Contexte
- `docs/GCO_SPEC.md` §8 classe « horizon γ 0,995 → 0,998 » deuxième intervention, après le calendrier d'états initiaux.
  ADR-006 D5 l'a pré-enregistrée comme suite de GCO-S quand rien ne montre que les états génériques aident.
- Run 4a (`docs/reports/run_4a.md` §4 bis, §8) : 81 à 86 % des épisodes d'éval finissent sans touche ; durée moyenne
  ≈ 20,5 s, soit la troncature « 20 s sans touche » ; à l'entraînement, buts / fins d'épisode ≈ 4 à 5 %. Le critique est
  bon (EV 0,96 dès 0,4B, §8) ; la politique bouge peu (entropie 94 à 97 % de ln 90, KL ≈ 10⁻⁴).
- Avec γ = 0,995, un but à 20 s ne pèse plus que 0,22 de sa valeur (calcul ci-dessous) : la plupart des buts arrivent loin
  de la décision qui les prépare, en particulier depuis le kickoff, seule distribution de l'éval.

## Options
| Option | Pour | Contre |
|---|---|---|
| **γ = 0,998 constant dès l'itération 0** | un seul champ (`ppo.gamma`) ; comparable à std s0 par la commande hors ligne d'ADR-006 ; aucun code | Five montre le gain d'un horizon long sur un agent **déjà compétent** (§1.2) ; ici dès le départ |
| Montée linéaire 0,995 → 0,998 « quand l'agent marque » (`docs/01_papers.md` §1.2, §2) | conforme à la proposition v0 | déclencheur à définir, calendrier à coder (`PPOConfig` : γ constant) ; deux leviers mêlés |
| Protocole Five Fig. 6 : reprendre le checkpoint 1B de std s0 avec γ = 0,998 | question propre « l'horizon aide-t-il un agent entraîné » | ne répond pas au critère D6 (ii) (S1 avant 0,3B, depuis zéro) ; reprise d'un checkpoint avec une config changée non prévue |
| γ = 0,997 | changement plus doux | effet attendu plus faible avec une seule graine ; §8 cite 0,998 |

## Mesures
Aucune mesure d'entraînement : décision réversible, fondée sur `docs/01_papers.md` §1.2 et §2 (sources Five Eq. 3, §4.5,
Fig. 6 ; Lucy-SKG γ ≈ 0,995 ; pulsar2 γ = 0,9969 et goal critic 0,9994), et sur les journaux du run 4a
(`docs/reports/run_4a.md` §4 bis, §8). Calculs ci-dessous (Python, exacts aux arrondis près) :

| | γ = 0,995 (actuel) | γ = 0,998 |
|---|---:|---:|
| Pas de politique : tick_skip 8 à 120 Hz = 15 Hz, Δt = 1/15 s (`GCO_SPEC.md` §2.1) | | |
| Horizon 1/(1−γ) | 200 pas = **13,3 s** | 500 pas = **33,3 s** |
| Demi-vie ln 2 / (−ln γ) | 138 pas = 9,2 s | 346 pas = 23,1 s |
| Poids d'un but à 20 s (γ³⁰⁰), troncature no-touch | 0,222 | 0,548 |
| Poids d'un but à 120 s (γ¹⁸⁰⁰), troncature de durée | 1,2·10⁻⁴ | 0,027 |
| Fenêtre de la trace GAE 1/(1−γλ), λ = 0,95 | 18,3 pas = 1,22 s | 19,3 pas = 1,28 s |

Source de 0,998 : `docs/01_papers.md` §1.2 (« γ₀ = 0,995, puis montée linéaire vers 0,997–0,998 ») et §2 (tableau v0) ;
la valeur exacte 0,998 est une **proposition dérivée** (demi-vie 23 s), pas une valeur publiée pour Rocket League en GCO.

## Décision

### D1. Une seule modification
`configs/int_gamma998.yaml` = `configs/run_4a_std.yaml` avec **`ppo.gamma: 0.998`** (hors `name` et commentaires ; vérifié
par `tests/unit/test_run_4a_interventions.py`). γ constant pendant tout le run, dès l'itération 0 (option 1).
GCO-std, graine 0, 1B appris, `p_kick` 0,2 → 0,8 sur 0,5B (ADR-005 D4 inchangé), pool, LR, entropie, kill criteria
identiques. Aucun signal ajouté ; C1–C4 non sollicités (γ ne modifie pas la récompense, `01_papers.md` §1.2 « GCO :
compatible »).

### D2. Ce que γ change, et ce qu'il ne change pas (lu dans `src/training/gae.py`, `src/training/ppo.py`)
- **Change** : δ_t = r_t + γ·V_next[t] − V[t] (`gae.py:63`) et la trace γλ·c_t (`gae.py:67`). Cible du critique
  `ret = A + V` (`gae.py:69`) : V apprend E[γ^τ·(±1)] avec τ = délai jusqu'au but ; les valeurs des états loin d'un but
  grossissent en module (×2,5 à 20 s), toujours dans [−1, 1] (récompense ±1 unique, `GCO_SPEC.md` §2.1).
- **λ inchangé** (0,95) : un seul levier à la fois (`.claude/rules/training.md`). La fenêtre de crédit de la trace passe de
  1,22 à 1,28 s seulement ; l'allongement de l'horizon passe **par le critique** (bootstrap), pas par la trace.
- **Pas de normalisation des valeurs** à adapter : le critique est un MLP linéaire en sortie (`src/models/value.py`), sans
  PopArt ni normalisation des retours ; les avantages sont normalisés par minibatch (`ppo.py:276`), donc l'échelle du
  gradient de politique ne dépend pas de γ.
- **Troncatures** : même traitement (`gae.py` docstring, `bootstrap_values`) : no-touch 20 s et 120 s bootstrappés sur
  V(`final_obs`), but terminal à 0. Le code ne change pas ; le poids γ·V(s′) passe de 0,995 à 0,998, et surtout V(s′)
  porte un horizon plus long. Comme ≈ 80 % des épisodes finissent par la troncature no-touch, la qualité de V(s′) y compte
  plus qu'avant.
- **Inchangés** : récompense, observation, éval (indépendante de γ), `p_kick_at` (version de calendrier inchangée), hash de
  récompense. Le hash de trajectoire change (nouveau run, pas de reprise d'un checkpoint std).

### D3. Exécution (après GO utilisateur seulement ; délégation du 2026-11-21, voir Statut)
- Prérequis (CLAUDE.md règle 5) : smoke test vert **sur le commit du run** (`scripts/train.py configs/smoke.yaml --smoke`),
  ce commit portant le Statut ci-dessus ; arbre propre (le lanceur refuse un arbre sale, `scripts/run_4a.py`). Contrôle après
  démarrage : SHA-256 des poids de θ₀ égal à std s0 (méthode de `tests/unit/test_run_4a_interventions.py:414`).
- Diagnostic suivi (pas un critère d'arrêt) : ‖g‖ avant clip (π et V partagent un Adam et un clip global 0,5,
  `src/training/ppo.py:334-341` ; std ≤ 0,093).
- Commande : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant int_gamma998 --seed 0 --max-hours 0`
  (run `int_gamma998_s0` ; éval en cours de run par défaut `configs/run_s1v2_eval.yaml`, protocole ADR-006 D2, D3 ; ajouter
  `--eval-config run_s1v2_eval.yaml` explicitement ne change rien). Pas de `--std-s1-observed` : c'est un GCO-std.
  Mêmes garanties que `--variant std` : sessions enfants, reprise après `DEVICE_LOST`, refus après STOP, arbre propre.
- Coût : ≈ 5,5 h (durée std mesurée, transposée ; estimé, non mesuré).

### D4. Comparaison à std s0 et critère de succès
- Comparaison **uniquement** par la commande hors ligne d'ADR-006 D4, identique à `runs/run_4a_offline_eval/s0_*` : pour
  chaque instantané à 0,1B (`checkpoints/int_gamma998_s0/eval_snapshots/ckpt_*_it<it>.pt`,
  aux mêmes itérations que std s0 : 0, 1 938, 3 876, 5 813, 7 751, 9 688, 11 626, 13 564, 15 501,
  17 439, 19 376)
  `scripts/eval.py --a <instantané> --b random --episodes 2048 --n-envs 256 --seed 0 --sprt-elo 100 --sprt-max-decisive 1200
  --out runs/int_gamma998_offline_eval/s0_b<0.k>_it<it>.json`, puis à 1B la même commande avec `--episodes 20000`
  (`s0_final20k_it19376.json`). Les évals en cours de run (v2) sont décrites au rapport, jamais comparées entre runs.
- **Succès (ADR-006 D6)** si l'un des deux :
  (i) à 1B, 20 000 épisodes contre l'aléatoire `--seed 0` : borne basse du Wilson bilatéral 95 % de la part de décisives
  ≥ **6,28 %** **et** SPRT +100 Elo accepte H1 sur la même éval ;
  (ii) 1re paire de points consécutifs avec H1 sur les évals hors ligne à 2 048 épisodes `--seed 0` au plus tard à **0,2B**
  (paire complète ≤ 0,3B).
  Référence std s0 : 3,14 % de décisives à 1B, 1re paire à 0,5B (ni (i) ni (ii)). Références descriptives seulement (jamais
  comparées pour ce critère) : GCO-S s0 (`docs/reports/gcos_s0.md`) et SIL s0 (`docs/reports/int_sil_s0.md`, 4,33 % à 1B).
- **Neutre** sinon : γ = 0,995 conservé ; suite selon l'ordre validé par l'utilisateur le 2026-11-21 : ADR-010 (tâche
  auxiliaire), puis variante SIL éventuelle, un ADR chacune. (Rédaction initiale : ADR-008 ou self-imitation ; la SIL a été
  faite ; ADR-008 ne s'applique pas, GCO-S s0 n'étant pas « S < std », ADR-006 D5.)

## Conséquences
- Plus facile : un seul champ, aucun code d'entraînement touché, run comparable à std s0 par une commande identique.
- **Risques** :
  1. **variance** des cibles du critique : l'horizon de 500 pas dépasse la longueur typique d'un épisode (≈ 300 pas) ; la
     cible dépend davantage de V(s′) aux troncatures, donc des erreurs du critique. Surveiller EV (K3 après S1) et la perte
     de valeur ; la normalisation par minibatch protège l'échelle de la politique, pas sa qualité ;
  2. Five mesure le gain d'un horizon long sur un agent déjà compétent ; dès l'itération 0, l'effet peut être nul ou négatif
     (crédit dilué sur un agent qui ne touche presque pas la balle) ;
  3. une graine : la variance entre graines std (victoires à 1B de 1,60 à 2,40 %, ×1,5) domine ; un « neutre » ne réfute pas
     γ = 0,998 ;
  4. KL et clip : un critique plus lent à converger peut changer le rythme d'apprentissage ; K4 inchangé (KL > 0,05).
- Révision si : EV < 0,5 durablement après 0,2B (diagnostic, pas un critère d'arrêt), ou succès ambigu (un seul volet de
  (i)) ; on passerait alors à l'option « montée 0,995 → 0,998 » ou au protocole Five Fig. 6 (nouvel ADR).

## Points soumis à l'utilisateur (couverts par la délégation du 2026-11-21, non relus individuellement)
1. Application conditionnelle : seulement si GCO-S s0 n'est pas « S < std » (ADR-006 D5).
2. γ = 0,998 constant dès l'itération 0 (et non montée progressive ni reprise d'un checkpoint) ; λ = 0,95 inchangé.
3. Commande de lancement D3, smoke test sur le commit du run d'abord.
