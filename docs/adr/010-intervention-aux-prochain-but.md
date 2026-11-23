# ADR-010 — Intervention (4) : tâche auxiliaire « prochain but » (GCO-std, graine 0, 1B)

- Statut : **accepté par l'utilisateur le 2026-11-23 (option float32 à H = 100)**, amendement de D3 et D8-5. Restent
  **GO de lancement donné le même jour** (choix « ADR-010 float32 H=100 », après le constat que l'ADR était prêt à lancer) :
  l'ADR est exécuté tel qu'écrit. Les points 1 à 4 de « Points à valider » sont couverts par ce GO, mais n'ont pas été relus un
  par un. λ_aux = 0,1 fixé par les bancs D5.
- Date : 2026-11-18 (esquisse) ; révisé le 2026-11-21 (leçons du run SIL, RAM, C2, débit, préalables) ; amendé le
  2026-11-23 (D3, D8-5 : float32 à H = 100)
- Décideurs : research-director (proposition) ; rl-algo-lead (implémentation, `configs/int_aux.yaml`, `--variant int_aux`,
  désactivée par défaut) ; utilisateur (validation, lancement)

## Contexte
- Intervention (4) de `docs/GCO_SPEC.md` §8 et d'ADR-005 D6. Autorisée en GCO-std (`GCO_SPEC.md:75`, « Tâches auxiliaires
  (prédire le prochain but) » : interdites en GCO-S, « **autorisées** (perte de représentation, pas récompense) »).
- Le critique apprend déjà E[γ^τ·(±1)] (EV 0,96 dès 0,4B, `docs/reports/run_4a.md` §8), mais politique et critique sont **séparés**
  (`src/models/policy.py:17` : π 3×256 ; `src/models/value.py:15` : V 3×512). Rien ne force la représentation de la **politique** à
  encoder ce qui mène à un but. Hypothèse : c'est une partie de la lenteur observée (politique quasi uniforme, 81 à 86 % d'épisodes
  sans touche).
- Run SIL `int_sil_s0` (ADR-009, `docs/reports/int_sil_s0.md`) : NEUTRE au critère ADR-006 D6 (4,33 % [4,06 ; 4,63] à 1B), mais
  bosse à 0,7B (8,79 % de décisives, touche 41,7 %) puis rechute pendant l'annealing de c_sil ; persistance D4 déclenchée (« gain
  dépendant de la SIL », §5). Débit appris 31 à 34k/s contre 51k/s sans SIL, boucle limitée par le learner (§6). Trois leçons pour
  cet ADR : la RAM impose la fenêtre de 150 pas (D2) ; l'annealing à 0 dans un run de 1B confond « effet » et « persistance » (D6) ;
  le coût d'une perte de plus ne se déduit pas du seul ratio d'échantillons (D7).
- Sources : Lucy-SKG (arXiv 2305.15801, lu via ar5iv le 2026-11-18) : tâche « reward prediction » à 3 classes (positif, négatif,
  ≈ 0) branchée sur les couches de l'acteur (§4.2, §4.5) ; récompense moyenne en plateau vers 100M contre 200M sans (§5.1, Fig. 3),
  **sur une récompense dense**, poids non publié. UNREAL (Jaderberg et al., arXiv 1611.05397, lu via ar5iv le 2026-11-18) :
  entropie croisée à 3 classes (§3.2), rejeu rééquilibré P(r ≠ 0) = 0,5 (§3.2–3.3), λ_RP = 1 (Annexe B).

## Options
| Option | Pour | Contre |
|---|---|---|
| **Classe du prochain but à horizon H (+1 / −1 / aucun), tête sur le tronc de π** | cible tirée du seul ±1 ; analogue GCO de Lucy et UNREAL | classes très déséquilibrées ; étiquette connue seulement H pas plus tard ; fenêtre en RAM |
| Récompense immédiate (H = 1, Lucy/UNREAL à la lettre) | fidèle aux sources ; aucune fenêtre | en GCO, r ≠ 0 au seul pas du but : ≈ 1 pas sur 300, signal quasi nul |
| Tête sur le critique | aucun effet sur π | redondant avec V, déjà bon (EV 0,96) |
| Régression du retour discounté sur le tronc de π | continue | duplique V ; échelle à régler |
| λ_aux annealé à 0 avant la fin du run (calendrier de la SIL) | lecture littérale de la clause d'annealing de C2 | le critère à 1B mesurerait la persistance après retrait, pas l'effet (leçon SIL) ; la tête n'est pas exportée, la garder ne coûte rien à l'inférence |

## Mesures
Aucune mesure d'entraînement avec la tâche auxiliaire : décision réversible, fondée sur les sources ci-dessus, les mesures de la SIL
et les calculs ci-dessous (Python, exacts aux arrondis ; **non mesurés** sauf mention).
- **Dimensions mesurées dans le code** : 7 373 agents appris par pas (4 096 arènes × 2, pool 20 % exclu), `tests/unit/test_sil.py:547` ;
  fenêtre SIL à 150 pas = **243 367 984 octets** (= 7 373 × 150 × (109 × 2 + 2) + 7 373 × 8 ; obs float16 et action),
  `tests/unit/test_sil.py:548` ; buffer SIL 58 250 000 octets (l. 549). Le « ≈ 0,54 Go » d'ADR-009 (Mesures) vaut 8 192 agents ×
  **300** pas × 218 octets : il ne correspondait pas à H = 200 et surestimait le nombre d'agents.
- **RAM mesurée (SIL)** : banc 1, fenêtre 300 (état SIL ≈ 545 Mo) : minimum disponible 154 Mo, à l'écriture d'un checkpoint de 771 Mo,
  chute transitoire médiane 2,86 Go ; banc 2, fenêtre 150 (état ≈ 302 Mo) : minimum 2 354 Mo, checkpoint 528 Mo, chute médiane
  1,62 Go (ADR-009, « Préalables D7 mesurés », bancs du 2026-11-21, commits 6ae65d6 et aedf0ca). Règle D7 d'ADR-009 : minimum
  < 1,5 Go ⇒ fenêtre de 150 pas. Run SIL complet à 150 pas : KS1 jamais déclenché, minimum 143 Mo **après** la libération du buffer
  (cause externe probable, non établie ; `int_sil_s0.md` §6).
- **Débit mesuré** : std 50,2 à 51,4k appris/s (`run_4a.md` §5) ; fil learner occupé **34,7 %** au banc M5 sans critique d'exploration
  (`docs/adr/001-infra.md:287`, pool 7 → 49, pas mesuré en vrai run) ; SIL 31,2 et 32,6k/s aux bancs, learner occupé 96,9 %
  (ADR-009 l. 239-240), pour un ratio d'échantillons de 0,24 seulement.
- **Calculs de cet ADR** :
  - fenêtre auxiliaire (obs float16 seules, 218 octets par pas, + compteur int64 par agent) : H = 150 → **241,2 Mo** ;
    H = 200 → 321,5 Mo ; rejeu 250 000 × 223 octets (obs 218, étiquette 1, itération 4) = **55,8 Mo**. Total : **297 Mo** à H = 150
    (≤ 301,6 Mo, état SIL de la configuration du banc 2, octets des tests l. 548-549), 377 Mo à H = 200 (+76 Mo) ;
  - pente de la chute transitoire au checkpoint entre les deux bancs SIL : (2,86 − 1,62) / (0,545 − 0,302) ≈ 5 Go par Go d'état
    (deux points, deux jours, disponibilité de base différente : **non causal, estimé**) ; à H = 200, ≈ +0,4 Go de chute transitoire ;
  - H = 150 pas = 10 s à 15 Hz ; γ¹⁵⁰ = 0,47 à γ = 0,995 (demi-vie 138 pas, ADR-007 Mesures) ; 1/(1−γ) = 200 pas ;
  - part d'étiquettes non nulles, **estimé** : buts / fins d'épisode ≈ 4 à 8 % (annexe A de 4a ; 4,15 à 5,31 % à `p_kick` 0,8,
    `run_4a.md:247`) × au plus 150 pas étiquetés par but sur ≈ 305 pas par épisode : ≈ 2 à 8 % des pas ; part masquée (les 150
    derniers pas des épisodes tronqués, ≈ 92 à 95 % des épisodes) : ≈ 45 % des pas (estimé) ;
  - coût de calcul : π 182 016 MAC par échantillon, dont tronc 158 976 ; V 580 608 ; tête 768. Par minibatch PPO (51 611 / 3
    = 17 204 échantillons, `src/training/ppo.py:279`) : 3 × 762 624 × 17 204 ≈ 3,9·10¹⁰ MAC ; lot auxiliaire : 3 × 159 744 ×
    8 192 ≈ 3,9·10⁹ MAC (**+10 %** par minibatch, ≈ +9 % des FLOP du learner par itération avec le calcul des avantages).

## Décision

### D1. Cible (C1 : seuls le flux ±1 et les drapeaux de l'environnement)
- Pour l'agent appris i au pas t : y_t = **+1** si l'équipe de i marque, **−1** si elle encaisse (csc compris), **0** si aucun but,
  dans les **H = 150** pas suivants (t inclus). Le signe est celui de r(i) au pas du but (`GCO_SPEC.md` §2.1) : aucune autre lecture.
- **Masquage** : un pas dont la fenêtre [t, t + H) est coupée par une **troncature** (no-touch 20 s, 120 s) sans but n'est jamais
  étiqueté (futur inconnu ; une troncature n'est pas une issue, `GCO_SPEC.md` §2.2). Un pas sort de la fenêtre avec y = 0 dès que
  H pas sont passés sans fin d'épisode. Biais assumé : la classe 0 vient surtout du début des épisodes (journalisé, `aux/frac_masked`).
- Agents admis : ceux de `learn_mask` (les deux camps des arènes en self-play, le seul camp appris des arènes contre le passé).
  L'origine d'épisode (kickoff / générique) n'est ni stockée ni lue.

### D2. Horizon H = 150 (révisé ; l'esquisse prenait 200)
- **RAM** : la seule configuration mesurée qui tient la règle D7 d'ADR-009 (minimum ≥ 1,5 Go) a un état de 302 Mo (fenêtre 150 et
  buffer SIL). À H = 150, l'état auxiliaire fait 297 Mo, **sous** cette configuration ; à H = 200, 377 Mo n'ont jamais été mesurés, et
  la chute au checkpoint croît ≈ 5 fois plus vite que l'état (estimé). H = 200 demanderait un banc de plus sans gain démontré.
- **Science** : H n'a pas de source (1/(1−γ) était un choix dérivé). Les sources prennent H = 1 : un H plus court s'en rapproche. H = 150
  ≈ demi-vie de γ = 0,995 (138 pas) : la classe ± désigne les états où le but pèse encore ≥ 0,47 de sa valeur.
- Repli pré-enregistré : si le banc D5 mesure un minimum < 1,5 Go, H = 100 et nouveau banc ; si le minimum reste < 1,5 Go, pas de GO.

### D3. Tête, perte, rejeu
- **Tête** : linéaire 256 → 3 sur la sortie de la dernière couche cachée de π (`policy.net[:-1]`, après ReLU) ; module **séparé**,
  **hors** des poids de politique exportés (`src/models/policy.py` inchangé ; éval `src/eval/match.py` et RLBot inchangés). Le
  critique ne la voit pas. Initialisation orthogonale gain 0,01 (convention de la tête de π, `policy.py:19`), tirée d'un générateur
  propre : θ₀ de π identique à std s0 (SHA-256 7fab4c74…, test).
- **Perte** : entropie croisée à 3 classes (UNREAL §3.2) ; à **chaque minibatch PPO** (6 par itération), L = L_PPO + λ_aux·L_aux,
  gradients sommés **avant** le clip global 0,5 (la tête est incluse dans le clip), **même Adam**, même LR que PPO (schéma joint
  d'ADR-009 D3). Aucun pas d'optimiseur ajouté.
- **Rejeu** : deux compartiments FIFO de **125 000** transitions (y = 0 et y ≠ 0), obs float16. Lot de **8 192** = 4 096 tirés
  uniformément dans chaque compartiment (rééquilibrage 50 / 50, UNREAL §3.2–3.3). Pas de lot tant que le compartiment y ≠ 0 a moins
  de 4 096 transitions (journalisé). Un FIFO unique de 250 000 serait renouvelé en ≈ 5 itérations par la classe 0 (≈ 25k pas
  étiquetés 0 par itération, estimé) et évincerait les rares ± ; le compartiment ± couvre ≈ 30 à 120 itérations (estimé).
- Obs float16 si la parité D8-5 passe ; sinon float32 avec H/2 (même budget d'octets : H = 75 à H = 150, **H = 50 depuis le
  repli D2 du 2026-11-22**).
- **Amendement du 2026-11-23 (validé par l'utilisateur)** : la parité D8-5 ayant échoué (0,0137 > 0,01), obs **float32 à
  H = 100**, au lieu de H/2 = 50 en float32. Le « même budget d'octets » est abandonné : le budget est désormais celui que
  mesure le banc RAM float32 à H = 100 (minimum disponible 1 884 Mo ≥ 1,5 Go, D2). `store_float16: false` dans les trois
  configs (tailles et preuves en fin de D8).

### D4. Garde-fous de calcul (pour tenir le coût de D7)
- Aucune synchronisation hôte par minibatch : indices des 6 lots tirés en début d'itération (générateur CPU propre), une copie
  hôte → appareil par itération (≈ 6 × 8 192 × 218 octets ≈ 10,7 Mo) ; statistiques accumulées sur l'appareil et lues une fois
  par itération (règle d'ADR-001 M4, `src/training/ppo.py:351`).
- Étiquetage vectorisé (numpy) au début de `update`, comme la SIL (`ppo.py:281-296`) ; aucun appel par agent en Python.

### D5. Poids λ_aux, fixé par mesure (pas deviné)
- UNREAL prend 1 avec d'autres échelles de perte, Lucy ne le publie pas, et nos normes de gradient sont minuscules (0,03 à 0,09,
  annexe A de 4a). Indicateur : **grad_share** = ‖g_aux‖ / (‖g_aux‖ + ‖g_PPO‖), normes avant clip sur les **paramètres du tronc de π**
  (les seuls que la tête touche), moyenne par itération.
- **Banc** (≤ 30 min chacun, buffer plein, pool plein, D8) : λ = 0,1 d'abord ; si la médiane de grad_share sur les 10 dernières
  minutes est > 0,5, λ ← 0,03 puis 0,01 (÷ √10, règle d'ADR-009 D5). On retient le plus grand λ dont la médiane est ≤ 0,5
  (seuil de C4). Si 0,01 dépasse encore 0,5 : pas de GO, révision. Si la médiane est < 0,05 à λ = 0,1 : bancs à 0,3 puis 1 (valeur
  d'UNREAL, plafond), même règle. Valeur retenue consignée ici avant le GO.
- **λ constant sur tout le run de 1B** (D6). Diagnostic secondaire : **disp_share** sur le tronc (moments fantômes d'Adam, méthode
  d'ADR-009 D5, `src/training/sil.py:383` `AdamView`), au rapport, jamais critère : au banc SIL, la part de déplacement valait ≈ 1,8 ×
  la part de gradient (0,103 contre 0,058, banc 2).

### D6. Conformité C1–C4 (`GCO_SPEC.md` §3) et statut de C2
- **C1** : la cible n'utilise que r(i) ∈ {−1, 0, +1}, `terminated`, `truncated` et `learn_mask` ; aucune notion du jeu.
- **C2, tranché** : C1–C4 valent « pour tout signal non-récompense en GCO-std » (`GCO_SPEC.md:80`) : la tâche auxiliaire y est
  soumise. C2 exige un « **objectif final inchangé** » (`GCO_SPEC.md:84`) ; ses deux formes (potentiel appris, coefficient annealé)
  visent les signaux qui modifient ce qui est optimisé : avantages (PBRS, composition critic), bonus intrinsèques (`GCO_SPEC.md:73`,
  « coefficient → 0 avant la fin », clause **absente** de la ligne des tâches auxiliaires, `GCO_SPEC.md:75`), ou perte sur la sortie
  de π (SIL : −log π·(R − V)₊, d'où l'annealing d'ADR-009 D4 et le classement « GCO-ext » d'une SIL sans libération). La perte
  auxiliaire ne touche **ni** la récompense, **ni** les retours, **ni** les avantages, **ni** la couche de sortie de π, **ni** V
  (gradient nul, test D8-3) : la surrogate PPO et la cible du critique sont inchangées. Le seul écart est le partage de capacité du
  tronc, une régularisation de même nature que l'entropie, autorisée sans annealing à 0 (`GCO_SPEC.md:78`) ; C2 accepte une
  invariance inexacte à condition qu'elle soit **loggée** (clamp ±3σ, `GCO_SPEC.md:85`) : c'est le rôle de grad_share.
  **Conclusion : λ constant est conforme à C2 dans ce run et dans la ligne principale.** C'est une interprétation de la spec, pas une
  modification : point 3 à valider. Repli si l'utilisateur retient la lecture littérale : calendrier de la SIL (λ plein jusqu'à 0,5B,
  linéaire jusqu'à 0 à 0,9B, puis 0) et diagnostic de persistance d'ADR-009 D4, par amendement avant le run.
- **C3** : `aux.enabled` (défaut `false`) ; section absente ou désactivée ⇒ trajectoire bit à bit identique à std et hashs de config
  inchangés (tests D8-1, D8-2). Ablation : std s0 et GCO-S s0 existent ; aucun run supplémentaire.
- **C4** : journal `aux/*` à chaque itération (D9) ; alerte si grad_share > 0,5 pendant 50 itérations consécutives.

### D7. Coût en débit (estimé, non mesuré ; mesuré par le banc D5)
- +≈ 9 % de FLOP learner (calcul ci-dessus). Le fil learner de std étant occupé 34,7 % (banc M5), il ne sature qu'au-delà d'un
  temps d'apprentissage par itération multiplié par ≈ 2,9 (100 / 34,7) ; à +9 % (si le coût suit les FLOP), la boucle resterait
  limitée par la collecte, débit ≈ std, **≈ 45 à 51k/s (estimé, non mesuré)**.
- Contre-exemple mesuré : la SIL (ratio d'échantillons 0,24) a coûté −37 % et saturé le learner (96,9 %) : ses coûts fixes (V pour
  les priorités, arbre de sommes, deuxième backward, moments fantômes) n'ont pas été décomposés. L'estimation ci-dessus peut donc
  être fausse d'un facteur important.
- Le banc D5 consigne : débit appris mural, **occupation du fil learner**, RAM (D2). **KS4** si < 40k/s : alerte au rapport, GO non
  bloqué (règle d'ADR-009) ; K6 à 25k inchangé. Durée du run : ≈ 5,5 h si débit ≈ std, ≈ 8,5 h si débit ≈ SIL (estimés).

### D8. Préalables (Definition of Done, avant le GO ; tests en `tests/unit`, CPU, < 60 s)
Fichiers : module `src/training/auxtask.py` et tests `tests/unit/test_auxtask.py` (pas `aux.py` : AUX est un nom de fichier réservé
de Windows, même suivi d'une extension, d'après la documentation Microsoft « Naming Files » ; non vérifié sur cette machine) ;
`configs/int_aux.yaml`, `configs/smoke_aux.yaml`, `configs/bench_int_aux.yaml` ; entrée `"int_aux"` dans les interventions de
`scripts/run_4a.py` (l. 70-75).
1. `aux.enabled: false` : métriques PPO et poids bit à bit identiques à std sur N itérations (CPU, même graine) ; aucun calcul
   auxiliaire exécuté.
2. Section `aux` absente : hashs de trajectoire de `run_4a_std.yaml` et `run_4a_s.yaml` inchangés (comme `test_sil.py:511`) ;
   `configs/int_aux.yaml` ne diffère de `run_4a_std.yaml` que par `aux` (hors `name` et commentaires) ; `aux` activée refusée en
   GCO-S (`GCO_SPEC.md:75`) ; clés inconnues refusées.
3. C2 : gradient de L_aux **exactement nul** sur la couche de sortie de π et sur tous les paramètres de V ; `state_dict` de π
   inchangé (mêmes clés que std) ; l'export et `src/eval/match.py` chargent π sans la tête.
4. Étiquettes sur épisodes synthétiques : but au pas T ⇒ ±1 sur [T − H + 1, T] selon le signe de r(i), csc compris ; épisode plus
   long que H ⇒ 0 hors fenêtre ; troncature ⇒ les H derniers pas masqués ; camp gelé exclu ; épisode plus court que H.
5. Parité float16 : |Δ log softmax(tête)| max ≤ 10⁻² (seuil de π, ADR-009 D7-5) sur lot synthétique, puis au banc sur un lot réel ;
   sinon float32 et H/2 (D3 ; H = 50 depuis le repli D2 du 2026-11-22). **Amendé le 2026-11-23** : échec au lot réel
   (0,0137) ⇒ float32 à H = 100 (amendement de D3, validé par l'utilisateur).
6. Rejeu : tirage 4 096 + 4 096, FIFO par compartiment, pas de lot sous 4 096 transitions ±.
7. Reprise exacte en CPU avec l'état auxiliaire (fenêtre, rejeu, tête, état Adam de la tête, générateur) dans le checkpoint
   (`.claude/rules/training.md`) ; θ₀ de π identique à std s0 (méthode de `tests/unit/test_run_4a_interventions.py:414`).
8. grad_share et disp_share : valeurs sur un cas manuel ; disp_share = 0 et trajectoire inchangée quand λ = 0.
Puis : revue `code-reviewer` ; **smoke test vert sur le commit du run** (`scripts/train.py configs/smoke_aux.yaml --smoke`, CLAUDE.md
règle 5) ; bancs D5 (`scripts/bench_run.py --config configs/bench_int_aux.yaml --eval-config configs/run_s1v2_eval.yaml --name-prefix
bench_int_aux --minutes 30 --prefill-pool 100 --sample-s 1`, arbre propre, machine libre : **jamais en parallèle d'un run**), résumé
par un script `scripts/aux_bench_summary.py` ; résultats (λ, débit, occupation learner, RAM, parité) consignés ici avant le GO.

**État des préalables (2026-11-21)** : tests 1 à 8 faits (rl-algo-lead, ea9cbf1, 51 tests dans `tests/unit/test_auxtask.py`,
11,5 s mesurés par la revue ; suite unitaire 750 passés, 3 ignorés), revue `code-reviewer` APPROVE (aucun bloquant ; hashs de toutes
les configs existantes inchangés, recalculés avec l'ancien et le nouveau `src/`). Restent : smoke, bancs D5, parité f16 sur lot réel
(`scripts/aux_f16_parity.py`, préparé). Précisions d'implémentation, validées en revue :
- λ du banc : 0,1 puis 0,0316 puis 0,01 (÷ √10 exact, règle d'ADR-009 D5 ; « 0,03 » et « 0,3 » ci-dessus en sont les arrondis),
  montée à 0,316 puis 1 si la médiane est < 0,05 à 0,1 ; le script compare les λ à 10 % près et retient la valeur mesurée.
- Un terminal avec r = 0 (fin sans but) donne la classe 0 (« signe de r(i) »).
- Calendrier optionnel `aux.anneal_start_timesteps` / `aux.anneal_end_timesteps` (absent ⇒ λ constant, D6) : permet la lecture
  prudente de C2 par la seule config si l'utilisateur la choisit ; fin < budget exigée ; fenêtre et rejeu libérés en fin de calendrier.
- SIL et aux actives ensemble : refusé (config et learner), conformément à D11.
- `configs/smoke_aux.yaml` : bloc `aux` identique à `int_aux` ; budget de 48 itérations au lieu de 18 (reprise à 24, soit 168 pas
  par agent > H = 150 : sans cela le compartiment y = 0 reste vide et aucun lot n'est tiré, smoke 690a828 du 2026-11-22). Écarts
  à `smoke.yaml` testés (`tests/unit/test_auxtask.py`). L'ancien écart `batch_size` 1 024 reposait sur une estimation fausse.

**État des préalables (2026-11-22) : banc D5 à λ = 0,1, H = 150, et repli D2.** Mesure : `scripts/bench_run.py --config
configs/bench_int_aux.yaml --eval-config configs/run_s1v2_eval.yaml --name-prefix bench_int_aux_l0p1 --minutes 30 --prefill-pool
100 --sample-s 1`, commit a81559f (arbre propre), graine 0, Core Ultra 5 125H, XPU, secteur, 2026-11-22 10:38 → 11:08 UTC ;
données `runs/bench_int_aux_l0p1_20261122T103801Z/`, résumés `aux_summary.json` et `bench_summary.json`
(`scripts/aux_bench_summary.py`).
- D5 : fenêtre concluante (511 itérations, rejeu plein) ; grad_share médiane **0,478** (p10 0,405, p90 0,564) ⇒ **λ = 0,1 retenu** ;
  disp_share médiane 0,454. Débit appris mural **43,7k/s** (KS4 non déclenché) ; fil learner occupé **0,96**.
- RAM : disponible minimum **807 Mo** (échantillons à 1 s, pendant le checkpoint de l'itération 750), chute maximale **2 088 Mo**
  pendant un checkpoint (médiane 1 663 Mo sur 8 checkpoints : 7 de 523 Mo, 189 Mo à l'itération 0) ; KS1 non déclenché ;
  RSS max 8 110 Mo, privée max 8 976 Mo.
- **Décision mécanique D2** (`scripts/aux_bench_summary.py:211-220`, sortie `ram_rule` : « H = 100 ») : minimum < 1,5 Go ⇒
  **H = 100** dans `configs/int_aux.yaml`, `configs/bench_int_aux.yaml` et `configs/smoke_aux.yaml` (blocs `aux` identiques,
  tests). H = 100 = 6,7 s à 15 Hz, γ¹⁰⁰ = 0,61 ; fenêtre 160,8 Mo, état auxiliaire 216,5 Mo (octets, test). Repli float32 de D3
  (« même budget d'octets ») : H = 50 en float32 (et non plus 75), à confirmer par l'utilisateur si la parité D8-5 échoue.
- Restent, dans l'ordre : smoke `configs/smoke_aux.yaml` vert sur le commit (1er lot possible à l'itération 15, reprise à 24,
  test) ; **nouveau banc λ = 0,1 à H = 100** (même commande) ; parité f16 sur un checkpoint de ce banc (`scripts/aux_f16_parity.py`).
  **Si ce banc mesure encore un minimum < 1,5 Go : PAS DE GO (D2).**
- Gain attendu de H = 100 (**estimé, non mesuré**) : −80,4 Mo résidents (fenêtre) et checkpoint 523 → ≈ 442 Mo ; la chute au
  checkpoint vaut 3,3 à 3,9 fois sa taille en médiane (4,2 au max) sur les bancs SIL et aux (copies de sérialisation de `_pack`,
  `src/training/checkpoint.py:352-357` : clone d'`encode`, `BytesIO`, deux concaténations), soit ≈ −0,25 à −0,32 Go ; total
  ≈ +0,32 à +0,40 Go, minimum attendu ≈ 1,1 à 1,2 Go, **sous 1,5 Go** : H = 100 ne suffit vraisemblablement pas. Ce qui domine
  (lecture hors ligne de `bench_samples.jsonl`, interprétation) : la mémoire privée du processus hors checkpoint (médiane 6,36 Go)
  est celle du banc SIL 2 (6,29 Go, état 302 Mo) et de M8 sans aux (6,09 Go) ; les pics de RSS et de mémoire privée tombent dans
  les checkpoints ; l'écart de disponible avec le banc SIL 2 (médiane 2,67 contre 4,16 Go hors checkpoint) vient surtout de la
  mémoire utilisée hors du processus (total − disponible − privée : 6,30 contre 5,29 Go), cause non identifiée.

**Préalables à H = 100 (2026-11-22, commit 7910742, arbre propre, après l'écriture des checkpoints en flux d'ADR-003 §7).**
- Smoke : `configs/smoke.yaml --smoke` PASS (`runs/smoke_7910742`) ; `configs/smoke_aux.yaml --smoke` PASS et
  `aux_bench_summary.py --smoke` PASS (`runs/smoke_aux_7910742`).
- Banc D5 λ = 0,1, H = 100 (`runs/bench_int_aux_l0p1_20261122T113256Z`, même commande) :
  - grad_share médiane **0,497** (p10 0,411, p90 0,581) ⇒ **λ = 0,1 retenu** (D5, marge de 0,003 sous 0,5) ;
  - disp_share médiane 0,436 ;
  - débit appris 44,0k/s (KS4 non déclenché ; 1B ≈ 6,3 h extrapolé) ; fil learner occupé 0,965 ;
  - **RAM disponible min 2 261 Mo** ≥ 1,5 Go ⇒ **D2 OK** ; chute max au checkpoint 1 314 Mo (2 088 Mo avant l'écriture en flux).
- **Parité float16 D8-5 : FAIL** (`runs/int_aux_d8/f16_bench_int_aux_l0p1_20261122T113256Z.json`, lot réel de 51 611 obs, CPU) :
  max |Δ log softmax(tête)| = **0,0137** > 0,01 ; moyenne des maxima par ligne 0,0012 ; aucune obs non finie.
  - La règle D3 littérale impose float32 avec H/2 = 50 (3,3 s).
  - Alternative, à trancher par l'utilisateur (amendement) : float32 à H = 100. La RAM est mesurée par un banc dédié (ci-dessous).
- Banc RAM float32, H = 100 (`runs/bench_int_aux_f32_h100_20261122T120430Z`, commit 50d90b5, arbre propre, même commande ;
  config = `bench_int_aux.yaml` avec `store_float16: false` seul changé, copie hors dépôt) :
  - RAM disponible min **1 884 Mo** ≥ 1,5 Go (D2 OK, marge 384 Mo) ; chute max au checkpoint 2 486 Mo ;
  - grad_share médiane 0,474 ⇒ λ = 0,1 ; débit appris 44,6k/s.
  - L'option float32 à H = 100 tient donc en RAM sur ce banc. Si elle est retenue, `store_float16: false` dans les trois configs
    (un amendement, validé par l'utilisateur).

**Amendement D3 / D8-5 (2026-11-23, validé par l'utilisateur) : float32 à H = 100.**
- `aux.store_float16: false` dans `configs/int_aux.yaml`, `configs/bench_int_aux.yaml` et `configs/smoke_aux.yaml` (blocs `aux`
  identiques, `tests/unit/test_auxtask.py`, `tests/unit/test_run_4a_interventions.py`) ; défaut `AuxConfig.store_float16`
  inchangé (`True`), les configs le fixent. Configs SIL inchangées.
- Tailles en float32 (octets des tableaux, `tests/unit/test_auxtask.py`
  `test_memoire_aux_dimensions_du_run_int_aux_mesuree_en_octets`, 7 373 agents appris, obs 109) : fenêtre 7 373 × 100 × 436
  + 7 373 × 8 = **321 521 784** octets ; rejeu 250 000 × 441 (obs 436, étiquette 1, itération 4) = **110 250 000** octets ;
  état auxiliaire **431 771 784** octets (431,8 Mo), contre 216 540 384 en float16 à H = 100. Il dépasse l'état SIL du banc 2
  (301 617 984) : l'argument RAM de D2 repose sur le banc float32 dédié ci-dessus, plus sur cette comparaison.
- Calculs (non mesurés) : fenêtre du smoke (8 192 agents) 357 236 736 octets ; copie hôte → appareil de D4 6 × 8 192 × 436
  = 21,4 Mo par itération (10,7 Mo en float16).
- Mesuré au banc float32 (`runs/bench_int_aux_f32_h100_20261122T120430Z/bench_summary.json`) : checkpoint de 657 567 622
  octets à l'itération 250 (523 Mo au banc float16 à H = 150).
- Restent avant le GO : smoke `configs/smoke_aux.yaml --smoke` vert sur le commit du run (CLAUDE.md règle 5), les points 1 à 4
  ci-dessous, puis le GO de lancement.

### D9. Journal C4 (`metrics.jsonl`, chaque itération ; `aux/*`)
`coef`, `loss`, `batch_present` (minibatchs avec lot), `recall_pos`, `recall_neg`, `recall_zero`, `precision_pos`, `precision_neg`,
`precision_zero` (sur le lot rééquilibré : le rappel ne dépend pas du rééquilibrage, la précision si), `frac_pos`, `frac_neg`,
`frac_zero`, `frac_masked` (parmi les pas sortis de la fenêtre dans l'itération), `replay_pos_size`, `replay_zero_size`,
`replay_pos_age_mean` (itérations), `grad_norm_trunk_aux`, `grad_norm_trunk_ppo`, `grad_share`, `disp_share`, `ram_available_mb`,
`ram_private_mb` (obligatoires, comme la SIL).

### D10. Protocole et critère de succès
- Run `int_aux_s0`, GCO-std, **graine 0, 1B appris**, config = `configs/run_4a_std.yaml` + bloc `aux` (seule différence, test D8-2).
  Éval en cours de run : protocole v2 (ADR-006 D2, D3), décrite au rapport, jamais comparée.
- Commande (après GO) : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant int_aux --seed 0 --max-hours 0`.
- Comparaison **uniquement** par la commande hors ligne d'ADR-009 D8 aux itérations de std s0 (0, 1 938, …, 19 376), sorties dans
  `runs/int_aux_offline_eval/s0_b<0.k>_it<it>.json`, puis 20 000 épisodes à 1B.
- **Succès = ADR-006 D6, inchangé** : (i) à 1B, borne basse du Wilson 95 % des décisives ≥ **6,28 %** **et** SPRT +100 Elo H1 ; ou
  (ii) 1re paire de points consécutifs H1 au plus tard à **0,2B**. **Neutre** sinon. Base de comparaison : std s0 (pas de base
  cumulée, ADR-006 D6). Si une intervention antérieure est promue en base par ADR avant le lancement, cet ADR est amendé d'abord.
- Descriptif (aucun test) : std s0, GCO-S s0, SIL s0, ADR-007 si fait ; touche, Elo décisif par point, `aux/*`.

### Kill criteria (en plus de K2 à K6 d'ADR-005 D6 amendés par ADR-006, inchangés)
| # | Critère | Action |
|---|---|---|
| KS1 | Mémoire disponible < 1,0 Go pendant > 10 min | STOP propre ; H ou rejeu révisés par amendement |
| KS4 | Débit appris < 40k pas/s sur 30 min | alerte, au rapport (K6 à 25k inchangé) |
Pas d'arrêt anticipé pour succès ni pour absence de but avant 1B (ADR-005 D3) ; les touches restent un diagnostic (ADR-006 D7).

### D11. Place
Ordre validé par l'utilisateur le 2026-11-21 : analyse post hoc SIL (faite, `int_sil_s0.md` §7) → ADR-012 (phase A en pause, reprise
vers 23 h ; phase B avant ADR-007 si SUCCÈS et GO, ADR-012 D8) → ADR-007 (γ 0,998, accepté pour exécution par délégation) →
**ADR-010** → variante SIL éventuelle (nouvel ADR). Le code et les tests D8 peuvent avancer pendant les runs (CPU, < 60 s) ; le smoke
et les bancs attendent une machine libre, entre la fin d'ADR-007 et le lancement de ce run. Combinaison SIL + aux seulement après deux
résultats isolés.

## Conséquences
- Plus facile : une tête linéaire, une perte de plus, aucun pas d'optimiseur ajouté ; π exporté inchangé ; infrastructure de la SIL
  réutilisée (schéma joint, moments fantômes, journal RAM).
- Plus difficile : une fenêtre et un rejeu de ≈ 0,3 Go en RAM et dans chaque checkpoint (≈ +0,3 Go, estimé) ; l'état Adam de PPO voit
  le gradient auxiliaire sur le tronc ; une graine (variance ×1,5 entre graines std) : un « neutre » ne réfute pas la tâche.
- Menaces : le seul gain publié (Lucy, ×2 en échantillons) est sur une récompense dense ; la classe 0 est biaisée vers le début des
  épisodes (masquage) ; l'interprétation de C2 (D6) peut être refusée.
- Révision si : rappel des classes ± ≤ 1/3 (prédiction uniforme à 3 classes) après 0,2B ; grad_share
  < 0,05 au λ retenu ; KS1 au banc ; publication des codes Terminal / Pulsar 2.5 (`GCO_SPEC.md` §8).

## Points à valider par l'utilisateur
1. Cible à 3 classes à horizon **150 pas** (10 s, au lieu de 200 : règle RAM mesurée ; **H = 100** depuis le repli D2 du
   2026-11-22, obs float32 depuis l'amendement du 2026-11-23), masquée aux troncatures ; tête sur le tronc de
   π, hors des poids exportés ; rejeu à deux compartiments de 125 000, lot 8 192 rééquilibré.
2. λ_aux fixé par les bancs D5 (grad_share médiane ≤ 0,5, descente 0,1 → 0,03 → 0,01), **constant sur 1B**.
3. Interprétation de C2 (D6) : une perte de représentation qui ne touche ni avantages, ni sortie de π, ni V, est conforme sans
   annealing, à condition d'être loggée ; sinon repli sur le calendrier de la SIL.
4. Débit : KS4 en alerte seulement (estimé ≈ std, non mesuré) ; protocole et critère d'ADR-006 D6 ; place après ADR-007 (D11).
