# GCO_SPEC — Goal Concede Only (v0.2)

Statut : **validé au Gate G1** (2026-11-10, décisions au §10). Auteurs : research-director + reward-engineer.
Sources : discussions communautaires (exports Discord **privés, non versionnés**), code public `lucamignatti/pulsar2`
(pré-GCO), papiers de `docs/01_papers.md`. Les citations et références `fichier:ligne` vers les discussions sont dans une
annexe privée locale (`docs/private/`), notée **[annexe]** ci-dessous. Ce dépôt n'en reproduit aucune citation.

## 1. Ce que « GCO » veut dire dans la communauté (synthèse, preuves en [annexe])

- **Récompense** : uniquement une constante positive au but marqué et son opposé au but encaissé (×150 chez Pulsar, l'échelle
  est sans importance). Dans le code public pré-GCO, `GoalReward` donne +1 à chaque joueur de l'équipe qui marque et −1 à
  l'autre, selon la **position de la balle** (les csc comptent), exactement à somme nulle (`CommonRewards.h:30-43`,
  `ExampleMain.cpp:253-254`).
- **Aucun autre terme** : pas de pénalité de temps ni de prior humain. Toute récompense façonnée, même temporaire, fait sortir
  du GCO.
- **Le GCO naïf ne marche pas** : l'exploration est trop difficile, puisque la plupart des actions ne touchent pas la balle et
  ne reçoivent aucun gradient au début.
- **Ce qui le fait marcher n'est pas une récompense** : des critiques appris injectés dans les avantages, un world model pour
  l'exploration, le self-play avec PFSP. La fonction de valeur apprend à partir du seul signal GCO.
- **Points contestés** :
  - **state setters** : certains auteurs GCO en utilisent, d'autres revendiquent du « pur » kickoff ;
  - **critiques PBRS** : comptent-ils comme GCO ? (débat ouvert) ;
  - **Terminal** : un « élément fait main » annoncé, jamais révélé.

## 2. Définition retenue (hypothèse principale H-GCO-std)

### 2.1 Signal de récompense — exact
Pour chaque agent *i* d'équipe *e(i)* et chaque step de décision *t* (tick-skip 8, soit 15 Hz) :

```
r_t(i) = +1  si un but est marqué pendant le step t ET la balle est entrée dans le but de l'équipe adverse de e(i)
r_t(i) = −1  si un but est marqué pendant le step t ET la balle est entrée dans le but de e(i)   (csc inclus)
r_t(i) =  0  sinon (y compris aux troncatures et au reset)
```
- L'attribution se fait **par équipe** (tous les coéquipiers reçoivent le même signal), d'après le **but touché par la balle**,
  pas d'après le dernier toucheur. C'est le comportement de `GoalReward`. La variante « scorer-only » est **hors spec**.
- **Échelle ±1** : elle est équivalente au ±150 de Pulsar à une constante positive près (pas de shaping). Elle garde les retours
  bornés dans [−1, 1].
- **Zéro-somme exact** à chaque step : Σ_i r_t(i) = 0 (1v1, 2v2, 3v3).
- **Aucun autre terme** : pas de pénalité de temps, pas de décroissance temporelle, pas de récompense de touche, de démo, de
  boost, de distance, de vitesse vers le but, de kickoff, d'« énergie ».

### 2.2 Épisodes et terminaisons
| Événement | Type | Bootstrap | Récompense | Source / justification |
|---|---|---|---|---|
| But | **terminal** | non (V = 0 après) | ±1 | `GoalScoreCondition` = NORMAL (`TC/GoalScoreCondition.h:8-14`) |
| Aucune touche depuis 20 s de jeu | **troncature** | oui, V(s′) | 0 | `NoTouchCondition(20)` (`ExampleMain.cpp:313-316`) ; [v0, ablation 20/45 s] |
| Durée d'épisode 120 s | **troncature** | oui, V(s′) | 0 | `maxEpisodeDuration = 120` (`PPOLearnerConfig.h:190`) |
| Horloge de match, overtime, nul | **n'existent pas à l'entraînement** | — | — | pratique pulsar2 ; les matchs de 5 min ne servent qu'à l'évaluation |

- Une troncature n'est **jamais** traitée comme un terminal à 0. Sinon le critique apprend une fausse valeur d'« état sans but »
  (WM §3.6 ne s'applique pas ici : nos limites sont courtes).
- L'observation **ne contient pas** le temps restant avant troncature : le temps n'est pas une variable du jeu réel en 1v1
  illimité, et l'exposer serait une fuite de signal.

### 2.3 États initiaux (H-GCO-std)
- **Kickoffs** (5 positions officielles, couleurs alternées) : proportion **p_kick**.
- **États génériques non ciblés** : positions et vitesses aléatoires de la balle et des voitures, boost aléatoire, balle près
  d'une voiture : proportion 1 − p_kick.
- Calendrier [proposition] : p_kick = 0,2 au départ, annealing linéaire vers **p_kick ≥ 0,8** avant toute promotion finale
  (analogue de la santé de Roshan aléatoire puis de l'anneal 0→100 %, Five App. O.2/B).
- **Interdits** en H-GCO-std : états ciblant une mécanique (flip-reset-ready, air-dribble setup, AirPlay/AirDrill de pulsar2),
  rejeu de replays humains, learned state setters.

## 3. Trois niveaux de définition (pour les ablations et pour lever l'ambiguïté)

| Élément | **GCO-S** (strict) | **GCO-std** (retenu) | **GCO-ext** (zone grise, pistes séparées seulement) | Preuve |
|---|---|---|---|---|
| Récompense | §2.1 | §2.1 | §2.1 | [annexe] |
| États initiaux | kickoffs seuls | kickoffs + génériques non ciblés, annealés vers kickoff | + ciblés, replays, learned state setters | [annexe] vs [annexe] ; [annexe] |
| Adversaires d'entraînement | self-play + passé (PFSP) | idem | + bots externes (Nexto) | [annexe] ; [annexe] |
| Signaux appris injectés dans les avantages (PBRS sur potentiel appris, composition/goal critic) | interdits | **autorisés** sous conditions C1–C4 | autorisés | [annexe], [annexe], [annexe] |
| Bonus intrinsèques (RND, world model, nouveauté) | interdits | autorisés sous C1–C4, coefficient → 0 avant la fin | autorisés | [annexe], [annexe] |
| Self-imitation (SIL/UPGO sur nos propres épisodes avec but) | interdit | **autorisé** (n'utilise que ±1) | autorisé | [annexe], PPO+D |
| Tâches auxiliaires (prédire le prochain but) | interdites | **autorisées** (perte de représentation, pas récompense) | autorisées | Lucy-SKG §5.1 |
| Démos / BC / KL vers Nexto (LOGO, DAPG) | interdits | interdits | autorisés (piste « GCO+demo ») | LOGO §5.1 ; [annexe] |
| Masquage d'actions invalides, obs/actions par défaut | autorisés | autorisés | autorisés | [annexe] |
| Entropie | autorisée | autorisée | autorisée | [annexe] |

**Conditions C1–C4** pour tout signal non-récompense en GCO-std :
- **C1, agnostique au jeu** : aucun concept de Rocket League codé à la main (balle, but, boost, touche, distance). Il doit être
  dérivé de modèles appris à partir des seules transitions et du signal ±1. Critère inspiré d'une
  exigence communautaire d'agnosticisme total au jeu [annexe].
- **C2, objectif final inchangé** : soit potential-based sur un potentiel appris, soit coefficient annealé à 0 avant la phase
  finale. Le clamp ±3σ du composition critic casse l'invariance exacte là où il est actif ; on l'accepte, mais il est **loggé**.
- **C3, interrupteur** : chaque signal est désactivable par config, et une ablation GCO-S est menée en parallèle.
- **C4, journal** : magnitude du signal vs |A| GCO loggée à chaque itération ; alerte si le signal domine durablement (> 50 % de
  la variance des avantages).

## 4. Critères de succès (mesurés selon le protocole de `01_papers.md` §1.9)

| Jalon | Critère | Mesure |
|---|---|---|
| **S0 Pipeline** | Tests de la reward GCO verts (§7), parité obs/actions entraînement↔RLBot, smoke test sans NaN | CI locale |
| **S1 Amorçage** | L'agent marque contre lui-même gelé : buts/min > 0 de façon stable sur 3 seeds, **et** bat un agent aléatoire (SPRT +100 Elo) | épisodes kickoff |
| **S2 Base** | Bat un bot scripté « fonce vers la balle » (ATBA) avec SPRT +50 Elo, en sim **et** en RLBot | sim + RLBot |
| **S3 Compétent** | Bat un bot communautaire faible de la botpack RLBot v5 (à choisir en Phase 5) avec significativité | RLBot |
| **S4 Nexto** (stretch) | Win-rate > 50 % contre Nexto sur 5 min, IC95 au-dessus de 0,5 | RLBot |
| **Conformité** | Audit reward : 0 terme hors §2.1, tous les signaux C1–C4 listés et loggés | revue indépendante |

## 5. Points de données disponibles pour calibrer le budget (auto-déclarés en discussion, preuves en [annexe])

| Bot (type) | Jalon | Steps | Matériel | Fiabilité |
|---|---|---|---|---|
| Bot GCO communautaire (SIL + learned state setters) | **commence à marquer** | **350M** (version précédente : ~800M) | non précisé | moyenne |
| Pulsar1 (GCRL, GCO pur) | bas niveau (bronze/argent) à l'arrêt | inconnu | — | faible |
| Pulsar 2.5 (GCO + stack complet) | niveau Nexto | 60–70B (annonce initiale 15B, corrigée) | supercalculateur | moyenne |
| Pulsar2 GCO | 42-0 contre Nexto | ≤ 100B | supercalculateur ≈ 160B/jour | élevée |
| Terminal (GCO + exploration non divulguée) | niveau Nexto | 2–6B (**chiffre d'un tiers**) ; final 1v1 100–120B | ≈ 4 GPU en moyenne | faible / élevée |
| Non-GCO (référence) | niveau Nexto | 30–50B habituellement ; Disco 6–30B | desktops RTX 5080 | moyenne |
| Débits desktop rapportés | — | Pulsar ≈ 200k sps (RTX 5080) ; Disco ≈ 187k sps (9800X3D + 5080) ; ≈ 1,5M sps (285K + 4090) | — | moyenne |

## 6. Budget de steps et temps d'entraînement estimé (ordre de grandeur, NON mesuré)

### 6.1 Budget de steps par jalon (incertitude ≈ ×3 à ×10, intervalles subjectifs à 80 %)
| Jalon | Fourchette | Médiane estimée | Raisonnement |
|---|---|---|---|
| S1 Amorçage | 0,2 – 3B | ~0,7B | bot GCO communautaire 0,35–0,8B avec SIL + LSS ; sans aide à l'exploration, **peut ne jamais arriver** |
| S2 bat ATBA | 0,5 – 10B | ~2B | après amorçage, un ballchaser se bat vite (inférence ; aucune donnée GCO publique) |
| S3 bot faible | 2 – 30B | ~8B | entre S2 et Nexto |
| S4 Nexto | 5 – 150B | ~40B | Terminal 2–6B (faible fiabilité) ; Pulsar 60–100B ; non-GCO 30–50B |

### 6.2 Conversion en temps sur CE laptop (débit soutenu **inconnu**, à mesurer en Phase 2)
Scénarios de débit bout-en-bout soutenu (collecte + apprentissage, CPU seul, laptop branché). Ce sont des **hypothèses** encadrant
les ordres de grandeur, **pas des mesures**. Chiffres rapportés ailleurs (détail et URLs : `docs/02_ecosystem.md`) :
- RLGym-PPO (Python) ≈ 10k SPS contre RLGymPPO_CPP ≈ 70k SPS, sur un i5-11400 + RTX 3060 Ti (README RLGymPPO_CPP) ;
- RocketSim seul ≈ 114k ticks/s/thread en 2v2 sur un i5-11400 (README RocketSim), soit ≈ 14k steps/s/thread à tick-skip 8
  avant tout coût d'obs, d'inférence ou d'apprentissage [dérivé] ;
- desktops GigaLearn avec RTX 5080 : 190–200k SPS ([annexe], [annexe]).

Estimation de l'ingénieur écosystème pour ce laptop, **non mesurée** : option A (Python, rlgym-learn) ≈ 3–6k SPS effectifs ;
option B (chemins chauds natifs, PPO et reward en Python) ≈ 10–25k SPS. Les trois colonnes ci-dessous encadrent ces options.

| Steps | 5k SPS (Python pessimiste) | 20k SPS (Python optimisé / hybride) | 60k SPS (natif optimiste) |
|---|---|---|---|
| 1B | 2,3 jours | 14 h | 4,6 h |
| 3B | 6,9 jours | 1,7 jour | 14 h |
| 10B | 23 jours | 5,8 jours | 1,9 jour |
| 40B | 93 jours | 23 jours | 7,7 jours |
| 100B | 232 jours | 58 jours | 19 jours |

Facteurs de dégradation à appliquer :
- throttling thermique soutenu de l'ultraportable (non mesuré) ;
- surcoût des critiques d'exploration : d'environ 5 % à presque ×0,5 selon les auteurs [annexe] ;
- évaluation (5–10 % du compute) ;
- machine partagée avec l'usage quotidien (16 Go, **1,3 Go libres** au moment de l'inventaire).

### 6.3 Conclusion du budget
- **S1–S2 sont atteignables** sur ce laptop en jours à semaines, **si** la machinerie d'exploration GCO-std fonctionne à petit
  budget. C'est le **risque n°1** du projet : aucune preuve publique d'un GCO amorcé sous 0,35B steps.
- **S4 (Nexto)** demande de l'ordre de 1 à 4 mois de calcul continu dans le scénario 20k SPS. C'est faisable mais fragile.
- **Le niveau Pulsar/Terminal (≥ 100B) est hors de portée de ce matériel** : ≥ 2 mois au mieux, ≈ 8 mois au pire. Ce plafond
  n'est pas un problème de code, c'est un problème de compute.

## 7. Tests exigés de l'implémentation reward (Phase 3)
1. But bleu ⇒ r = +1 pour chaque bleu, −1 pour chaque orange (et l'inverse) ; somme nulle.
2. Csc ⇒ signe selon le but touché, pas selon le dernier toucheur.
3. Aucun step hors but ne porte de récompense non nulle, troncatures et reset compris (test de propriété sur N épisodes aléatoires).
4. But ⇒ `terminated = True`, `truncated = False` ; no-touch / durée max ⇒ `terminated = False`, `truncated = True`.
5. Symétrie miroir : un état et son miroir bleu↔orange donnent des récompenses opposées et des observations miroir identiques.
6. Pas de fuite : la récompense ne dépend que de l'événement « but » (mock de l'état : toute autre variable modifiée laisse r inchangé).
7. Interface de variantes : `GCO-S`, `GCO-std`, `GCO-ext` et le scorer-only (hors spec, ablation) sont instanciables par config.
   Le hash de la config de récompense est écrit dans chaque checkpoint.
   Variante supplémentaire **hors spec**, diagnostic, jamais promue : `dense-diag` (§2.1 + façonnage dense par pas, pas de
   zéro-somme exact ; `src/rewards/dense.py`), run témoin de l'ADR-011. Les tests 1 à 6 ne portent que sur les variantes in-spec.
8. Non-régression : toute modification de `src/rewards/` exige une mise à jour de cette spec et un ADR.

## 8. Implications pour la conception
- L'effort principal ne porte pas sur la récompense (6 lignes de code). Il porte sur **(a) le débit** et **(b) l'amorçage de
  l'exploration** sans récompense ajoutée.
- Priorité R&D, par ordre de gain attendu / coût :
  1. distribution d'états initiaux génériques annealée ;
  2. horizon γ 0,995 → 0,998 ;
  3. self-imitation sur nos buts ;
  4. tâche auxiliaire « prochain but » ;
  5. composition critic (spec publique, bugs connus) ;
  6. world model / bonus de nouveauté (aucun détail public).
- Les codes de **Terminal** et de **Pulsar 2.5** doivent être publiés **après la RLBC** (fin novembre 2026). Une veille est
  prévue : ils pourraient remplacer les points 5–6.

## 9. Corrections d'une analyse antérieure
Une analyse locale antérieure contenait plusieurs chiffres erronés ou datés. Le détail est dans l'annexe privée ;
les chiffres corrects sont ceux des §5 et §6.

## 10. Décisions du Gate G1 (validé le 2026-11-10)
| # | Décision |
|---|---|
| H1 | Niveau **GCO-std** (conditions C1–C4), avec un témoin GCO-S en run court |
| H2 | États initiaux génériques autorisés, annealés vers ≥ 80 % de kickoffs ; aucun setter ciblé sur une mécanique |
| H3 | Pistes GCO-ext permises **uniquement** en ablations séparées et étiquetées, jamais dans la ligne principale |
| H4 | **Modifié** : l'entraînement se fait sur **Kaggle** (2× T4 ou TPU v5e-8, via la clé API). Le laptop sert au développement, aux smoke tests locaux, à l'orchestration et au déploiement RLBot (voir §11 et `docs/research/kaggle.md`) |
| H5 | Cible : S2 garanti, S4 en stretch ; 1v1 d'abord, architecture prête pour le 2v2 |
| H6 | Code sans licence (GigaLearn leak, pulsar2) : référence uniquement, jamais copié ni utilisé comme dépendance |

## 11. Addendum G1 : budget sur Kaggle (hypothèses, NON mesuré)
> **Remplacé au G2 (2026-11-11)** par les mesures de `docs/adr/001-infra.md` (voir §12). Ce tableau est conservé
> comme trace des hypothèses du G1 ; ses débits (T4 à 6k, laptop à 10k) sont démentis par la mesure.

Contraintes rapportées (détail et sources : `docs/research/kaggle.md`) : ≈ 20 h TPU et ≈ 30 h GPU par semaine, sessions
≤ 9 h (TPU) ou ≤ 12 h (GPU), seulement 4 cœurs CPU en session T4. La collecte étant CPU-bound, le débit dépend surtout du
**nombre de cœurs hôte**, inconnu en TPU v5e-8 (entre 96 et 224 possibles, à sonder).

| Scénario | SPS soutenu | Heures/semaine | Steps/semaine | S2 (~2B) | S4 (~40B) |
|---|---|---|---|---|---|
| TPU v5e-8, hôte large | 100k | 20 | 7,2B | < 1 semaine | ≈ 6 semaines |
| TPU v5e-8, hôte moyen | 40k | 20 | 2,9B | ≈ 1 semaine | ≈ 14 semaines |
| T4 ×2 (4 cœurs) | 6k | 30 | 0,65B | ≈ 3 semaines | hors de portée |
| T4 ×2 en relais sur ~6 comptes opérés par leurs titulaires (§5 de `kaggle.md`) | 6k | 168 | 3,6B | < 1 semaine | ≈ 11 semaines |
| TPU v5e-8 avec file d'attente ≥ 35 h (≈ 9 h utiles / 44 h) | 40–100k | ≈ 34 | 4,9–12B | < 1 semaine | 4–8 semaines (si l'hôte est large) |
| Laptop 24/7 (rappel §6.2) | 10k | 168 | 6,0B | < 1 semaine | ≈ 7 semaines (throttling non mesuré) |

Ce tableau montre surtout ceci : **sans hôte TPU large, Kaggle n'apporte pas plus que le laptop**. Mise à jour du 2026-11-10 : le TPU impose une file d'attente d'au moins 35 h, la ligne principale passe donc sur des **sessions T4 en relais** (chaque compte opéré par son titulaire, voir les CGU dans `docs/research/kaggle.md` §5), avec le TPU en appoint opportuniste. Le cumul des deux (Kaggle
+ laptop) reste une option. La Phase 2 tranche par mesure : sonde Kaggle d'abord, puis benchmark. Aucun chiffre de ce
tableau n'est mesuré.

## 12. Décisions du Gate G2 (validé le 2026-11-11)
Détail, commandes et commits : `docs/adr/001-infra.md`.
- **Infra A3** : simulateur RocketGPU @f0508a1 en CPU, environnement maison batché, PPO et récompense en Python
  (torch), learner sur GPU intégré (XPU) en local ou CUDA sur Kaggle T4.
- **Débit de planification** : plateau soutenu **mesuré** du laptop ≈ 50k timesteps/s (57 min, politique aléatoire,
  sans critiques d'exploration). 1B ≈ 5,6 h, 10B ≈ 2,3 jours, 40B ≈ 9,3 jours. Avec le facteur ×0,7 (estimé, non
  mesuré) pour GCO-std et l'évaluation : 7,9 h, 3,3 jours, 13 jours.
- **Plateformes** : laptop = plateforme principale ; Kaggle T4 = appoint (4 vCPU mesurés, bout-en-bout non mesuré) ;
  rôle du TPU décidé après sa sonde (en file au moment du GO).
- **Acceptés non mesurés au GO** : RAM du laptop, learner JAX sur TPU, relais Kaggle jamais exécuté contre Kaggle. Le
  débit sera remesuré avec la vraie boucle au run court (Phase 4a), avant le G3.

