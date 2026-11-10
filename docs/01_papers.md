# 01 — Papiers : techniques, hyperparamètres rapportés, adaptation RL-GCO (Phase 0b)

Date : 2026-11-10. Synthèse du `research-director`. Notes détaillées, avec chaque chiffre sourcé (section/table/annexe) :
- `docs/research/paper_openai_five.md` : OpenAI Five, arXiv 1912.06680, **lu intégralement** (§1–6 + annexes A–Q), plus PPO,
  GAE, McCandlish (critical batch), Bansal 2017, Balduzzi 2019.
- `docs/research/paper_logo_2202_and_refs.md` : arXiv 2202.04628 **lu intégralement**, plus AlphaStar, What Matters, Implementation
  Matters, 37 details, PPG, TrueSkill/Elo/SPRT, Pleines 2022, Lucy-SKG, DAPG, R2D3.
- annexe privée (non versionnée) : techniques d'exploration communautaires (discussions Discord + rapports du dépôt public pulsar2).

Conventions : **[src]** = valeur rapportée par la source citée. **[dérivé]** = notre calcul à partir de valeurs sourcées.
**[proposition]** = choix du studio, à valider par la mesure (Phase 2–4). Rien ici n'a encore été mesuré sur notre machine.

## 0. Alerte sur l'identité de 2202.04628

arXiv 2202.04628 = **LOGO**, « Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration »
(Rengarajan et al., ICLR 2022). Ce n'est **pas** un papier PPO à grande échelle ni de self-play. Son apport porte sur la
**récompense éparse**, via un guidage KL décroissant vers une politique de démonstration sous-optimale. Les sujets self-play,
pools et ratings sont couverts par les références (OpenAI Five, AlphaStar, Bansal, Balduzzi, TrueSkill).

## 1. Techniques — apport, valeurs rapportées, adaptation, compatibilité GCO

### 1.1 Boucle PPO et détails d'implémentation
- **Apport** : PPO-clip + GAE, robustes à grande échelle (Five : seul algorithme à progresser au départ [src §3.2]). Selon
  Engstrom 2020, les détails d'implémentation (scaling, init, LR annealing, grad clip) pèsent plus que PPO vs TRPO [src §3, Tab. 2].
- **Valeurs** : clip 0,2 (Five Tab. 2 ; PPO Tab. 1 : meilleur sur MuJoCo) ; Adam β 0,9/0,999, ε 1e-5 (37 details) ; grad clip
  global 0,5 (37 details, Engstrom, Pleines) ou par paramètre ±5√v (Five §3.2) ; init orthogonale, sortie politique ×0,01
  (37 details ; What Matters §3.2 : ×1/100) ; value clipping **nuisible** (What Matters §3.4) ; avantages recalculés à chaque
  epoch (WM §3.5) ; alarme si approx_kl > 0,02 (37 details).
- **Adaptation** [proposition] : PPO synchrone, clip 0,2, grad clip global 0,5, init 0,01 de la tête politique (≈ uniforme sur
  90 actions, donc exploration initiale maximale, utile en sparse), pas de value clipping, normalisation des avantages par
  minibatch (indispensable en GCO : l'échelle de A varie avec la fréquence des buts).
- **GCO** : compatible (optimisation pure).

### 1.2 Horizon, γ et λ (crédit à long terme)
- **Apport** : H = T/(1−γ) (Five Eq. 3). Allonger l'horizon d'un agent **déjà compétent** améliore le win-rate jusqu'à 6–12 min
  de jeu, avec des rendements décroissants (Five §4.5, Fig. 6). Le seul run Dota **win/loss pur** a utilisé H = 1 h, γ = 0,99996
  (Five App. G, Fig. 16).
- **Valeurs** : Five 180 s → 360 s (Rerun), λ = 0,95 fixe (Tab. 2) ; Lucy-SKG γ ≈ 0,995 (demi-vie 10 s, tick-skip 8)
  [src App. 5] ; pulsar2 pré-GCO γ = 0,9969 (demi-vie ≈ 15 s), goal critic γ = 0,9994 (l'annexe privée) ; en GCO,
  un γ « normal » selon un auteur communautaire [annexe] ; GAE λ optimal ∈ [0,9 ; 0,99], nettement plus bas que γ optimal
  (GAE §6.3) ; What Matters : λ = 0,9, γ = hyperparamètre parmi les plus importants (§3.4, §3.6).
- **Adaptation** [dérivé] : à 15 Hz, T = 0,0667 s, demi-vie = 0,693·T/(1−γ). On a γ 0,995 → 9,2 s ; 0,9969 → 14,9 s ;
  0,998 → 23 s. [proposition] γ₀ = **0,995**, puis montée **linéaire** vers 0,997–0,998 une fois que l'agent marque
  régulièrement. Le protocole d'ablation est celui de Five Fig. 6 : reprendre un checkpoint avec H ∈ {10, 20, 40 s} et mesurer
  le win-rate contre le départ. λ = 0,95 par défaut, ablation {0,9 ; 0,95 ; 0,975} (0,975 donne la même fenêtre temporelle de
  2,67 s que Five).
- **GCO** : compatible (γ ne modifie pas la récompense).

### 1.3 Taille de batch, staleness, réutilisation d'échantillons
- **Apport** : un batch plus gros accélère, mais **de façon sous-linéaire**. Five : 8× plus de données/itération → 2,5× moins
  d'itérations (§4.3), soit ≈ 3,2× plus d'échantillons consommés [dérivé]. Un gros batch achète du wall-clock **avec du matériel
  en plus**, pas de l'efficacité par échantillon. B_crit ≈ B_simple = tr(Σ)/|G|², qui croît d'un ordre de grandeur ou plus
  pendant le training (McCandlish §3, App. A.1) ; Dota 1v1 : 50k → 3M.
- **Staleness** : +8 versions ⇒ ralentissement significatif ; asynchrone à staleness 0–1 ⇒ ≈ 3× plus rapide en wall-clock à
  performance égale par itération (Five §4.4, Fig. 5b, Fig. 26). **Reuse** 2–3 ⇒ ≈ 2× plus lent, 8 ⇒ échec (Five §4.4) ; en
  PPO petite échelle, 3–10 epochs restent la norme (PPO Tab. 3/5) ; PPG : 1 epoch politique quasi optimal, la valeur tolère
  plus de réutilisation (PPG Fig. 3, §1).
- **Communauté** : pulsar2 pré-GCO 200k/itération, minibatch 25k, 2 epochs ; Pulsar GCO 200k (août) → 5M (nov.)
  (l'annexe privée).
- **Adaptation** [proposition] : débit fixe sur un laptop, donc rester **≤ B_crit mesuré**. Départ à 50k–100k steps/itération,
  2 epochs (ablation {1, 2, 3} en wall-clock), minibatch 12,5k–25k. Logger B_simple et grossir le batch quand il dépasse ~2×
  le batch courant. Staleness ≤ 1 (synchrone ou double buffer collecte/update), jamais de file de rollouts.
- **GCO** : compatible. Contrainte spécifique : viser **assez d'événements « but » par batch** (≥ quelques dizaines) pour que
  l'avantage porte un signal ; métrique à logger.

### 1.4 Normalisations
- **Valeurs** : obs = moyenne/écart-type courants, clip ±5 (Five App. E) ; normaliser les obs est « toujours » utile (WM §3.3) ;
  normalisation de la valeur : effet fort et **mitigé** selon l'environnement (WM §3.3) ; récompense normalisée par écart-type
  courant (Five Tab. 2 note c) ; reward scaling important (Engstrom §3). pulsar2 : `standardizeReturns = true`, obs non standardisées.
- **Adaptation** [proposition] : obs mises à l'échelle par **constantes physiques fixes** (dimensions d'arène, vitesse max), pas
  par running stats. Sinon chaque snapshot du pool et le bot RLBot doivent embarquer leurs propres stats (risque de parité).
  Récompense **R = ±1** (l'échelle ±150 de Pulsar est équivalente à une constante près). Avec des épisodes coupés au but, les
  retours sont bornés dans [−1, 1] : on démarre **sans** normalisation de retour ni de valeur, puis A/B test. Un scaling constant
  positif n'est pas du shaping.
- **GCO** : compatible.

### 1.5 Architecture
- **Valeurs** : Five, LSTM 4096 (justifié par le brouillard de guerre de Dota) ; réseaux séparés meilleurs sur 4/5 env (WM §3.2) ;
  interférence politique/valeur si partage (PPG §1) ; Pulsar : trunk+policy ≈ 5,8–6 M paramètres, **critic plus gros que la
  policy**, policy surdimensionnée = dégradation ; Terminal : premier run en petit MLP, attention après 10–15B
  (l'annexe privée).
- **Adaptation** [proposition] : Rocket League vu par RLGym/RLBot est **entièrement observé**, donc pas de LSTM au départ.
  Politique MLP (≈ 3×256–512) et **critique séparé plus large** (≈ 3×512–1024), dernière action dans l'obs. Taille contrainte
  par le CPU : chaque step d'inférence coûte du débit. Ablation de taille après les premières mesures de SPS.
- **GCO** : compatible.

### 1.6 Récompense éparse/terminale : ce que disent les papiers
| Source | Résultat | GCO ? |
|---|---|---|
| Five App. G, Fig. 16 | Win/loss seul, H = 1 h : TrueSkill 155 > agent scripté (≈ 100), « grosse pénalité » en échantillons (non chiffrée) ; seule preuve sparse du papier, à petite échelle | référence |
| Bansal 2017 §5.3 | Sans récompense dense annealée, les agents n'explorent pas la récompense de compétition ; ceux qui finissent en sparse battent ceux qui gardent la dense (Fig. 3) | dense = **interdit** |
| Pleines 2022 (2205.05061) | Striker +1 au but seulement : ~75 % de buts après transfert, **mais 2 runs sur 5** seulement réussissent ; shaping naïf essayé, sans gain (§V-C, §VI-C) | argument pro-GCO + **variance inter-seeds** |
| LOGO (2202.04628) | Guidage KL décroissant vers π_b sous-optimale : atteint ~l'expert dense ; BC warm-start seul stagne (§5.1) ; δ₀ 0,008–0,2, α 0,9–0,95, K_δ 5–50 (Tab. 1) | **zone grise** (prior de politique) |
| R2D3 (1909.01387) | Ratio de démos optimal très petit mais non nul (1/256) | zone grise |
| PPO+D (2007.03328), SIL, UPGO (AlphaStar) | Rejouer ses **propres** trajectoires réussies | **compatible** (n'utilise que ±R) |
| Lucy-SKG (2305.15801) | Tâche auxiliaire « reward prediction » : plateau à ~100M vs ~200M (§5.1) ; state setter : 70 % d'états de replays | aux = compatible ; replays = zone grise |
| Five App. O.2 | Randomiser l'état initial (santé de Roshan aléatoire) pour rendre une tâche dure apprenable, puis anneal 0→100 % (App. B) | **zone grise** : modifie la distribution initiale, pas la récompense |

**Leçon** : aucun papier ne montre un sparse pur qui marche **à petit budget** sans aide à l'exploration. Les leviers qui
n'ajoutent aucune récompense sont la distribution d'états initiaux, un horizon assez long, la self-imitation, les tâches
auxiliaires et l'entropie. Ils sont classés dans `GCO_SPEC.md` §4.

### 1.7 Machinerie d'exploration communautaire (pas de papier arXiv ; sources = repo pulsar2 + threads)
| Technique | Ce qui est public | Coût / risque | GCO ? |
|---|---|---|---|
| **Composition critic V†** (`repos/pulsar2/research/reports/COMPOSITION_CRITIC.md`) | twin heads, TD 1-step **expectile τ = 0,75**, cible min(V†₁, V†₂), H = relu(min V† − V), PBRS **Φ = +H** injecté dans les avantages, centré, σ-matché (β 0,15 papier / 0,30 code), clamp ±3σ ; le signe est critique ; bench : 0 % → 4–9 % d'aerials hauts | « serious issues in its public version » ([annexe]), « fall apart over time in self play » (`:6411`) ; surcoût SPS 1 % à 50 % selon les sources | admis par Jumper775, contesté par d'autres |
| Reachability critic (HER + InfoNCE) | distance non euclidienne, gate de rewards (pulsar 2.1) | moyen | admis (pas une récompense) |
| Optimistic / expectile critic | τ ≈ 0,8, « knowing–doing gap » (`KNOWING_DOING.md`) | mesure seule dans le code | admis |
| Goal critic à long γ | critique goal-only γ = 0,9994, blend β = 0,25 | faible | admis |
| World model pour l'exploration (Pulsar 2.5) | concept seulement (« un ICM amélioré », selon son auteur) | inconnu | admis |
| SIL + learned state setters (bot GCO communautaire) | **commence à marquer à 350M steps** ([annexe]) | le seul point de données GCO à petit budget | SIL compatible ; LSS zone grise |
| RND / ICM | ICM jugé cassé, RND fonctionnel mais décevant, selon un auteur communautaire [annexe] | bonus intrinsèque = récompense ajoutée | **interdit** en lecture stricte |

Lu pour mémoire, **non relu** dans cette phase : Ng, Harada & Russell 1999 (PBRS et invariance de la politique optimale). Le
clamp ±3σ du composition critic **casse** l'invariance exacte là où il est actif (l'annexe privée.5).

### 1.8 Self-play et pools d'adversaires
- **Five** (App. N) : 80 % contre la dernière version, 20 % contre le passé. Score de qualité q_i, tirage p_i ∝ e^{q_i} ; si
  l'agent courant gagne, q_i ← q_i − η/(N·p_i), **η = 0,01** ; snapshot toutes les **10 itérations** avec q = max. Objectif :
  éviter le **strategy collapse**.
- **AlphaStar** : PFSP, f_hard = (1−x)^p (p non trouvé), f_var = x(1−x). Main agents : 35 % SP, 50 % PFSP contre tout le passé,
  15 % contre les « oubliés » et les exploiters. Exploiters ajoutés au-dessus de 70 % de victoires, reset 25 %.
- **Bansal** (Tab. 1) : s'entraîner seulement contre le plus récent = **le pire** choix. **Balduzzi** : le self-play naïf cycle
  dans les jeux non transitifs.
- **Communauté** : pulsar2 pré-GCO 30 % d'anciennes versions ; Pulsar GCO, PFSP ([annexe]) ; Disco ≈ 20 %.
- **Adaptation** [proposition] : 75–80 % self-play courant, 20–25 % pool. Pool à scores de qualité Five (30 lignes, coût nul) au
  départ, PFSP-f_hard ensuite ; snapshot toutes les 10–20 itérations ; pool plafonné. Pas d'exploiters avant d'avoir du compute
  à revendre. Données d'entraînement : les deux camps contre le courant, seulement le nôtre contre le passé.
- **GCO** : compatible (on ne change que la distribution d'adversaires). L'entraînement **contre des bots externes** (Nexto)
  est une zone grise (jugé « inorganique » en discussion) ; on les réserve à l'**évaluation**.

### 1.9 Évaluation et promotion statistique
- **TrueSkill (Five App. J)** : σ = 25/3, β = σ/2, τ = 0, draw 0,02 ; 83 références fixes, aléatoire = 0 ; on ne joue que contre
  des références à ≤ 10 TS ; 750 parties par version ; 8,3 TS ≈ 80 % de victoires.
- **Protocole retenu** [proposition, calculs dans `paper_logo_2202_and_refs.md` §13] :
  - **unité** = épisode kickoff → premier but (timeout = nul), joué par paires avec échange des côtés ;
  - **rating** = Bradley–Terry/Elo MLE ancré sur l'agent aléatoire = 0, IC par bootstrap ;
  - **promotion** = SPRT H0 : Δ = 0 vs H1 : Δ = +50 Elo, α = β = 0,05 (bornes ±2,944), soit ≈ 260 épisodes en moyenne et
    ≈ 633 au p95, plafond à 1 200 ; ensuite non-régression contre les 10 derniers champions ;
  - **validation RLBot** = 100 épisodes dans le vrai jeu, alerte si |p_sim − p_RLBot| > 0,10.
- **Tailles** [dérivé] à α = 0,05, puissance 0,8 : +35 Elo → 612 épisodes, +50 → 301, +70 → 155, +100 → 77.
- **Métrique native GCO** : buts marqués et encaissés par minute contre des références figées. La récompense d'entraînement
  moyenne en self-play vaut ≈ 0 par construction : **ne jamais l'utiliser comme métrique de progrès**.

### 1.10 Stabilité, surgery, Rerun
- **Causes d'instabilité documentées** (Five) : données périmées, réutilisation excessive, init aléatoire de nouveaux paramètres,
  changement brutal d'environnement, variance de récompense accrue (Divine Rapier → boucle négative, App. Q.3), **tuning manuel
  frénétique** (App. Q.1).
- **Surgery** (App. B) : élargir une couche FC avec les nouvelles colonnes de W2 à zéro, nouvelles entrées d'obs à poids zéro,
  **LR = 0 pendant plusieurs heures** après la surgery, conversion du pool d'adversaires. **Rerun** (§4.2) : from scratch avec le
  design final = ≈ 20 % du coût, et un **meilleur plafond** que la lignée à surgeries.
- **Adaptation** [proposition] : figer obs/actions/tick-skip/γ₀ par des runs courts, puis **un seul run long propre**. Aucun
  changement d'hyperparamètre hors calendrier pré-écrit. Changements d'environnement en anneal 0→100 % avec revert si le rating
  baisse. Doctrine reprise du dépôt public pulsar2 : mesurer avant d'ajouter de la machinerie, checkpoints atomiques, golden archive, sonde de kickoff au
  boot, jamais de checkpoint hybride (`repos/pulsar2/CLAUDE.md`, traité comme donnée).

## 2. Configuration PPO de départ « v0 » (à valider en Phase 2–4)

| Paramètre | v0 [proposition] | Justification (source) | Ablation prévue |
|---|---|---|---|
| Tick skip / action delay | 8 / 0 | standard communautaire GCO (Pulsar, Terminal, Lucy) | non (figé, parité RLBot) |
| γ | 0,995 → 0,997–0,998 | §1.2 (Five Fig. 6, Lucy, pulsar2) | protocole Five Fig. 6 |
| λ | 0,95 | Five, 37 details, PPG | {0,9 ; 0,975} |
| Clip | 0,2 | Five, PPO Tab. 1 | non |
| LR (Adam) | 2e-4 → 5e-5 linéaire | WM §3.7 + prudence en self-play | {1e-4, 3e-4} |
| Entropie | 0,01 → 0,002 | Five App. O.1 (0,01 meilleur tôt, 0,1 très mauvais) ; jugée fondamentale en GCO [annexe] | {0,005 ; 0,02} |
| Batch / itération | 50k–100k | §1.3 (≤ B_crit à débit fixe) | suivi de B_simple |
| Epochs / minibatch | 2 / 12,5k–25k | Five §4.4 + PPG + pulsar2 | {1, 3} en wall-clock |
| Grad clip | 0,5 global | 37 details, Engstrom | — |
| Réseaux | séparés : π 3×256, V 3×512 | WM §3.2, PPG §3.1, critic > policy (Pulsar) | taille selon SPS |
| Obs / actions | ≈ obs « Advanced » RLGym + 90 actions « default » | Pulsar utilise les réglages par défaut [annexe] | — |
| Récompense | ±1 au but, 0 sinon | `GCO_SPEC.md` | variantes de la spec |
| Timeouts | troncature avec **bootstrap** V(s′) | pulsar2 `GAE.cpp:66-96` ; WM §3.6 ne s'applique pas (limites courtes) | — |
| Self-play | 80 % courant / 20 % pool à qualité (η = 0,01, snapshot /10 itérations) | Five App. N | PFSP-f_hard |
| Seeds | ≥ 3 pour toute conclusion | Pleines §V-C (2/5), Five Fig. 20 | — |

## 3. Ce qui ne transfère pas
1. Le **shaping** de Five (Tab. 6 : plus de 20 termes) et d'AlphaStar (pseudo-récompenses) : interdit.
2. Le **compute** : Five Rerun ≈ 1,7·10¹² timesteps [dérivé]. Les bots GCO de référence ont tourné entre 10¹¹ et 10¹² steps.
   Nous sommes 3 à 4 ordres de grandeur en dessous (voir `GCO_SPEC.md` §6).
3. « Plus gros batch = mieux » : faux à débit fixe (§1.3).
4. LSTM 4096, infra distribuée asynchrone, LR calibrés pour 159 M paramètres.
5. La surgery comme mode de vie : la physique de Rocket League ne change pas, la bonne leçon est Rerun.

## 4. Lacunes de lecture (honnêteté)
- Valeurs des courbes des figures de Five non lisibles en texte (seules les légendes le sont).
- AlphaStar : page Nature derrière un login, exposants « 2×10^? » illisibles (2e9 selon la réimplémentation DI-engine, non vérifié).
- TrueSkill (NeurIPS 2006) : PDF illisible ; paramètres pris dans Five App. J, sur la page MSR et sur trueskill.org.
- Ng 1999, Net2Net, IMPALA (abstract seulement), appendices de What Matters : non lus.
- **Aucun papier ne décrit Pulsar GCO, Pulsar 2.5 ni Terminal** : leur code est privé jusqu'après la RLBC.
