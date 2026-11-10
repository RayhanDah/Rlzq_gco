# arXiv 2202.04628 + références cœur — fiche de lecture pour RL-GCO

Contexte cible : bot Rocket League 1v1, PPO, GCO (seule récompense : +R but marqué / −R but encaissé, aucun shaping), laptop CPU-only (Core Ultra 5 125H 14C/18T, 15.4 Go RAM, iGPU Arc, pas de CUDA), déploiement RLBot v5, tick-skip 8 (15 décisions/s), ~90 actions discrètes.

Convention : chaque nombre porte sa source (papier, section/table/figure). "non trouvé" = absent des textes lus. "[calcul]" = calcul fait par moi (script Python, formules données). "[proposition]" = recommandation de ma part, non issue d'une source.

---

## 0. ALERTE identité de 2202.04628

- **arXiv 2202.04628 n'est PAS un papier PPO à grande échelle / self-play.** C'est :
  **"Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration"**, Desik Rengarajan, Gargi Vaidya, Akshay Sarvesh, Dileep Kalathil, Srinivas Shakkottai (Texas A&M), **ICLR 2022** (en-tête du PDF), v1 9 fév. 2022, v2 13 fév. 2022. Algorithme : **LOGO** (Learning Online with Guidance Offline).
- Ce papier traite de : récompenses éparses (pertinent GCO), démonstrations sous-optimales hors ligne, TRPO à deux étapes. Il ne parle **ni** de self-play, **ni** de ligues, **ni** de TrueSkill/Elo, **ni** de PPO à grande échelle. Ces sujets sont couverts via les références de la Tâche B (+ OpenAI Five en bonus, car cité par PPG et directement pertinent).

---

## 1. Ce que j'ai réellement lu / pas pu lire

| Document | Accès | Profondeur |
|---|---|---|
| 2202.04628 LOGO (PDF v2, 21 p.) | PDF → `pdftotext` local | **Intégral** : §1–6, réf., App. A–H (Tables 1–2, Fig. 3). Lettres grecques perdues à l'extraction → recoupées avec la version HTML arXiv (App. F, Tables 1–2, App. H). |
| AlphaStar (Vinyals et al., Nature 2019) | PDF "unformatted" DeepMind (storage.googleapis.com) | Main text + Methods (RL, PFSP, Populating the League, Infrastructure, Evaluation, Metrics) + légendes Ext. Data. Nature.com : redirection login (non accessible). Exposants "2×10^?" illisibles dans le PDF (voir §14). |
| OpenAI Five, 1912.06680 (bonus) | PDF → pdftotext | §3–4, App. C (Table 2 hyperparamètres), App. J (TrueSkill), App. N (self-play). |
| What Matters…, 2006.05990 | abs + HTML (premiers 100k car.) | Recommandations §3.1–3.8 + constats. Appendices non lus. |
| Implementation Matters…, 2005.12729 | abs + HTML | §3–6, Tables 2–3. |
| 37 Implementation Details of PPO (ICLR blog 2022) | page blog | Liste complète des détails + notes. |
| PPG, 2009.04416 | PDF → pdftotext | **Intégral** (§1–5, App. A–D). |
| TrueSkill (Herbrich, Minka, Graepel, NeurIPS 2006) | PDF MSR : polices Type3 **illisibles** ; proceedings.neurips.cc : connexion refusée | **Papier non lu.** Remplacé par : page projet Microsoft Research TrueSkill, docs `trueskill.org`, et usage dans OpenAI Five App. J. |
| SPRT / fishtest | Wikipedia SPRT, chessprogramming.org SPRT, wiki fishtest "Fishtest-mathematics" | Formules de seuils (Wikipedia), bornes usuelles (CPW). PDFs détaillés fishtest (nElo, GSPRT) **non lus**. |
| Pleines et al. 2022, **arXiv 2205.05061** (IEEE CoG 2022) | PDF → pdftotext | **Intégral.** |
| Lucy-SKG, 2305.15801 | PDF → pdftotext | **Intégral** (texte + appendice). |
| Autres RL × Rocket League | recherche web | "Rocket Learn" (SMU Data Science Review 5(2) art. 12, 2021) : abstract seulement. Survey 2509.03682 (IEEE ToG 2025) : abstract seulement (cite RL comme plateforme, sans détails). Aucun autre papier arXiv RL×RL trouvé. |
| Réfs citées par LOGO | abs arXiv / PDF | DAPG 1709.10087 (PDF, §III-C, §IV-C), R2D3 1909.01387 (PDF, §4, §6.2, Fig. 6, App. D.3), PPO+D 2007.03328 (abs), POfD ICML 2018 (page PMLR, abs), AWAC 2006.09359 (abs), DQfD 1704.03732 (abs), Nair 2018 1709.10089 (abs). Non lus : GAIL, TRPO, CPO, Vecerik 2017 (connus mais non re-vérifiés → aucun chiffre cité). |

---

## 2. 2202.04628 — LOGO (lecture complète)

### 2.1 Problème et hypothèse
- Setting : MDP à récompense éparse + données de démonstration D = {(s,a)} d'une politique de comportement π_b **sous-optimale**, **sans récompenses** (§2.2, App. G). Variante : démonstrations à état censuré s̃ = o(s) (projection vers sous-espace) (§2.2, §4.1).
- **Assumption 1** (§2.2) : en début d'apprentissage, E_{a∼π_b}[A^{π_k}(s,a)] ≥ ϱ > 0 ∀s — π_b donne un avantage positif sur la politique courante. Pas besoin que π_b soit optimale.

### 2.2 Algorithme (§3.1, §4, App. D, Algorithm 1)
- **Étape 1 (amélioration)** = TRPO : π_{k+1/2} = argmax E[A^{π_k}] s.t. D̄_KL(π, π_k) ≤ δ (Eq. 1).
- **Étape 2 (guidage)** : π_{k+1} = argmin D̄_KL(π, π_b) s.t. D^max_KL(π, π_{k+1/2}) ≤ δ_k (Eq. 2). Le rayon δ_k **décroît** → l'influence de π_b disparaît (§3.1 ; "LOGO only uses the behavior policy to guide initial exploration").
- Surrogate échantillonnable (Prop. 2, Eq. 8–9) : coût politique-dépendant C_π(s,a) = log(π(s,a)/π_b(s,a)) ; J_C(π) = (1−γ)^{-1} D̄_KL(π, π_b) (Lemma 9) ; on minimise E_{s∼d^{π_{k+1/2}}, a∼π}[A_C^{π_{k+1/2}}(s,a)]. Lemma 2 = "performance difference lemma" généralisé à une récompense dépendant de la politique (Eq. 7).
- Mise à jour pratique (Eq. 10) : deux pas de type gradient naturel ; θ_{k+1/2} = θ_k + sqrt(2δ/(gᵀF⁻¹g)) F⁻¹g ; θ_{k+1} = θ_{k+1/2} − sqrt(2δ_k/(hᵀL⁻¹h)) L⁻¹h.
- Si π_b **connue** : C = log(π_k/π_b) directement (Alg. 1, l.4–5). Sinon **discriminateur** GAN B(s,a) entraîné sur D vs D_{π_k} (Eq. 18), C = −log B(s,a) (App. D) ; repose sur la bijection politique ↔ mesure d'occupation (Syed et al. 2008).
- Censuré : C̃(s,a) = log(π(s,a)/π̃_b(o(s),a)) ; discriminateur sur (s̃, a) (§4.1).

### 2.3 Garanties (§3.2)
- Prop. 1 (Achiam 2017) : J(π_{k+1/2}) − J(π_k) ≥ −sqrt(2δ)·γ·ε_{R,k}/(1−γ)² (Eq. 3).
- Théorème 1 (i) si Assumption 1 tient : borne TRPO + terme ≥ 0 ϱ/(1−γ) − ε·sqrt(2 D̄_KL(π_{k+1}, π_b))/(1−γ) (Eq. 5) ; (ii) sinon : −(sqrt(2δ)γε + 3R_max·δ_k)/(1−γ)² (Eq. 6) → même garantie que TRPO si δ_k = O(sqrt(δ)) (Remark 2).

### 2.4 Expériences et résultats
- MuJoCo sparsifié (§5.1) : Hopper/HalfCheetah/Walker2d : récompense +1 par pas **seulement** après avoir avancé de 2/20/2 unités ; InvertedDoublePendulum : récompense **seulement en fin d'épisode**. Évaluation finale sur la récompense dense Gym (§5.1). 5 seeds, moyenne ± écart-type (Fig. 1).
- Baselines : Expert (TRPO dense), Behavior (expert partiellement entraîné), TRPO sparse, GAIL, POfD, BC-TRPO (warm-start BC), DAPG (§5.1).
- Constats (§5.1, Fig. 1a) : TRPO sparse ne progresse quasi pas ; GAIL plafonne au niveau Behavior ; **BC-TRPO profite du warm-start puis n'apprend plus** ; POfD reste biaisé par les démos tout au long ; LOGO atteint ~l'expert dense ; LOGO > DAPG sauf 1 env. Valeurs numériques finales : **non données en tableau** (courbes seulement).
- Censuré (§5.1 Setting 2, Fig. 1b) : LOGO reste bon mais plus lent ; BC-TRPO et DAPG inapplicables.
- TurtleBot (§5.2, App. E) : waypoint à ≤1 m, épisode 20 s, récompense +1 si atteint sinon 0 ; obstacle : −1 collision ; 15 actions discrètes (App. E.1) ; Lidar 360 → 6 valeurs (min par secteur 60°, App. E.5) ; Gazebo pas 0.001 (App. E.2) ; π_b entraînée par TRPO **dense** dans un simulateur cinématique basse fidélité (App. E.1, récompense dense donnée) → transfert sim-to-sim + sim-to-real réussi (Fig. 2).

### 2.5 Hyperparamètres (App. F, Table 1, Table 2, App. H)
- Réseau politique & valeur : MLP 2×128, tanh ; lr 3e-4 ; γ = 0.99 ; TRPO δ = 0.01 (App. F).
- Décroissance : δ_k = δ_0 pendant K_δ itérations, puis **δ_k ← α·δ_k uniquement quand le retour moyen de l'itération courante > moyenne des 10 itérations précédentes** (App. F).
- Table 1 (δ_0 complet / incomplet ; α ; K_δ ; batch) : Hopper 0.01/0.02, 0.95, 50, 20000 ; HalfCheetah 0.2/0.1, 0.95, 50, 20000 ; Walker2d 0.03/0.05, 0.95, 50, 20000 ; IDP 0.2/0.1, 0.9, 5, 5000 ; Waypoint 0.01, 0.95, 5, 2048 ; Obstacle 0.008, 0.9, 5, 2048.
- Table 2 (démos) : Hopper 3869 échantillons (retour démo 1369.81) ; HalfCheetah 5000 (2658.34) ; Walker2d 4584 (2449.21) ; IDP 183 (340.73) ; TurtleBot : politique π_b connue (pas d'échantillons). → **quelques milliers de paires (s,a) suffisent** (App. G : "small amount" de démos).
- App. H, Fig. 3 (Hopper) : α ∈ {0.85, 0.9, 0.95, 0.97, 0.99, 1} → peu sensible ; α = 1 (pas de décroissance) → plafonne au niveau Behavior. K_δ ∈ {1, 50, 100, 500, 1000, 1500} → peu sensible ; la perf monte dès que la décroissance commence.
- Code : github.com/DesikRengarajan/LOGO (note 1, §5).

### 2.6 Techniques extraites de LOGO → RL-GCO

**T1. Guidage KL vers une politique de comportement, à rayon décroissant**
1. Apport : exploration dirigée en récompense éparse sans imiter à vie ; converge au-delà de π_b (§5.1).
2. Hyperparams : δ_0 ∈ [0.008, 0.2], α ∈ [0.9, 0.95], K_δ ∈ [5, 50], TRPO δ = 0.01 (Table 1, App. F).
3. Adaptation [proposition] : en PPO, ajouter un terme β_k · KL(π_θ ‖ π_b) sur les états du rollout (ou sur un buffer de démos), β_k décroissant géométriquement (×0.95) **seulement** quand la métrique de progrès s'améliore. Point clé : Lucy-SKG (App. A) indique que **Nexto sort 90 actions discrètes** mappées vers les 8 contrôles → si l'on reprend la même table de 90 actions, **π_b (Nexto) est connue explicitement** → C = log(π/π_b) sans discriminateur (Alg. 1 l.4–5). Requiert d'exécuter le modèle Nexto sur son propre obs-builder pour chaque état (coût CPU d'inférence en plus ; licence du dépôt à vérifier).
   - **Piège GCO/self-play** : la règle de décroissance d'App. F utilise "retour moyen > moyenne des 10 dernières itérations". En self-play symétrique le retour moyen ≈ 0 par construction → utiliser à la place le **win-rate vs un pool de référence figé** (cf. §13).
4. Compatibilité GCO : **zone grise.** La récompense d'environnement reste ±R ; mais l'étape 2 optimise J_C, le retour d'une "récompense" log-ratio (Lemma 9) → c'est un objectif auxiliaire d'imitation, et π_b (Nexto) a été entraînée avec du shaping. Décision studio requise ; garder un run **GCO-strict** témoin. À δ_k → 0, l'objectif redevient pur GCO (Remark 2).

**T2. Démonstrations sans récompenses, à état censuré**
1. Apport : utilisable quand les démos n'ont que (s̃, a) (§4.1, App. G).
2. Hyperparams : idem T1 ; δ_0 souvent différent en censuré (Table 1).
3. Adaptation [proposition] : démos issues de bots (Nexto/Necto) rejoués dans le simulateur, en ne gardant qu'un sous-ensemble de features communes (ex. sans timers de boost) ; discriminateur MLP petit (CPU-friendly).
4. GCO : zone grise (même raison que T1).

**T3. Warm-start par BC seul = piège**
1. Apport négatif documenté : BC-TRPO démarre vite puis stagne (§5.1).
2. Hyperparams : non trouvé.
3. Adaptation : si BC initial, le coupler à un guidage décroissant (T1) ou à un KL vers le modèle supervisé (AlphaStar, §4), pas BC seul.
4. GCO : zone grise (prior de politique).

**T4. Évaluer sur la vraie métrique, pas sur la récompense d'entraînement**
1. LOGO entraîne en sparse et évalue en dense standardisé (§5.1).
2. —
3. Pour GCO : l'unique métrique d'évaluation = taux de victoire / différentiel de buts vs références figées (§13). Compatible.

---

## 3. Références citées par LOGO et pertinentes (récompense éparse + démos)

| Réf. | Résultat clé (source) | Hyperparams trouvés | Usage RL-GCO | GCO ? |
|---|---|---|---|---|
| **DAPG**, Rajeswaran et al., RSS 2018, 1709.10087 | Pré-entraînement BC + PG augmenté ; démos réduisent fortement la complexité d'échantillon (abs) ; sans démos, NPG nécessite du shaping soigné (§V-A) | **25 démos** par tâche (§III-C) ; poids w(s,a) = λ0·λ1^k·max A, **λ0 = 0.1, λ1 = 0.95**, "peu sensible" (§IV-C) | Terme BC sur buffer de démos à poids décroissant 0.95^k : variante PPO simple de T1 | Zone grise |
| **R2D3**, Paine, Gülçehre et al., 1909.01387 (ICLR 2020) | Démo ratio = **paramètre clé** ; optimum **très petit mais non nul** (abs, §6.2) ; autres méthodes : 0 trajectoire réussie après "tens of billions of steps" (abs) | Ratios testés 1/256, 1/128, 1/64, 1/32 ; **1/256 meilleur** sur 4 tâches (App. D.3) ; 256 acteurs CPU + 1 learner GPU (§4) ; DQfD ≈ ratio 0.25 (§4) | Si démos dans le batch PPO : ≤1 % du batch | Zone grise |
| **PPO+D**, Libardi & De Fabritiis, 2007.03328 | PPO + une démo humaine + replay de ses propres trajectoires priorisées par récompense et valeur max (abs) | non trouvé (abs seulement) | Rejouer ses propres épisodes "but marqué" (self-imitation) : **compatible GCO** (n'utilise que la récompense GCO) | Self-imitation : compatible ; démo humaine : grise |
| **POfD**, Kang et al., ICML 2018 | Matching de mesure d'occupation → "implicit dynamic reward shaping" (abs) ; LOGO : biaisé tout du long (§5.1) | non trouvé | À éviter (shaping explicite) | **Non compatible** (modifie la récompense) |
| **DQfD**, Hester et al., AAAI 2018, 1704.03732 | Meilleur départ sur 41/42 jeux Atari ; PDD DQN met ~83 M pas pour rattraper ; dépasse la meilleure démo sur 14/42 (abs) | ratio démo auto via replay priorisé (abs) | Off-policy, peu transposable à PPO | Grise |
| **Nair et al.**, ICRA 2018, 1709.10089 | Démos + DDPG + HER : "order of magnitude" plus rapide sur tâches longues (abs) | non trouvé | Idée "reset vers états de démo" ≈ state setter (voir §11) | Grise |
| **AWAC**, Nair et al., 2006.09359 | Fine-tuning online après offline "exceptionally difficult" ; AWAC le permet (abs) | non trouvé | Peu pertinent (nécessite récompenses offline) | Grise |

---

## 4. AlphaStar — League & PFSP (Vinyals et al., Nature 575:350–354, 2019)

Source : PDF "AlphaStar_unformatted" (DeepMind). Sections : Main, Methods "Reinforcement Learning", "Multi-agent Learning", "Populating the League", "Infrastructure", "Evaluation", "Metrics used in Figures".

**Récompense** : issue du match (−1/0/+1) comme récompense terminale, **sans discount** (Methods RL). MAIS : (a) init supervisée + **KL continu vers la politique supervisée** ; (b) **pseudo-récompenses** (distance d'édition build-order, Hamming sur stats cumulées) actives chacune avec proba **25 %**, valeur séparée par pseudo-récompense (Methods "Exploration and diversity"). "Human data … critical" (Fig. 3E ; seule valeur lisible : variante "No human data" ≈ 149 Elo). → **AlphaStar n'est pas GCO-pur.**

**Mise à jour** : politique V-trace, valeurs TD(λ), **UPGO** (retour "upgoing", proche self-imitation), valeur avec observations adverses (training only) ; Adam ; entropie (Methods "Value and policy updates").

**PFSP** (Methods "Prioritised Fictitious Self-Play") : P(B) = f(P[A bat B]) / Σ_C f(P[A bat C]).
- **f_hard(x) = (1 − x)^p**, p ∈ ℝ+ contrôle l'entropie ; f_hard(1) = 0 → pas de matchs contre adversaires déjà battus ; vu comme approximation lisse de max-min (vs max-avg de FSP). Poids par défaut. **Valeur de p : non trouvée.**
- **f_var(x) = x(1 − x)** : adversaires de niveau proche ; utilisé pour main exploiters et main agents en difficulté.
- SP seul cycle/oublie (Fig. 3D) ; PFSP > FSP sur toutes les mesures (Ext. Data Fig. 5).

**Populating the League** :
- **Main agents** : **35 % SP, 50 % PFSP vs tous les joueurs passés, 15 % PFSP vs main players "oubliés" (qu'il ne bat plus) + main exploiters passés** (si aucun, ces 15 % → SP). Snapshot **tous les 2×10^9 pas** (exposant illisible dans le PDF ; 2e9 selon la réimplémentation DI-engine, voir §14). Jamais réinitialisés.
- **League exploiters** : PFSP ; snapshot quand ils battent **tous** les joueurs à **> 70 %** ou timeout 2×10^9 pas ; puis reset vers supervisé avec proba **25 %**.
- **Main exploiters** : jouent contre les main agents courants ; la moitié du temps, et si P(gagner) < **20 %**, PFSP-f_var sur les joueurs issus des main agents ; ajoutés quand ils battent les 3 main agents à > 70 % ou timeout 4×10^9 pas ; reset systématique.
- Composition : 3 main, 3 main exploiters, 6 league exploiters ; 32 TPUv3 par agent, **44 jours** ; ~900 joueurs créés (Main text).
- Infra : 16 000 matchs concurrents par agent ; learner 128 cœurs TPU ; batch **512 séquences** ; ~50 000 pas agent/s ; données rejouées 2× ; params acteurs rafraîchis toutes les 10 s (Methods "Infrastructure").
- Évaluation : Elo interne P[r1 bat r2] = 1/(1+e^{−(r1−r2)/400}) (forme logistique lue telle quelle dans le PDF ; ancrage Elite bot = 0) ; **RPP** (Relative Population Performance via Nash) ; distribution de Nash sur la ligue ; validation vs 17 agents à stratégie fixée ; "Validation Agents qui battent les main agents > 80/160 parties" (Ext. Data Fig.) ; Battle.net : 30/60/30 parties (Main text "Evaluation").

**Techniques → RL-GCO**

**T5. PFSP (f_hard / f_var)**
1. Concentre l'entraînement sur les adversaires problématiques, limite l'oubli/cycles.
2. f_hard=(1−x)^p (p non trouvé), f_var=x(1−x) (Methods).
3. [proposition] Pool de snapshots figés ; win-rate P[A bat B] estimé par EMA des résultats (épisodes kickoff→but) ; mélange : 70–80 % latest self-play, 20–30 % pool via PFSP-f_hard avec p = 2 (choix à tester, non sourcé) ; basculer vers f_var si le main agent est en difficulté (win-rate global < 0.5 vs pool). Coût nul en compute (juste du matchmaking), adapté au laptop.
4. **Compatible GCO** (n'affecte que la distribution d'adversaires).

**T6. Exploiters (main / league)**
1. Trouvent les failles du main agent ; le rendent robuste (Fig. 3A–B).
2. Seuils 70 % / 20 %, reset 25 %, timeouts 2e9/4e9 (Methods).
3. [proposition] Laptop : 1 "main exploiter" léger en alternance (ex. 1 cycle sur 5 du budget), initialisé depuis un snapshot du main, entraîné **uniquement** contre le main courant ; ajouté au pool quand > 70 % vs main (seuil AlphaStar), puis reset. Coût : réduit le budget du main → à mesurer.
4. Compatible GCO.

**T7. KL vers politique supervisée / pseudo-récompenses humaines**
3./4. KL vers un modèle supervisé : **zone grise** (prior). Pseudo-récompenses de stratégie : **non compatible GCO** (shaping).

---

## 5. OpenAI Five (1912.06680) — bonus, très pertinent (PPO + self-play + horizon long)

Sources : §3, §4.3–4.5, App. C (Table 2), App. J, App. N.
- Self-play : **80 % vs dernière politique, 20 % vs anciennes** (§3, Table 2 "Past opponents 20 %").
- Pool : chaque ancien adversaire i a un score de qualité q_i ; échantillonnage softmax p_i ∝ e^{q_i} ; **ajout de l'agent courant toutes les 10 itérations**, q initial = max des q existants ; si l'agent courant bat l'adversaire i : q_i ← q_i − η/(N·p_i), **η = 0.01** ; si l'adversaire gagne : pas de mise à jour (App. N, Eq. 14 ; symboles partiellement perdus à l'extraction).
- PPO + GAE λ = 0.95, clip 0.2, Adam β1 0.9 / β2 0.999, LSTM unroll 16, frameskip 4 (Dota 30 fps → décision toutes les 4 frames, ≈ 0.133 s), entropie 0.01 → 0.001, lr 5e-5 → 5e-6, value loss weight 0.25 → 1.0 (Rerun), **reward normalisée par écart-type courant** (Table 2 note c).
- Horizon H = T/(1−γ), T = 0.133 s ; horizon GAE 180 s → 360 s (Rerun) ; allonger l'horizon d'un agent entraîné **améliore** la perf jusqu'à 6–12 min testées (§4.5, Fig. 6). [calcul] γ = 1 − T/H = 0.99926 (180 s), 0.99963 (360 s).
- Batch : 1–3 M timesteps (§3) ; Rerun 983 040 timesteps ; speedup ~2.5× vs 123k (Fig. 5a, §4.3), **sous-linéaire**.
- Qualité des données : staleness de ~8 versions ralentit fortement ; sample reuse élevé nuit (§4.4, Fig. 5b–c).
- Récompense : **shaped** (team spirit 0.3 → 0.8 etc., Table 2) → **pas GCO**.
- Éval TrueSkill (App. J) : pool de **83 agents de référence** (μ figé), σ = 25/3, β = σ/2, **τ = 0**, draw_probability **0.02** ; ancre : agent aléatoire μ = 0 ; ne joue que vs références à **≤ 10 points TrueSkill** (≈ win-rate entre 15 % et 85 %) ; **750 parties par version** puis log ; mise à jour ~1 fois / 2 h.

**T8. Pool à score de qualité (OpenAI Five)**
1. Alternative simple à PFSP, auto-adaptative.
2. 80/20, η = 0.01, ajout /10 itérations (App. N).
3. [proposition] Implémentation 30 lignes ; snapshot toutes les 10 itérations PPO ; limiter le pool (ex. 50–100 snapshots, RAM : un MLP ~1 M params ≈ 4 Mo → négligeable sur 15.4 Go).
4. Compatible GCO.

**T9. Horizon long**
3. [proposition] GCO = signal à ±1 au but uniquement → horizon long indispensable ; voir table §12 (γ 0.995→0.998).
4. Compatible (γ ne change pas la récompense).

---

## 6. What Matters In On-Policy RL (Andrychowicz et al., 2006.05990)

Setup : > 250 000 agents, 5 env MuJoCo continus (Hopper, Walker2d, HalfCheetah, Ant, Humanoid), 1 M ou 2 M pas, base ≈ PPOv2 à 256 envs parallèles (§2). **Attention : continu uniquement, pas de self-play, pas de discret** → transfert à valider.

Recommandations (texte des encadrés, §3) :
- §3.1 : perte PPO, **clip initial 0.25**, tester plus/moins.
- §3.2 : **dernière couche de politique initialisée ×100 plus petite** (+66 % Humanoid seul) ; softplus pour std (continu) ; tanh ; **réseaux séparés** (meilleurs sur 4/5 env), MLP valeur large, politique éventuellement plus étroite ; 2 couches cachées suffisent.
- §3.3 : **toujours normaliser les observations** (crucial sauf Hopper) ; normalisation de la valeur : effets **forts et mitigés** (crucial HalfCheetah/Humanoid, nuit Walker2d) → à tester ; normalisation d'avantage par minibatch : peu d'effet ; clipping d'obs normalisées : peu d'effet ; gradient clipping : petit gain, seuil peu important.
- §3.4 : **GAE λ = 0.9** ; **pas** de Huber ni de value clipping PPO (nuisent).
- §3.5 : plusieurs passes sur les données ; mélanger les transitions ; **recalculer les avantages à chaque passe** ; plus d'envs parallèles/batch → plus rapide en wall-clock mais peut nuire en échantillons ; trop d'envs ↔ morceaux plus courts, bootstrap plus tôt.
- §3.6 : **γ = hyperparamètre parmi les plus importants**, partir de 0.99 ; frame skip aide sur 2/5 env ; gestion des timeouts sans effet notable **car limites hautes (1000 transitions)**.
- §3.7 : Adam β1 = 0.9, **lr 3e-4** par défaut ; décroissance linéaire : léger gain.
- §3.8 : pas de bénéfice net d'entropie/KL régularisation (sauf HalfCheetah) — en continu.

**→ RL-GCO** (toutes compatibles GCO, ce sont des détails d'optimisation) :
- Normalisation d'obs : obligatoire ; [proposition] normalisation fixe par constantes physiques RL (positions/vitesses divisées par échelles arène) plutôt que running-stats, sinon chaque snapshot du pool doit embarquer ses propres stats.
- Value normalization : avec GCO, retours ∈ [−R, R] naturellement bornés → [proposition] R = 1, pas de normalisation de valeur au départ ; A/B test ensuite.
- Timeouts : en GCO, **beaucoup** d'épisodes finissent sans but au début → [proposition] traiter le timeout comme **troncature** (bootstrap V(s_T)), pas comme terminal à 0 ; le "sans effet" de WM §3.6 ne s'applique pas (limites courtes ici).
- Recalcul des avantages à chaque epoch : coût CPU = 1 passe valeur sur le batch ; acceptable.

---

## 7. Implementation Matters (Engstrom et al., ICLR 2020, 2005.12729)

- 9 optimisations "code-level" de baselines PPO (§3) : value clipping, **reward scaling** (division par l'écart-type d'une somme actualisée glissante, sans centrer ; App. A.2), init orthogonale + scaling par couche, annealing lr Adam, reward clipping ([−5,5] ou [−10,10]), normalisation d'obs, clipping d'obs ([−10,10]), tanh, **clip global du gradient 0.5**.
- Ablation 2^4 (value clip, reward scaling, init, lr annealing) sur Humanoid/Walker2d : reward scaling, annealing et init **façonnent fortement** la perf (§3, Fig. 1).
- PPO **ne maintient pas** le trust region par ratio (max ratio > 1+ε) ; le KL moyen reste < 0.07 mais dépend des optimisations (§4, Fig. 2).
- Optimisations code-level > choix PPO vs TRPO : Hopper PPO 2513 vs PPO-M 2142 ; TRPO+ 2466 vs TRPO 2043 (Table 2) ; ACLI 557/421/444 vs AAI 242/99/224 (Walker2d/Hopper/Humanoid).
- PPO-NoClip ≈ PPO après réglage (Walker2d 2867 vs 3292 ; Humanoid 831 vs 806, Table 3).

**→ RL-GCO** : 1. le "PPO" est un ensemble de détails ; 2. chiffres ci-dessus ; 3. [proposition] fixer une config de référence et **ablater un détail à la fois** avec le protocole §13 (pas sur la récompense d'entraînement, qui est ~0 en self-play) ; surveiller approx_kl, clipfrac ; 4. reward scaling/clipping : sur ±1 constant ce n'est **pas** du shaping (transformation monotone d'un signal unique) → compatible, mais inutile si R = 1.

---

## 8. The 37 Implementation Details of PPO (Huang et al., ICLR Blog Track 2022)

- 13 cœur : envs vectorisés (Atari N = 8, M = 128) ; **init orthogonale gain √2, biais 0, sortie politique 0.01, sortie valeur 1** ; **Adam eps 1e-5** ; lr linéaire → 0 (2.5e-4 Atari, 3e-4 MuJoCo) ; GAE λ = 0.95 ; minibatchs (4 en Atari) ; normalisation d'avantage **par minibatch** ; clip 0.1 Atari / 0.2 MuJoCo ; value clipping (gardé pour fidélité) ; entropie 0.01 Atari / 0 MuJoCo ; **clip grad global 0.5** ; variables de debug (policy/value/entropy loss, clipfrac, approx_kl) ; réseaux partagés vs séparés (séparés par défaut en MuJoCo, 2×64 tanh).
- Atari : 4 epochs, γ 0.99. Continu : obs norm running, clip obs [−10,10], reward scaling, reward clip [−10,10].
- LSTM (5) : reset aux frontières d'épisode, séquences ordonnées, reconstruction des états.
- MultiDiscrete (1) : composantes indépendantes.
- Auxiliaires : clip range annealing (par défaut dans le code), early stop KL (target 0.01, off par défaut), **invalid action masking** (logits → −∞).
- Notes : value clipping sans preuve d'intérêt (Engstrom) voire nuisible (Andrychowicz) ; **approx_kl généralement < 0.02**, au-delà → bug ou politique qui bouge trop vite ; sanity check ~400 sur Breakout.
- vf_coef : valeur 0.5 présente dans la configuration procgen d'après le résumé du blog ; **non vérifié pour le détail cœur**.

**→ RL-GCO** (compatibles GCO) : init 0.01 sur la tête politique (≈ quasi-uniforme sur 90 actions → exploration initiale maximale, utile en sparse) ; masquage d'actions invalides si la table de 90 actions contient des actions impossibles (ex. flip sans jump dispo) — **zone grise mineure** (prior de domaine, pas de reward) ; alarme si approx_kl > 0.02.

---

## 9. Phasic Policy Gradient (Cobbe, Hilton, Klimov, Schulman, 2009.04416)

- Constat 1 : l'**interférence** politique/valeur nuit quand les paramètres sont partagés ; constat 2 : la valeur **tolère beaucoup plus de réutilisation** d'échantillons que la politique (§1).
- PPG : phase politique (PPO, réseaux disjoints) + phase auxiliaire : L_joint = L_aux + β_clone·KL[π_old ‖ π] avec L_aux = ½(V_aux − V_targ)² sur la tête valeur auxiliaire du réseau politique, + entraînement supplémentaire du réseau valeur (§2, Alg. 1).
- Hyperparams (App. A.1) : **N_π = 32, E_π = 1, E_V = 1, E_aux = 6, β_clone = 1, 16 minibatchs par epoch aux par N**. App. A.2 (en-têtes partiellement illisibles, ordre déduit) : γ 0.999, λ 0.95, 256 pas/rollout, 8 minibatchs/epoch, entropie 0.01, clip 0.2, reward normalisée oui, lr 5e-4, 4 workers × 64 envs, 100 M pas, pas de LSTM ni frame stack.
- Résultats : PPG > PPO en efficacité d'échantillons sur ~tous les 16 jeux Procgen (Fig. 2) ; **1 epoch politique quasi optimal** (Fig. 3) ; E_aux plateau vers 6 (Fig. 4) ; phases aux trop fréquentes nuisent (Fig. 5, N de 2 à 32) ; KL fixe (β = 1) ≈ clipping (Fig. 6) ; variante 1 réseau avec gradient détaché ≈ 2 réseaux (Fig. 7) ; PPO : 3 epochs optimal (App. D) ; le partage de features est **critique en Procgen (vision)** mais souvent inutile en faible dimension (§3.1, App. B).

**→ RL-GCO** : obs vectorielle ~100–230 dims → [proposition] **réseaux séparés** d'abord (WM §3.2 + PPG §3.1) ; si on veut réutiliser la valeur davantage (valeur dure à apprendre en GCO), essayer E_V = 2–4 ou une phase aux tous les 32 updates. Coût CPU : la phase aux ≈ +E_aux passes sur un buffer de N_π rollouts → RAM : buffer 32 × 100k transitions × ~200 floats × 4 o ≈ **2.6 Go** [calcul] → réduire N_π à 8–16 sur 15.4 Go. Compatible GCO.

---

## 10. TrueSkill, Elo, intervalles et tests

**TrueSkill (Herbrich, Minka, Graepel, NeurIPS 2006)** — papier PDF non lisible ; sources de substitution :
- Page MSR TrueSkill : prior μ = 25, σ = 25/3 ≈ 8.333 ; classement conservateur **μ − 3σ** (≈ quantile 1 %) ; nombre minimal de parties pour identifier le niveau : **2 joueurs (1v1) = 12**, 4 FFA = 5, 8/16 FFA = 3, 4v4 = 46, 8v8 = 91 ; "le nombre réel peut être jusqu'à 3× plus élevé".
- `trueskill.org` (lib Python) : MU 25.0, SIGMA 8.333, **BETA 4.1667 (= σ/2)**, **TAU 0.0833 (= σ/100)**, **DRAW_PROBABILITY 0.10** ; `quality_1vs1` ≈ proba de nul ; 2 joueurs par défaut → 44.7 %.
- OpenAI Five App. J : β = σ/2, **τ = 0**, draw 0.02, 750 parties par version, adversaires à ≤ 10 points.
- Modèle (performance ~ N(s, β²), marge de nul ε) : décrit par MSR/lib ; formules exactes du papier **non lues**.

**Elo** : P(A bat B) = 1/(1 + 10^{−d/400}) (convention standard ; AlphaStar utilise une forme logistique équivalente, Methods "Metrics"). [calcul] d = 35 → p = 0.550 ; 50 → 0.572 ; 70 → 0.599 ; 100 → 0.640 ; 200 → 0.760.

**SPRT (Wald)** : S_n = Σ log Λ_i ; continuer tant que a < S_n < b ; **a ≈ log(β/(1−α)), b ≈ log((1−β)/α)** (Wikipedia SPRT). Chessprogramming : H0 Elo = Elo0, H1 Elo = Elo1 ; exemples fast-chess `elo0=0 elo1=5 alpha=0.05 beta=0.05` (gainer), `elo0=-5 elo1=0` (non-régression) ; bornes usuelles (table aplatie, appariement ambigu) : Stockfish STC gainer [0, 3] / non-reg [−3, 1], LTC [0, 2] / [−1.75, 0.25] (en "normalized Elo" selon fishtest wiki), Top-30 engines [0, 5]/[−5, 0], Top-200 [0, 10]/[−10, 0]. Fishtest : modèle **pentanomial** (paires de parties) plus précis que trinomial ; GSPRT. Formules nElo / LLR exactes : **non lues** (PDF fishtest non consultés).

**CI** [méthodes standard, calcul] : SE(p̂) = sqrt(p(1−p)/n) ; Wilson pour p proche de 0/1 ; SE(Elo) ≈ (400/ln 10)·sqrt(p(1−p)/n)/(p(1−p)) = 173.7/sqrt(n·p(1−p)). Pour un payoff matrix multi-agents : Bradley–Terry (= Elo MLE) + **bootstrap** par ré-échantillonnage des parties.

---

## 11. Papiers RL × Rocket League

### 11.1 Pleines et al. 2022, "On the Verge of Solving Rocket League…" — **arXiv 2205.05061**, IEEE CoG 2022 (TU Dortmund, Rhine-Waal, Leiden)
- Simulateur Unity réimplémentant la physique RL ; transfert **sim-to-sim** vers le vrai jeu via RLBot (abs). Unity ~**950 pas/s** vs RLBot temps réel **120 pas/s** (Windows, GTX 1080 + Ryzen 7 3700X) (§III).
- Tâches isolées : **goalie** (+1 par arrêt, 1000 tirs échantillonnés) et **striker** (+1 **seule** récompense au but) (§III-B/C) → **récompenses éparses, proches de GCO**.
- Obs : **23** variables normalisées ; actions **multi-discrètes 8 dims** (throttle 5, steer 5, yaw 5, pitch 5, roll 3, boost 2, drift/air-roll 2, jump 2) (§III-D) ; réseau partagé FC256 puis 2 têtes FC256, ReLU (Fig. 4).
- PPO (Table III ; tableau désaligné à l'extraction, interprétation) : γ 0.99, λ 0.95, **16 workers × 512 pas = batch 8192**, 3 epochs, 8 minibatchs, clip grad 0.5, clip 0.2, c1 (valeur) 0.25, lr 3e-4 → 3e-6 linéaire, entropie c2 5e-4 → 1e-5 linéaire, AdamW, value clipping utilisé (Eq. 2–3), avantage normalisé par minibatch (§IV-B).
- Résultats : goalie ~**100 %** d'arrêts, striker ~**75 %** de buts après transfert (abs, Fig. 6) ; ablation de chaque adaptation physique → "catastrophique" (Fig. 6) ; éval = 10 tirs d'entraînement + 10 inédits × 3 → 150 épisodes par point ; 5 runs ; IQM + IC 95 % (Agarwal et al.) (§V).
- **Striker difficile : 2 runs sur 5 réussissent**, les autres échouent totalement (§V-C) → variance inter-seeds énorme en sparse.
- **Shaping essayé sans gain** (§VI-C) : récompense premier touché, distance balle-agent, produit scalaire vitesse/direction → biais, exploitations ("cuddle with the ball"), instabilité de la valeur si on coupe le shaping. Propose curriculum (avec échantillonnage de distribution de scénarios contre l'oubli) ou shaping comme pistes.

**→ RL-GCO** : (a) argument empirique **pour** GCO (le shaping naïf n'a pas aidé ici) ; (b) **lancer ≥ 3–5 seeds** ou relancer les runs morts : 2/5 succès en sparse (§V-C) ; (c) les écarts simulateur/jeu peuvent être catastrophiques → valider systématiquement dans RLBot v5 (protocole §13, étape 4) ; (d) batch 8192 a suffi pour des tâches isolées, pas pour le 1v1 complet.

### 11.2 Lucy-SKG (Moschopoulos, Kyriakidis, Lazaridis, Vlahavas), **arXiv 2305.15801**, 25 mai 2023 (template NeurIPS 2023 ; venue non indiquée sur arXiv)
- SB3 1.5.0 + RLGym 1.1.0 ; PPO "device-alternating" (CPU rollouts / GPU training) (App. 5) ; architecture Perceiver/cross-attention de Necto, acteur et critique **séparés** (§4.2) ; **2v2 self-play**, 10–20 instances (App. 5).
- Hyperparams (App. 5) : **320 000 pas de rollout, batch 4000**, clip 0.2, Adam lr **1e-4**, **entropie 0**, vf 0.5, **frame skip 8 (15 actions/s)**, **γ ≈ 0.995 (demi-vie 10 s)**, poids aux SR/RP = 1 ; batch limité par la mémoire (~16 Go par séquence de batch). Ablation aux (Table 6) : 10 epochs, lr 5e-5, entropie 0.01, vf 1, n_steps 1 M, batch 10 % du rollout, γ = e^{log 0.5/(fps·demi-vie)}.
- Terminaison : 5 min, 45 s sans touche, but (Table 4). **State setter** (copie Necto) : états de replays réels (Platine→SSL) **0.7**, aléatoire 0.15, kickoff 0.05, kickoff-like 0.05, goalie-practice 0.05 (App. 5).
- Obs : 8 objets (sans boost pads), drapeau "démoli" sans timers, **empilement des 5 dernières actions** (§4.3). Simplification ≈ ×2 FPS (~4000 vs ~2000) (§5.4).
- Récompense : **fortement shapée** (KRC : Rc = sgn·(Π|R_i|)^{1/n}, §4.4) → hors GCO.
- Tâches auxiliaires : Reward Prediction (LSTM 32, séquence 20, classes +/0/−, seuil ±0.009) et State Representation (autoencodeur 128-32-16) (App. A.2) ; RP plateau à ~100 M pas vs ~200 M baseline (§5.1).
- Résultats (Table 1, 1 B pas) : Lucy-SKG vs Nexto "300-4", vs Necto "300-54" ; ~5× plus rapide que Necto (Fig. 4, Table 7). **Protocole ambigu** : "total score sur 300 matchs single-goal" mais scores 300-54 (somme > 300) ; "Win %" = % du temps où l'équipe 1 mène au score cumulé ; **aucun IC**. Notable : Necto bat Nexto 300-4 à 1 B pas (Table 1).
- Nexto : **90 actions discrètes** mappées vers 8 contrôles (App. A.1.3).

**→ RL-GCO** :
- Tick-skip 8, γ ≈ 0.995 : point de départ cohérent (compatible).
- Tâche auxiliaire RP **sous GCO** = prédire "but pour / contre / rien dans les k prochains pas" : représentation, **pas** une récompense → **compatible GCO** (même logique que la phase aux PPG). [proposition] tête aux sur le réseau valeur ou politique, perte CE, poids 0.1–1.
- Empilement des 5 dernières actions : compatible (obs).
- State setter replays 70 % : **zone grise** (modifie la distribution initiale, pas la récompense ; injecte des connaissances humaines). [proposition] Run GCO-strict = kickoffs uniquement ; run GCO-relax = state setter avec proba décroissante vers 0, puis fine-tune final en kickoff pur.
- Ne pas réutiliser leur métrique d'évaluation ; utiliser §13.

### 11.3 Autres
- "Rocket Learn" (Ibrahim, Stacy, Stroud, Zhang ; SMU Data Science Review vol. 5 n° 2 art. 12, 2021) : PPO jugé meilleur parmi les frameworks testés ; aucun chiffre dans l'abstract (abstract seulement).
- Survey MARL jeux vidéo, arXiv 2509.03682 (IEEE ToG 2025) : mentionne Rocket League, sans détails dans l'abstract.
- Aucun autre papier arXiv RL × Rocket League trouvé (recherche web, nov. 2026). Les bots communautaires (Necto/Nexto, Ripple…) sont documentés hors littérature (GitHub, W&B) → non couverts ici.

---

## 12. Tableau consolidé des hyperparamètres

| Param | Valeur(s) rapportée(s) | Source | Proposition RL-GCO | Justification |
|---|---|---|---|---|
| Fréquence de décision | 15/s (frame skip 8) ; Dota 7.5/s (frameskip 4 à 30 fps) | Lucy App. 5 ; Five §2 | 15/s (tick-skip 8) | Aligné communauté & RLBot 120 Hz |
| γ | 0.99 (WM §3.6, Pleines T.III, LOGO F) ; ≈0.995 (Lucy) ; 0.999 (PPG A.2) ; 0.99926–0.99963 [calcul depuis Five §4.5] ; 1 (AlphaStar) | voir col. 1 | **0.995** au départ, monter à **0.998** quand le bot marque régulièrement | GCO : seul signal au but ; [calcul] demi-vie 9.2 s (0.995) / 23 s (0.998) ; Five : horizon plus long = mieux |
| GAE λ | 0.95 (37 det., Five, Pleines, PPG) ; 0.9 (WM §3.4) | — | 0.95 ; tester 0.9 | Conflit WM vs reste ; WM continu seulement |
| Clip ε | 0.2 (37 MuJoCo, Five, PPG, Lucy, Pleines) ; 0.1 (Atari) ; 0.25 (WM) | — | 0.2 | Consensus ; surveiller approx_kl < 0.02 |
| lr | 3e-4 (WM, 37 MuJoCo, LOGO, Pleines début) ; 2.5e-4 Atari ; 1e-4 (Lucy) ; 5e-4 (PPG) ; 5e-5→5e-6 (Five) | — | 2e-4 → 5e-5 linéaire | Gros batch + self-play : prudence ; décroissance léger gain (WM §3.7) |
| Adam | β1 0.9 (WM, Five), β2 0.999 (Five), eps 1e-5 (37) | — | idem | Défauts éprouvés |
| Entropie | 0.01 (37 Atari, PPG, Five→0.001) ; 0 (Lucy, 37 MuJoCo) ; 5e-4→1e-5 (Pleines) | — | 0.01 → 0.002 sur le run | Sparse + 90 actions : besoin d'exploration ; WM sans gain mais en continu |
| Coef valeur | 0.5 (Lucy) ; 0.25 (Pleines) ; 0.25→1.0 (Five) | — | sans objet (réseaux séparés) ; sinon 0.5 | Interférence (PPG) |
| Réseaux | séparés > partagés 4/5 (WM) ; partagé critique en vision (PPG) ; séparés (Lucy) ; partagé (Pleines) | — | **Séparés**, politique 3×256, valeur 3×512 [proposition] | Obs vectorielle ; WM : valeur large |
| Init | orth. √2, sortie π 0.01, V 1 (37) ; dernière couche π ×1/100 (WM) | — | idem | Politique quasi uniforme au départ |
| Obs norm | crucial (WM §3.3) ; running mean/std (Five, 37) | — | Constantes fixes (échelles physiques) | Snapshots du pool cohérents |
| Value norm | mitigé (WM) | — | Off (retours ∈ [−1,1]) ; A/B plus tard | GCO borné |
| Reward scale/clip | important (Engstrom §3) ; Five : norm. par std | — | R = ±1, pas de scaling | Inutile sur signal binaire ; compatible GCO de toute façon |
| Value clipping | nuit (WM §3.4) ; pas de preuve (Engstrom) ; utilisé (Pleines, 37 fidélité) | — | **Off** | 2 sources contre |
| Grad clip | 0.5 (37, Engstrom, Pleines) | — | 0.5 | Consensus |
| Adv. norm | par minibatch (37) ; peu d'effet (WM) | — | On | Standard |
| Epochs | 3 (Pleines ; PPO optimal PPG App. D) ; 4 (37 Atari) ; 10 (Lucy ablation) ; E_π=1/E_aux=6 (PPG) | — | 2–3 ; recalcul avantages à chaque epoch (WM §3.5) | CPU limité : peu d'epochs |
| Minibatchs | 4 (37 Atari) ; 8 (Pleines, PPG) | — | 4–8 (minibatch 12.5–25k) | — |
| Batch / itération | 1024 (37 Atari 8×128) ; 2048–20000 (LOGO T.1) ; 8192 (Pleines) ; 65 536 (PPG 256×4×64) ; 320k rollout/4k batch (Lucy) ; 123k→983k, speedup 2.5× sous-linéaire (Five §4.3) ; communauté 100–200k (brief) | — | **50k–100k** selon débit mesuré [proposition] | Compromis : variance du gradient en sparse vs nb d'updates/h sur CPU |
| Envs parallèles | 256 (WM) ; 16 workers (Pleines) ; 10–20 instances (Lucy) | — | 12–16 processus sim (laisser 2–4 threads au learner) | 14C/18T |
| Timeout épisode | 5 min / 45 s sans touche / but (Lucy T.4) | — | kickoff→but, timeout ~45–60 s sans but = **troncature** (bootstrap) | Éviter un faux "0 terminal" |
| Self-play mix | 80/20 (Five) ; 35 SP/50 PFSP/15 forgotten (AlphaStar) | — | 75 % latest / 25 % pool | — |
| Snapshot | toutes les 10 itérations (Five) ; 2e9 pas (AlphaStar, exposant via DI-engine) | — | toutes les 10–20 itérations | Laptop : pool petit |
| Qualité pool | η = 0.01, q init = max (Five App. N) | — | idem | Simple |
| PFSP | f_hard=(1−x)^p (p non trouvé) ; f_var=x(1−x) | AlphaStar Methods | p = 2 [proposition] | À tester |
| Exploiters | seuil 70 % ; <20 % → f_var ; reset 25 % (league) ; timeouts 2e9/4e9 | AlphaStar Methods | 1 main exploiter intermittent | Budget CPU |
| Démo-guidage | δ0 0.008–0.2, α 0.9–0.95, K_δ 5–50 (LOGO T.1) ; λ0 0.1, λ1 0.95 (DAPG) ; ratio 1/256 (R2D3) | — | Track "GCO+demo" séparé seulement | Zone grise GCO |
| Action stacking | 5 dernières actions (Lucy §4.3) | — | 5 | Obs, compatible |
| Aux task | RP/SR poids 1 (Lucy) ; PPG N_π 32, E_aux 6, β_clone 1 | — | RP "but dans k pas" poids 0.5 | Compatible GCO |
| TrueSkill | μ 25, σ 25/3, β σ/2, τ σ/100, draw 0.1 (lib) ; τ 0, draw 0.02, 750 parties (Five) | trueskill.org ; Five App. J | τ = 0 pour références figées ; draw selon taux de timeouts observé | — |
| SPRT | α = β = 0.05 ; bornes [0,5]…[0,2] Elo (échecs) | CPW | α = β = 0.05, [0, +35] (voir §13) | Effets RL bien plus grands qu'en échecs |

---

## 13. Évaluation & promotion statistique — protocole recommandé

### 13.1 Unité statistique
- **Épisode-kickoff** : reset kickoff (5 positions RL, couleurs alternées), jusqu'au premier but ou timeout T_eval (ex. 60 s simulées). Résultat ∈ {victoire, défaite, nul=timeout}. [proposition] Plus de données/heure qu'un match de 5 min, et quasi-iid grâce au reset (dans un match complet, buts successifs corrélés).
- **Paires** : chaque position de kickoff jouée 2× en échangeant les couleurs/côtés (analogue des paires d'ouvertures fishtest → modèle pentanomial, CPW/fishtest wiki) ; réduit la variance due au spawn.
- Nuls (timeouts) : compter 0.5 (score) pour Elo, et les **rapporter** séparément ; si > 20 %, allonger T_eval.

### 13.2 Système de rating
- Hors ligne : **Bradley–Terry / Elo MLE** sur la matrice de résultats (tous snapshots + références), ancre fixe : **agent aléatoire = 0** (comme Five) ; IC par **bootstrap** (≥ 1000 rééchantillonnages des épisodes).
- Références figées : snapshots du main (tous les ~N itérations), bots scriptés (ex. bot "ATBA"/ballchaser), et si accessibles sous RLBot v5 : Necto/Nexto (éval uniquement — **compatible GCO**, ils ne fournissent aucune récompense).
- En ligne (matchmaking pendant training) : TrueSkill façon Five (μ des références figé, τ = 0, ne jouer que les adversaires avec win-rate attendu 15–85 %).
- Garde-fou non-transitivité : rapporter aussi le **min win-rate vs pool** (proxy d'oubli, AlphaStar Fig. 3D) et la distribution de Nash du pool (AlphaStar Fig. 4C).

### 13.3 Formules [calcul, formules standard]
- p(d) = 1/(1 + 10^{−d/400}) ; d(p) = −400·log10(1/p − 1).
- Test binomial unilatéral H0 : p = 0.5, H1 : p = p1, taille fixe :
  **n = [ (z_{1−α}·sqrt(p0(1−p0)) + z_{1−β}·sqrt(p1(1−p1))) / (p1 − p0) ]²**
- Demi-largeur IC 95 % sur p à p = 0.5 : n = 1.96²·0.25/h² → h = 0.10 : 97 ; 0.05 : 385 ; 0.03 : 1068 ; 0.02 : 2401.
- SE Elo ≈ 173.7 / sqrt(n·p(1−p)) → n = 100 : ±68 Elo (IC95) ; 400 : ±34 ; 1000 : ±21.5 ; 2000 : ±15.2 (à p = 0.5).
- SPRT Bernoulli (sans nuls) : incrément par victoire w = ln(p1/p0), par défaite l = ln((1−p1)/(1−p0)) ; arrêt à S ≥ ln((1−β)/α) = **+2.944** (accepter H1) ou S ≤ ln(β/(1−α)) = **−2.944** (accepter H0) pour α = β = 0.05 ; E[N|H1] ≈ [(1−β)·b + β·a]/KL(p1‖p0) (approximation de Wald).

### 13.4 Nombres d'épisodes [calcul ; SPRT vérifié par Monte-Carlo 4000 réplications]

| Effet visé (Elo) | p1 | n fixe α=0.05, puissance 0.8 | n fixe α=0.05, puissance 0.9 | n fixe α=0.01, p 0.9 | SPRT α=β=0.05 E[N] sous H0/H1 | SPRT p95(N) | SPRT si vrai effet = moitié |
|---|---|---|---|---|---|---|---|
| +20 | 0.529 | 1868 | 2587 | 3933 | — | — | — |
| +35 | 0.550 | 612 | 846 | 1287 | ≈523 / 525 (MC : 532 / 543) | ≈1285 | moyenne 879, p95 2335 |
| +50 | 0.572 | 301 | 416 | 633 | ≈257 / 259 | ≈633 | 435, p95 1147 |
| +70 | 0.599 | 155 | 213 | 325 | ≈131 / 133 | ≈330 | 228, p95 610 |
| +100 | 0.640 | 77 | 106 | 162 | ≈65 / 67 | ≈167 | 115, p95 299 |

(MC : P(accepter H1) mesurée 0.043–0.050 sous H0 et 0.950–0.957 sous H1 → erreurs conformes à α = β = 0.05.)

### 13.5 Protocole concret de promotion (laptop)
1. **Gate rapide** (chaque snapshot candidat C vs champion actuel K) : SPRT H0 Elo = 0 vs H1 Elo = **+50**, α = β = 0.05, épisodes-kickoff appariés ; cap dur à 1200 épisodes (≈ p95) → si non conclu, "pas de promotion". Coût moyen ≈ 260 épisodes.
2. **Non-régression vs pool** (si gate passé) : 200 épisodes répartis sur les 10 dernières références → win-rate global avec IC Wilson ; échec si borne haute IC95 < 0.5 vs n'importe quel ancien champion **et** win-rate < 0.4 (détection d'oubli/cycle).
3. **Rating global** (toutes les ~24 h) : BT-MLE + bootstrap sur toute la matrice ; publier μ, IC95, min win-rate vs pool.
4. **Validation RLBot v5** (toutes les promotions majeures) : 100 épisodes-kickoff dans le vrai jeu (temps réel, 120 Hz) vs la même référence qu'en sim → IC95 ±0.10 sur p ; alerte si |p_sim − p_RLBot| > 0.10 (sim-to-sim gap : Pleines Fig. 6). Durée = 100 × durée moyenne d'épisode en temps réel (à mesurer).
5. Multiplicité : chaque gate a α = 0.05 ; sur 20 candidats nuls, ~1 fausse promotion attendue [calcul : 20 × 0.05] → l'étape 2 et le rating bootstrap servent de second filtre ; pour les décisions de "release", utiliser α = 0.01.
6. Coût temps : heures = N_épisodes × durée_moyenne_épisode_simulée / (accélération_sim × envs_parallèles × 3600) — à instancier après mesure du débit RocketSim sur ce CPU (non mesuré ici).

---

## 14. Conflits et incertitudes entre sources

- **Identité 2202.04628** : LOGO (sparse + démos), pas un papier self-play/PPO à grande échelle (§0).
- **λ GAE** : 0.9 (WM §3.4) vs 0.95 (37 det., Five Table 2, Pleines T.III, PPG A.2).
- **Clip** : 0.25 (WM) vs 0.2 (la plupart) vs 0.1 (Atari, 37).
- **Value clipping** : nuisible (WM §3.4), sans preuve (Engstrom), mais utilisé (Pleines Eq. 2–3 ; 37 pour fidélité baselines).
- **Partagé vs séparé** : séparés meilleurs (WM 4/5 MuJoCo ; Lucy) vs partage critique en Procgen-vision (PPG App. B) vs partagé (Pleines).
- **Entropie** : pas de gain (WM, continu) vs 0.01 (PPG, Five, Atari) vs 0 (Lucy).
- **AlphaStar "récompense terminale"** : vrai pour la récompense de victoire, mais pseudo-récompenses humaines (proba 25 %) + KL supervisé → pas GCO-pur.
- **Exposants AlphaStar** : PDF extrait "2 × 10s" / "4 × 10s" (glyphe perdu) ; DI-engine (réimplémentation) : 2e9 (snapshot main, timeout league exploiter) ; une copie miroir du code DI-engine indique 4e9 pour le main exploiter. Valeurs 10^9 = **probables, non vérifiées sur Nature**.
- **Exposant p de f_hard** : non trouvé.
- **Formule d'Elo AlphaStar** : lue "1/(1+e^{−(r1−r2)/400})" (garbled) ; convention standard base 10. À vérifier dans Nature.
- **TrueSkill nuls/dynamique** : lib draw 0.10, τ = σ/100 ; Five draw 0.02, τ = 0.
- **TrueSkill "1v1 = 12 parties"** : ligne "2 Players FFA" de la page MSR ; "jusqu'à 3× plus" selon la même page.
- **R2D3 auteurs** : LOGO cite "Gülçehre et al. ICLR 2020" ; arXiv 1909.01387 liste Paine en premier.
- **Lucy-SKG** : protocole d'éval ambigu (scores 300-54 sur "300 matchs single-goal" ; "Win %" non standard ; pas d'IC) ; hyperparam "batch 4000" vs "320 000 rollout steps" (minibatch vs rollout, non explicité).
- **Pleines Table III** : colonnes désalignées à l'extraction ; clip 0.2 et c1 0.25 = mon interprétation.
- **PPG Table A.2** : en-têtes γ/λ perdus ; valeurs .999/.95 attribuées par ordre.

---

## 15. Récapitulatif compatibilité GCO

| Technique | GCO |
|---|---|
| PFSP, pool à qualité, exploiters, snapshots, références d'éval (Necto/Nexto en éval) | **Compatible** |
| γ/λ/batch/epochs/init/normalisation obs/grad clip/timeout-troncature | **Compatible** |
| Tâches auxiliaires (RP "but bientôt", PPG aux), action stacking | **Compatible** (aucune récompense ajoutée) |
| Self-imitation sur ses propres buts (PPO+D sans démo, UPGO) | **Compatible** (n'utilise que ±R) |
| Scaling constant de R | Compatible (inutile) |
| Masquage d'actions invalides | Zone grise mineure (prior de domaine) |
| State setter (replays, scénarios), curriculum de scénarios | **Zone grise** (distribution initiale ≠ récompense) |
| BC warm-start, KL vers modèle supervisé, guidage LOGO/DAPG, démo ratio R2D3 | **Zone grise** (prior de politique ; π_b issue de shaping) |
| Discriminateur GAIL/POfD comme récompense, pseudo-récompenses AlphaStar, KRC Lucy, rewards Necto/Nexto, team spirit Five | **Non compatible** |

---

## 16. Références (vérifiées via pages abs/PDF consultées)

- Rengarajan, Vaidya, Sarvesh, Kalathil, Shakkottai. *Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration.* ICLR 2022. **arXiv:2202.04628**. Code : github.com/DesikRengarajan/LOGO.
- Vinyals et al. *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature 575:350–354, 2019 (PDF DeepMind "AlphaStar_unformatted.pdf"). Pas d'arXiv.
- OpenAI (Berner et al.). *Dota 2 with Large Scale Deep Reinforcement Learning.* **arXiv:1912.06680**.
- Andrychowicz et al. *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study.* **arXiv:2006.05990**.
- Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry. *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO.* ICLR 2020. **arXiv:2005.12729**.
- Huang et al. (liste complète des auteurs non vérifiée). *The 37 Implementation Details of Proximal Policy Optimization.* ICLR Blog Track 2022, iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.
- Cobbe, Hilton, Klimov, Schulman. *Phasic Policy Gradient.* **arXiv:2009.04416**.
- Herbrich, Minka, Graepel. *TrueSkill™: A Bayesian Skill Rating System.* NeurIPS 2006 (papier non lu ; MSR project page + trueskill.org consultés).
- Wald, SPRT (via Wikipedia "Sequential probability ratio test") ; chessprogramming.org "Sequential Probability Ratio Test" ; github.com/official-stockfish/fishtest/wiki/Fishtest-mathematics.
- Pleines et al. *On the Verge of Solving Rocket League using Deep Reinforcement Learning and Sim-to-sim Transfer.* IEEE CoG 2022. **arXiv:2205.05061**.
- Moschopoulos, Kyriakidis, Lazaridis, Vlahavas. *Lucy-SKG: Learning to Play Rocket League Efficiently Using Deep Reinforcement Learning.* **arXiv:2305.15801**.
- Ibrahim, Stacy, Stroud, Zhang. *Rocket Learn.* SMU Data Science Review 5(2), art. 12, 2021 (abstract).
- Li, Ji, Ling, Liu. *A Comprehensive Review of Multi-Agent RL in Video Games.* **arXiv:2509.03682** (abstract).
- Rajeswaran et al. *Learning Complex Dexterous Manipulation with Deep RL and Demonstrations (DAPG).* RSS 2018. **arXiv:1709.10087**.
- Paine, Gülçehre et al. *Making Efficient Use of Demonstrations to Solve Hard Exploration Problems (R2D3).* **arXiv:1909.01387**.
- Libardi, De Fabritiis. *Guided Exploration with PPO using a Single Demonstration.* ICML 2021. **arXiv:2007.03328**.
- Kang, Jie, Feng. *Policy Optimization with Demonstrations (POfD).* ICML 2018, PMLR 80:2469–2478.
- Hester et al. *Deep Q-learning from Demonstrations.* AAAI 2018. **arXiv:1704.03732**.
- Nair, McGrew, Andrychowicz, Zaremba, Abbeel. *Overcoming Exploration in RL with Demonstrations.* ICRA 2018. **arXiv:1709.10089**.
- Nair, Gupta, Dalal, Levine. *AWAC.* **arXiv:2006.09359**.
- Cités par LOGO mais non relus ici (aucun chiffre utilisé) : TRPO (Schulman 2015, arXiv:1502.05477), CPO (Achiam 2017, arXiv:1705.10528), GAIL (Ho & Ermon 2016, arXiv:1606.03476), Vecerik 2017 (arXiv:1707.08817), Kakade & Langford 2002.
