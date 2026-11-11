# PLAN d'ensemble (v0.2, Gate G1 validé le 2026-11-10)

Principe directeur : **mesurer avant de construire**. Les deux risques dominants sont, dans l'ordre :
1. **R1 — l'amorçage de l'exploration en GCO à petit budget** : aucune preuve publique d'un GCO qui marque sous 0,35B steps.
2. **R2 — le débit disponible** : à la rédaction (G1), laptop 3k à 25k SPS estimés et rien de mesuré. Depuis la Phase 2,
   voir `docs/adr/001-infra.md` (prototype ≈ 78–99k timesteps/s sur le laptop, Kaggle T4 = 4 vCPU, TPU en attente de sonde).

Le plan attaque R2 en Phase 2 et R1 dès le premier run court (Phase 4a), avant tout gros calcul.

| Phase | Contenu | Livrable | Critère de sortie | Durée estimée |
|---|---|---|---|---|
| **G1** | validation de la spec, du budget et du plan | réponses aux questions | GO humain | — |
| **1 Studio** | `.claude/` (CLAUDE.md < 200 lignes, 17 agents à descriptions courtes, 10 skills, hooks fail-closed, rules path-scopées), `git init`, `production/state.md` | config studio + ADR-000 | hooks testés (déclenchement prouvé), `/status` fonctionnel | 0,5–1 j |
| **2 Infra** | **sonde Kaggle d'abord** (`docs/research/kaggle.md` §4 : cœurs hôte TPU v5e-8 et T4, RAM, versions, micro-bench RocketSim) ; venv `uv` ; A1 = rlgym-learn + RLGym `main` (numpy 2) + rocketsim 2.2.1 ; B = chemins chauds natifs (obs/collecte) avec PPO et reward en Python ; C1 = GigaLearn en **plafond de référence uniquement** si le build est faisable sans copier son code ; benchmarks ≥ 30 min (throttling) : SPS collecte / apprentissage / bout-en-bout, scaling par workers (P vs E cores), RAM, torch CPU vs XPU ; learner JAX/TPU vs torch CPU hôte ; orchestration de sessions chaînées ≤ 9 h (push / reprise / output) | `docs/adr/001-infra.md` | décision mesurée + estimation de temps de calcul | 2–5 j |
| **G2** | décision d'infra + temps de calcul | — | GO humain | — |
| **3 Implémentation** | `src/{env,obs,actions,rewards,models,training,eval,rlbot_bot}` ; reward GCO + tests §7 de la spec ; parité obs/actions via rlgym-compat ; PPO v0 (`01_papers.md` §2) ; pool self-play à qualité ; checkpoints atomiques + reprise exacte ; garde NaN/divergence ; TensorBoard | code + tests verts | smoke test 1M steps sans NaN, test de parité vert | 4–8 j |
| **4a Validation** | smoke test → run court (≈ 0,3–1B steps, 3 seeds) : GCO-S **et** GCO-std en parallèle | courbes sanity + `/postmortem` | **S1 atteint** sur ≥ 1 seed en GCO-std, sinon kill + diagnostic (§ kill criteria) | 1–4 j de calcul |
| **G3** | config du run long, critères de succès et d'arrêt | — | GO humain | — |
| **4b Run long** | sessions T4 en relais reprises depuis checkpoint (chaque compte opéré par son titulaire), TPU et laptop en appoint, jalons 1B / 3B / 10B / … avec rapport à chaque jalon | checkpoints promus | S2, puis S3, puis S4 | semaines |
| **5 Déploiement** | export TorchScript/ONNX, latence < 8,33 ms/tick vérifiée, bot RLBot v5 (`bot.toml`), matchs humain vs bot et vs botpack ; écart sim/jeu mesuré | bot jouable + dashboard d'éval | promotion par SPRT seulement | 2–4 j |
| **6 Boucle** | postmortem, `state.md`, re-priorisation | — | — | continu |

Note du 2026-11-11 : l'option retenue en Phase 2 est A3 (RocketGPU CPU + env maison batché), pas A1 ; la parité
obs/actions de la Phase 3 ne passe donc plus forcément par rlgym-compat (outil à fixer en Phase 3). Le relais T4 de la
phase 4b reste conditionné aux sondes (T4 faible pour la simulation, TPU en attente).

## Kill criteria préliminaires (affinés au G3)
- **Run court GCO-std** : aucun but contre l'adversaire gelé après **1B steps** sur 3 seeds ⇒ arrêt. Diagnostic : exploration,
  distribution d'états, horizon. Une seule intervention à la fois.
- Entropie < 10 % de ln(90) avant S1 ⇒ effondrement de l'exploration ⇒ arrêt.
- explained variance du critique < 0 pendant 50 itérations après S1 ⇒ arrêt.
- NaN/Inf, ou KL approx > 0,05 pendant 5 itérations ⇒ arrêt automatique et restauration du dernier checkpoint sain.

## Priorités R&D GCO (gain attendu / coût), à tester une par une en Phase 4a
1. États initiaux génériques annealés vers le kickoff.
2. γ 0,995 → 0,998.
3. Self-imitation sur nos propres buts.
4. Tâche auxiliaire « prochain but ».
5. Composition critic (spec publique, bugs connus, surcoût SPS à mesurer).
6. World model / nouveauté (pas de détail public ; veille sur les publications post-RLBC de Pulsar et de Terminal).
