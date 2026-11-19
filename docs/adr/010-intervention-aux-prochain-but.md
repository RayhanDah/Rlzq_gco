# ADR-010 — Intervention (4) : tâche auxiliaire « prochain but » (esquisse, GCO-std, graine 0, 1B)

- Statut : **proposé (esquisse)** : assez précis pour être implémenté ; λ_aux est fixé par la mesure de D4 avant le GO.
  Validation utilisateur requise avant tout lancement.
- Date : 2026-11-18
- Décideurs : research-director (proposition) ; rl-algo-lead (implémentation) ; utilisateur (validation, lancement)

## Contexte
- Intervention (4) de `docs/GCO_SPEC.md` §8 et d'ADR-005 D6 ; autorisée en GCO-std (§3, « Tâches auxiliaires (prédire le prochain
  but) » : « perte de représentation, pas récompense »).
- Le critique apprend déjà E[γ^τ·(±1)] (EV 0,96 dès 0,4B, `docs/reports/run_4a.md` §8), mais politique et critique sont **séparés**
  (`src/models/policy.py:17` : π 3×256 ; `src/models/value.py:15` : V 3×512). Rien ne force la représentation de la **politique** à
  encoder ce qui mène à un but. Hypothèse : c'est une partie de la lenteur observée (politique quasi uniforme, 81 à 86 % d'épisodes
  sans touche).
- Sources : Lucy-SKG (arXiv 2305.15801, lu via ar5iv le 2026-11-18) : tâche « reward prediction » à 3 classes (positif, négatif,
  ≈ 0) branchée sur les couches de l'acteur (§4.2, §4.5) ; récompense moyenne en plateau vers 100M contre 200M sans (§5.1, Fig. 3),
  **sur une récompense dense**, poids non publié. UNREAL (Jaderberg et al., arXiv 1611.05397, lu via ar5iv le 2026-11-18) :
  entropie croisée à 3 classes (§3.2), rejeu rééquilibré P(r ≠ 0) = 0,5 (§3.2–3.3), λ_RP = 1 (Annexe B).

## Options
| Option | Pour | Contre |
|---|---|---|
| **Classe du prochain but à horizon H (+1 / −1 / aucun), tête sur le tronc de π** | cible tirée du seul ±1 ; analogue GCO de Lucy et UNREAL | classes très déséquilibrées ; étiquette connue seulement H pas plus tard |
| Récompense immédiate (γ = 0, Lucy/UNREAL à la lettre) | fidèle aux sources | en GCO, r ≠ 0 au seul pas du but : ≈ 1 pas sur 300, signal quasi nul |
| Tête sur le critique | aucun effet sur π | redondant avec V, déjà bon (EV 0,96) |
| Régression du retour discounté sur le tronc de π | continue | duplique V ; échelle à régler |

## Mesures
Aucune mesure : esquisse réversible, fondée sur les sources ci-dessus et le rapport 4a. Chiffres calculés : H = 200 pas = 13,3 s
(= 1/(1−γ) à γ = 0,995, ADR-007). Taux d'étiquettes non nulles **estimé, non mesuré** : buts / fins d'épisode 4,6 à 7,8 %
(annexe A de 4a) × au plus 200 pas étiquetés par but sur ≈ 305 pas par épisode, soit quelques % des pas.

## Décision (esquisse)
- **Cible** : pour l'agent appris i au pas t, y_t ∈ {+1 : l'équipe de i marque, −1 : elle encaisse, 0 : aucun but} dans les
  **H = 200** pas suivants. Définie par le seul flux de récompense §2.1 (csc compris) et les drapeaux de fin. **Masquée** si
  l'épisode est tronqué avant t + H sans but (le futur est inconnu ; une troncature n'est pas une issue, `GCO_SPEC.md` §2.2).
- **Tête** : linéaire 256 → 3 sur la dernière couche cachée de la politique ; module séparé, **hors** des poids de politique
  exportés (éval `src/eval/match.py` et RLBot inchangés). Le critique ne la voit pas.
- **Perte** : entropie croisée à 3 classes (UNREAL §3.2), ajoutée à la perte de chaque minibatch PPO : L = L_PPO + λ_aux·L_aux.
  Lot auxiliaire de 8 192 transitions tiré d'un rejeu FIFO de 250 000 transitions étiquetées, **rééquilibré** : 50 % y = 0, 50 %
  y ≠ 0 (UNREAL §3.2–3.3). Les deux camps appris et les épisodes tronqués (pas étiquetés 0 ou masqués) y entrent.
- **Infrastructure** : réutilise la fenêtre par agent d'ADR-009 D1 (≈ 0,54 Go) ; rejeu ≈ 58 Mo (calcul d'ADR-009, non mesuré).
- **Poids λ_aux, fixé par mesure, pas deviné** : UNREAL prend 1 avec d'autres échelles de perte, Lucy ne le publie pas, et nos normes
  de gradient de politique sont minuscules (0,03 à 0,09, annexe A de 4a). Banc ≤ 30 min (smoke + `scripts/bench_run.py`) avec
  λ ∈ {0,01 ; 0,03 ; 0,1} : retenir le plus grand λ tel que la part médiane ‖g_aux‖ / (‖g_aux‖ + ‖g_PPO‖) sur le tronc de π
  reste ≤ 0,5 (seuil C4). Résultat consigné dans cet ADR avant le GO. λ constant sur le run de 1B.
- **C1** : cible issue du ±1 et des drapeaux `terminated` / `truncated` ; aucune notion du jeu. **C2** : pas de récompense, pas de
  terme dans les avantages ; la tête ne change que la représentation partagée. Pour tout run long, λ_aux annealé à 0 avant la phase
  finale (lecture prudente de C2) ; constant dans ce run court. **C3** : `aux.enabled` (défaut `false`), désactivé ⇒ trajectoire
  identique à std (test). **C4** : `aux/loss`, rappel et précision par classe, fréquences d'étiquettes, part de gradient ci-dessus
  à chaque itération ; alerte si > 0,5 pendant 50 itérations.
- **Protocole** : `int_aux_s0`, graine 0, 1B, config = `configs/run_4a_std.yaml` + bloc `aux` ; éval v2 en cours de run (ADR-006
  D2, D3) ; comparaison hors ligne identique à ADR-009 D8 (`runs/int_aux_offline_eval/`) ; **succès = ADR-006 D6** (6,28 % et SPRT
  à 1B, ou 1re paire SPRT ≤ 0,2B) ; neutre sinon. Kill criteria : K2 à K6 (ADR-005/006) et KS1 d'ADR-009 (RAM).
- **Place** : après la SIL et l'intervention (1) ou (2) (ADR-009 D9). Combinaison SIL + aux seulement après deux résultats isolés.

## Conséquences
- Peu coûteux en calcul (une tête linéaire, un lot de plus par minibatch : coût estimé, non mesuré) ; partage l'infrastructure SIL.
- Le seul gain publié (Lucy, ×2 en échantillons) est sur une récompense dense : transfert au GCO non démontré.
- Révision si : rappel des classes ± ≤ hasard après 0,2B (la tête n'apprend rien) ; part de gradient < 0,05 au λ retenu.

## Points à valider par l'utilisateur
1. Cible à 3 classes à horizon 200 pas, masquée aux troncatures ; tête sur le tronc de π, hors des poids exportés.
2. λ_aux fixé par la mesure du banc (règle de la part de gradient ≤ 0,5), constant sur 1B.
3. Protocole et critère d'ADR-006 D6 ; place après la SIL.
