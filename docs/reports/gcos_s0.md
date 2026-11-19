# Rapport : GCO-S graine 0 (1B, protocole S1 v2, ADR-006)

- Run : `run_4a_s_s0`, `scripts/run_4a.py --variant s --seed 0 --max-hours 0 --eval-config run_s1v2_eval.yaml
  --std-s1-observed`, commit f2218a6 (code identique à 1b98a0d, smoke PASS), laptop Core Ultra 5 125H + Arc (XPU).
- Lancé par l'utilisateur le 2026-11-19 à 00:07 ; dernier checkpoint (itération 19 376, 1 000 014 736 pas appris) ≈ 5,57 h
  plus tard (horodatages de `runs/run_4a_s_s0/events.jsonl`). Une perte du GPU (`UR_RESULT_ERROR_DEVICE_LOST`, itération
  10 632), reprise automatique depuis l'itération 10 500 (`exact_env=True`). Aucun STOP, aucune restauration K4.

## Évaluation en cours de run (S1 v2)
| Itération | ≈ pas | (a) SPRT | (b) Garwood vs θ₀ |
|---:|---:|---|---|
| 1 938 | 0,1B | non | non |
| 3 876 | 0,2B | oui | oui |
| 5 813 | 0,3B | oui | oui |
| 7 751 à 19 376 | 0,4B à 1B | oui (8/8) | oui (8/8) |

- `s1_reached` à la 2e paire consécutive (événement à l'itération 6 001). **S1-seed officiel = 0,2B** (1er point, ADR-006).
- K7 jamais déclenché.

## Mesures hors ligne (2026-11-19, `.venv`, CPU, `runs/run_4a_offline_eval/`)
Commande : `scripts/eval.py --a <instantané> --b random --episodes 20000 --n-envs 256 --seed <g> --sprt-elo 100
--sprt-max-decisive 1200`, commit f2218a6, arbre propre.

| Fichier | Graine | V / N / D | Victoires / épisode [Wilson 95 %] | Décisives [Wilson 95 %] | SPRT +100 | A touche | Aucune touche |
|---|---:|---|---|---|---|---:|---:|
| `gcos_s0_s1confirm_it03876_seed1` (0,2B) | 1 | 361 / 19 487 / 152 | 1,81 % [1,63 ; 2,00] | 2,56 % [2,35 ; 2,79] | H1 à 876 ép. | 0,112 | 0,834 |
| `gcos_s0_final20k_it19376` (1B) | 0 | 861 / 18 768 / 371 | 4,31 % [4,03 ; 4,60] | 6,16 % [5,84 ; 6,50] | H1 à 478 ép. | 0,146 | 0,803 |
| Référence std s0 `s0_final20k_it19376` (1B, commit 6db79a4) | 0 | 479 / 19 372 / 149 | 2,40 % [2,19 ; 2,62] | 3,14 % [2,91 ; 3,39] | H1 à 652 ép. | 0,139 | 0,810 |

Wilson calculé à partir des comptes ci-dessus (z = 1,96).

## Lecture
- **Confirmation hors ligne du S1-seed : oui** (SPRT accepte H1 sur un échantillon indépendant, graine 1).
- **Règle D5 (ADR-006)** : « S < std » si la borne haute de GCO-S est sous 1,60 % de victoires par épisode. Borne haute
  mesurée 4,60 % : **S ≥ std**, l'intervention désignée est donc **(2) γ 0,998 (ADR-007)**.
- À 1B, une seule graine : GCO-S gagne ≈ 1,8 × plus d'épisodes que std s0 (4,31 % contre 2,40 %, IC disjoints). Le taux de
  touche reste du même ordre (14,6 % contre 13,9 %) : les états génériques de std n'ont pas amélioré le contact au kickoff.
  Une graine ne suffit pas à conclure sur les variantes (ADR-005 : médiane sur 3 graines).
- La part de décisives à 1B (borne basse 5,84 %) reste sous le seuil de succès d'une intervention (6,28 %, ADR-006 D6).
