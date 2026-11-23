# Rapport : `int_aux_s0` (ADR-010, GCO-std + tâche auxiliaire « prochain but », graine 0, 1B)

- Date : 2026-11-23. Rédigé par l'orchestrateur. Verdict appliqué par script (`aux_run_chain.py`, scratchpad), seuils d'ADR-010
  D10 recopiés tels quels ; même calcul que pour ADR-007 et la SIL.
- Run : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant int_aux --seed 0 --max-hours 0`, commit 0d930e6.
  - Smokes `configs/smoke.yaml --smoke` et `configs/smoke_aux.yaml --smoke` (+ `aux_bench_summary.py --smoke`) PASS sur le même
    commit (`runs/smoke_0d930e6`, `runs/smoke_aux_0d930e6`).
  - Laptop Core Ultra 5 125H + Arc (XPU), torch 2.15.0+xpu.
  - Début le 2026-11-23 à 01 h 07, fin à 06 h 43 (5,59 h).
  - Config lue dans le run : λ_aux = 0,1 constant, H = 100, `store_float16: false`, γ 0,995.
- θ₀ identique à std s0 : SHA-256 7fab4c747a778f6a… de l'instantané de l'itération 0 (même méthode qu'ADR-007).
- Conventions : m = `runs/int_aux_s0/metrics.jsonl` (19 376 lignes), g = `segments.jsonl`, v = `events.jsonl`. Évals hors ligne
  dans `runs/int_aux_offline_eval/`, commit 0d930e6, `.venv` CPU, `scripts/eval.py --b random --n-envs 256 --seed 0 --sprt-elo 100
  --sprt-max-decisive 1200`. Wilson 95 % (z = 1,959964).

## 1. Verdict ADR-006 D6 (ADR-010 D10) : **NEUTRE**
| Volet | Règle | Mesure | Résultat |
|---|---|---|---|
| (i) | 1B, 20 000 épisodes `--seed 0` : Wilson bas des décisives ≥ 6,28 % **et** SPRT H1 | 586 / 20 000 = 2,93 % [2,71 ; 3,17], SPRT H1 | non rempli |
| (ii) | 1re paire consécutive H1 (2 048 épisodes) au plus tard à 0,2B | 1re paire (0,8B ; 0,9B) | non rempli |

Référence std s0 : 3,14 % à 1B, 1re paire à 0,5B (`docs/reports/run_4a.md`). Écart à 1B : −0,21 point, dans le bruit d'une graine
(les intervalles se recouvrent). Élo décisif final +221 [+188 ; +258] (458 V / 128 D). La tâche auxiliaire n'avance pas
l'apprentissage ; la 1re paire H1 arrive plus tard que std (0,8B contre 0,5B), descriptif seulement.

## 2. Trajectoire hors ligne (2 048 épisodes `--seed 0` contre l'aléatoire, descriptif)
| B | aux : décisives (%) | SPRT | touche (%) | std s0 : décisives | SPRT | touche |
|---|---|---|---|---|---|---|
| 0,0 | 1,32 | plafond | 6,6 | 1,32 | plafond | 6,6 |
| 0,1 | 1,61 | plafond | 5,5 | 1,76 | H1 | 7,2 |
| 0,2 | 1,56 | plafond | 6,4 | 1,46 | plafond | 7,3 |
| 0,3 | 1,61 | H1 | 6,6 | 1,51 | plafond | 7,0 |
| 0,4 | 1,86 | plafond | 7,8 | 1,81 | plafond | 9,6 |
| 0,5 | 1,81 | plafond | 7,8 | 2,00 | H1 | 8,1 |
| 0,6 | 2,34 | H1 | 9,7 | 1,71 | H1 | 9,0 |
| 0,7 | 1,32 | plafond | 7,9 | 2,34 | H1 | 10,2 |
| 0,8 | 3,32 | H1 | 13,5 | 2,69 | H1 | 10,8 |
| 0,9 | 2,88 | H1 | 12,5 | 2,59 | H1 | 11,3 |
| 1,0 | 3,12 | H1 | 12,2 | 2,64 | H1 | 13,9 |
| **1,0 (20 000)** | **2,93 [2,71 ; 3,17]** | H1 | 12,4 | 3,14 | H1 | — |

- Le point 0,0 est identique à std s0 (même θ₀, même graine d'éval). La touche reste au niveau de std (12,4 % à 1B) : R1 inchangé.

## 3. Diagnostics (m, moyennes par tranche de 0,1B, aux / std s0)
| Tranche | EV | perte de valeur | ‖g‖ | KL | entropie | buts/it. |
|---|---|---|---|---|---|---|
| 0–0,1 | 0,822 / 0,821 | 3,97·10⁻⁴ / 3,94·10⁻⁴ | 0,225 / 0,082 | 1,26·10⁻³ / 6,6·10⁻⁴ | 4,392 / 4,457 | 7,44 / 7,38 |
| 0,2–0,3 | 0,949 / 0,942 | 1,82·10⁻⁴ / 1,89·10⁻⁴ | 0,214 / 0,046 | 2,4·10⁻³ / 1,3·10⁻⁴ | 4,017 / 4,485 | 5,47 / 5,43 |
| 0,4–0,5 | 0,969 / 0,961 | 1,16·10⁻⁴ / 1,30·10⁻⁴ | 0,169 / 0,057 | 1,8·10⁻³ / 2,3·10⁻⁴ | 3,994 / 4,455 | 4,18 / 4,20 |
| 0,6–0,7 | 0,971 / 0,962 | 1,10·10⁻⁴ / 1,28·10⁻⁴ | 0,200 / 0,070 | 1,5·10⁻³ / 2,3·10⁻⁴ | 3,893 / 4,434 | 4,03 / 4,14 |
| 0,8–0,9 | 0,969 / 0,960 | 1,34·10⁻⁴ / 1,53·10⁻⁴ | 0,222 / 0,083 | 1,4·10⁻³ / 1,8·10⁻⁴ | 3,510 / 4,388 | 4,68 / 4,83 |
| 0,9–1,0 | 0,969 / 0,959 | 1,34·10⁻⁴ / 1,62·10⁻⁴ | 0,236 / 0,092 | 1,3·10⁻³ / 1,4·10⁻⁴ | 3,354 / 4,357 | 4,78 / 5,04 |

| Tranche | grad_share | disp_share | perte aux | rappel + | précision + | rappel − | part classe 0 |
|---|---|---|---|---|---|---|---|
| 0–0,1 | 0,485 | 0,439 | 0,146 | 0,950 | 0,934 | 0,950 | 0,687 |
| 0,2–0,3 | 0,338 | 0,391 | 0,083 | 0,965 | 0,980 | 0,964 | 0,674 |
| 0,4–0,5 | 0,375 | 0,378 | 0,115 | 0,935 | 0,969 | 0,935 | 0,676 |
| 0,6–0,7 | 0,379 | 0,362 | 0,138 | 0,920 | 0,955 | 0,920 | 0,675 |
| 0,8–0,9 | 0,392 | 0,368 | 0,154 | 0,912 | 0,936 | 0,912 | 0,673 |
| 0,9–1,0 | 0,376 | 0,365 | 0,152 | 0,915 | 0,935 | 0,915 | 0,672 |

- La tâche est apprise : rappels des classes ± ≥ 0,91 sur tout le run (clause de révision « ≤ 1/3 après 0,2B » non déclenchée).
  grad_share médiane 0,385 sur le run (0,474 au banc float32) ; disp_share médiane 0,383.
- Effet sur PPO : l'entropie de π baisse nettement plus que chez std (3,35 contre 4,36 en fin de run ; minimum 3,13) et la KL
  approchée est 6 à 18 fois celle de std. ‖g‖ est 2,6 à 4,6 fois celle de std. La représentation partagée bouge donc π, sans
  gain sur les buts. Lecture possible (non testée) : la tâche « prochain but » est déjà résolue par le critique (EV ≈ 0,97),
  elle n'apporte pas de signal nouveau sur le contact, qui reste le goulot (R1).
- KL > 0,05 sur 2 itérations (au plus 2 consécutives, K4 exige 5), max 0,066 : 0 restauration.
- RAM : disponible min 1 896 Mo (médiane 3 813 Mo), KS1 non déclenché ; mémoire privée max 7 427 Mo. Fenêtre 321 521 784 o, rejeu
  110 250 000 o, comme au banc.
- Débit : 1 000 014 736 pas appris en 20 111 s, soit 49,7k appris/s en moyenne murale ; 59,1k collectés/s (g). Les fenêtres de
  30 min de KS4 n'ont pas été recalculées ; la moyenne est au-dessus de 40k.
- Santé (g, v) : 1 segment `completed` 0 → 19 376, **0 perte du GPU**, 0 restauration, 0 NaN, aucun STOP. `s1_reached` à
  l'itération 11 751 (protocole v2, en cours de run, jamais comparé). Checkpoints ≈ 658 Mo, écriture ≈ 3 s.

## 4. Limites
Une graine. Un neutre ne réfute pas la tâche (ADR-010 Conséquences). Base std s0 d'une seule graine ; évals de std s0 sur un autre
commit, équivalence contrôlée sur le seul point θ₀. Lecture de l'entropie et de la KL : descriptive, aucun test.

## 5. Suite
λ_aux = 0,1 non promu ; la ligne principale reste GCO-std sans tâche auxiliaire. Ordre validé : ADR-013 (variante SIL à fondu
tardif, **proposé**, à valider par l'utilisateur avant lancement), puis seeds 1 et 2, puis préparation de G3.
