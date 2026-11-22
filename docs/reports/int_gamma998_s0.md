# Rapport : `int_gamma998_s0` (ADR-007, GCO-std, γ 0,998, graine 0, 1B)

- Date : 2026-11-22. Rédigé par l'orchestrateur, sous la délégation de l'utilisateur du 2026-11-21. Le verdict est appliqué par
  script (`post_adr007_chain.py`, scratchpad), seuils d'ADR-007 D4 recopiés tels quels. Ce même calcul, appliqué aux fichiers de
  la SIL, redonne exactement le verdict publié de `int_sil_s0` (NEUTRE, 4,33 % [4,06 ; 4,63], paire 0,3B ; 0,4B).
- Run : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant int_gamma998 --seed 0 --max-hours 0`, commit 86aa57a.
  - Smoke `configs/smoke.yaml --smoke` PASS sur le même commit (`runs/smoke_86aa57a`).
  - Laptop Core Ultra 5 125H + Arc (XPU), torch 2.15.0+xpu.
  - Début le 2026-11-22 à 05 h 05, fin à 11 h 14 (≈ 6,15 h).
- θ₀ identique à std s0 : SHA-256 7fab4c747a778f6a…cdcb8a des deux instantanés de l'itération 0 (méthode
  `tests/unit/test_run_4a_interventions.py:414`, mesuré le 2026-11-22). γ = 0,998 lu dans `run_meta.json`.
- Conventions : m = `runs/int_gamma998_s0/metrics.jsonl` (19 376 lignes, itérations uniques), g = `segments.jsonl`,
  v = `events.jsonl`. Évals hors ligne dans `runs/int_gamma998_offline_eval/`, commit 690a828, `.venv` CPU. Wilson 95 %
  (z = 1,959964) calculé à partir des JSON.

## 1. Verdict ADR-006 D6 (ADR-007 D4) : **NEUTRE**
| Volet | Règle | Mesure | Résultat |
|---|---|---|---|
| (i) | 1B, 20 000 épisodes `--seed 0` : Wilson bas des décisives ≥ 6,28 % **et** SPRT H1 | 649 / 20 000 = 3,24 % [3,01 ; 3,50], SPRT H1 | non rempli |
| (ii) | 1re paire consécutive H1 (2 048 épisodes) au plus tard à 0,2B | 1re paire (0,5B ; 0,6B) | non rempli |

Référence std s0 : 3,14 % à 1B, 1re paire à 0,5B (`docs/reports/run_4a.md`). L'écart à 1B est de +0,10 point, dans le bruit
d'une graine (variance entre graines std ×1,5, ADR-007 Conséquences 3). **γ = 0,995 conservé.** Suite (ADR-007 D4, neutre) :
ADR-010, puis la variante SIL. L'ADR-013 (proposé) propose de passer la variante SIL avant ADR-010 : décision de l'utilisateur.

## 2. Trajectoire hors ligne (2 048 épisodes `--seed 0` contre l'aléatoire, descriptif)
| B | γ 0,998 : décisives [Wilson] | SPRT | touche | std s0 : décisives | SPRT | touche |
|---|---|---|---|---|---|---|
| 0,0 | 1,32 [0,91 ; 1,91] | plafond | 6,6 | 1,32 | plafond | 6,6 |
| 0,1 | 1,42 [0,99 ; 2,03] | plafond | 5,6 | 1,76 | H1 | 7,2 |
| 0,2 | 1,37 [0,95 ; 1,97] | plafond | 6,9 | 1,46 | plafond | 7,3 |
| 0,3 | 1,81 [1,31 ; 2,48] | plafond | 6,2 | 1,51 | plafond | 7,0 |
| 0,4 | 1,37 [0,95 ; 1,97] | plafond | 6,4 | 1,81 | plafond | 9,6 |
| 0,5 | 2,64 [2,03 ; 3,42] | H1 | 9,8 | 2,00 | H1 | 8,1 |
| 0,6 | 2,59 [1,98 ; 3,37] | H1 | 10,7 | 1,71 | H1 | 9,0 |
| 0,7 | 2,64 [2,03 ; 3,42] | H1 | 8,4 | 2,34 | H1 | 10,2 |
| 0,8 | 3,52 [2,80 ; 4,40] | H1 | 13,0 | 2,69 | H1 | 10,8 |
| 0,9 | 3,71 [2,98 ; 4,62] | H1 | 15,8 | 2,59 | H1 | 11,3 |
| 1,0 | 3,52 [2,80 ; 4,40] | H1 | 15,0 | 2,64 | H1 | 13,9 |
| **1,0 (20 000)** | **3,24 [3,01 ; 3,50]** | H1 | 14,4 | 3,14 | H1 | — |

- Le point 0,0 est identique à std s0 (même θ₀, même graine d'éval). Ensuite, γ 0,998 est au-dessus de std de 0,5 à 1B sur les
  2 048 épisodes (+0,3 à +1,1 point), mais les intervalles se recouvrent à chaque point. À 20 000 épisodes, l'écart final tombe à
  +0,10 point. Élo décisif final +231 (513 V / 136 D).
- La touche reste au niveau de std (14,4 % à 1B) : l'horizon plus long n'amorce pas le contact (R1 inchangé).

## 3. Diagnostics (m, moyennes par tranche de 0,1B, γ 0,998 / std s0)
| Tranche | EV | perte de valeur | ‖g‖ | KL | entropie | buts/it. |
|---|---|---|---|---|---|---|
| 0–0,1 | 0,875 / 0,821 | 4,06·10⁻⁴ / 3,94·10⁻⁴ | 0,093 / 0,082 | 9,7·10⁻⁴ / 6,6·10⁻⁴ | 4,436 / 4,457 | 7,38 / 7,38 |
| 0,2–0,3 | 0,949 / 0,942 | 2,30·10⁻⁴ / 1,89·10⁻⁴ | 0,054 / 0,046 | 2,0·10⁻⁴ / 1,3·10⁻⁴ | 4,472 / 4,485 | 5,63 / 5,43 |
| 0,4–0,5 | 0,967 / 0,961 | 1,65·10⁻⁴ / 1,30·10⁻⁴ | 0,055 / 0,057 | 2,1·10⁻⁴ / 2,3·10⁻⁴ | 4,464 / 4,455 | 4,20 / 4,20 |
| 0,6–0,7 | 0,968 / 0,962 | 1,77·10⁻⁴ / 1,28·10⁻⁴ | 0,067 / 0,070 | 2,0·10⁻⁴ / 2,3·10⁻⁴ | 4,429 / 4,434 | 4,39 / 4,14 |
| 0,8–0,9 | 0,967 / 0,960 | 2,08·10⁻⁴ / 1,53·10⁻⁴ | 0,085 / 0,083 | 2,1·10⁻⁴ / 1,8·10⁻⁴ | 4,371 / 4,388 | 4,96 / 4,83 |
| 0,9–1,0 | 0,963 / 0,959 | 2,44·10⁻⁴ / 1,62·10⁻⁴ | 0,098 / 0,092 | 1,5·10⁻⁴ / 1,4·10⁻⁴ | 4,325 / 4,357 | 5,63 / 5,04 |

- Risque 1 d'ADR-007 (variance des cibles du critique) : la perte de valeur est 1,2 à 1,5 fois celle de std à partir de 0,2B,
  mais l'EV reste au moins égale à celle de std. Elle ne descend jamais sous 0,5 après 0,2B (0 itération) ; son minimum après
  l'itération 200 est de 0,742. La clause de révision « EV < 0,5 durablement » ne se déclenche pas.
- ‖g‖ suit std à ±15 % (diagnostic D3). KL max d'une itération : 7,5·10⁻³ (K4 à 0,05). Entropie minimale : 4,176.
- Santé :
  - g : `error` sur 0 → 16 118, perte du GPU (`UR_RESULT_ERROR_DEVICE_LOST`), reprise automatique depuis l'itération 16 000 ;
    puis `completed` 16 000 → 19 376, à 55 031 pas collectés/s.
  - 0 restauration, aucun STOP.
  - v : `s1_reached` à l'itération 11 751. C'est une mesure en cours de run (protocole v2), jamais comparée.
  - 8e perte du GPU du projet (R14).

## 4. Limites
Une graine. Reprise XPU non bit à bit (1 reprise). Le verdict dépend de la base std s0 d'une graine. Un neutre ne réfute pas
γ = 0,998 (ADR-007 Conséquences 3). Les évals de std s0 et de γ 0,998 ont été faites sur des commits différents : leurs points
θ₀ (0,0B) sont identiques, ce qui contrôle l'équivalence sur ce seul point.
