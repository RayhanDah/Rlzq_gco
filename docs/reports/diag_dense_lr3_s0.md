# Rapport : `diag_dense_lr3_s0` (ADR-012 phase A, HORS SPEC, LR ×3, graine 0, 0,5B)

- Date : 2026-11-22. Rédigé par l'orchestrateur, sous la délégation de l'utilisateur (« je te laisse gérer le projet et les runs »,
  2026-11-21). Verdict calculé par script, avec les seuils d'ADR-012 D3 recopiés tels quels, sans relecture humaine.
- Run : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant diag_dense_lr3 --seed 0 --max-hours 0 --max-iterations 9688`,
  laptop Core Ultra 5 125H + Arc (XPU), torch 2.15.0+xpu. Récompense dense de diagnostic d'ADR-011 (`dense-diag`, HORS SPEC),
  diagnostic, jamais promu.
  - Segment 1 : commit 042b335, smoke `runs/smoke_dense_lr3_042b335` PASS, début le 2026-11-21 à 18 h 34. Il s'arrête à
    l'itération 1 148 sur une perte du GPU (`UR_RESULT_ERROR_DEVICE_LOST`), reprise automatique depuis l'itération 1 000.
  - Segment 2 : interrompu par une pause demandée par l'utilisateur, vers 21 h 15 (it. ≈ 5 935, processus tués, pas de ligne
    dans `segments.jsonl`).
  - Segment 3 : reprise le 2026-11-22 à 03 h 27 depuis `ckpt_000023_it00005750.pt`, sur 70c64ed. Ce commit fusionne le code
    d'ADR-010, sans effet hors `aux` (revue APPROVE). Smoke `runs/smoke_dense_lr3_70c64edb` PASS. Fin à 04 h 46 sur l'itération
    9 688 (`paused 9688`, 52 240 pas collectés/s en moyenne).
  - Reprises XPU non bit à bit (ADR-012, menaces).
- θ₀ identique au témoin `diag_dense_s0` : SHA-256 7fab4c74…cb8a (mesuré au lancement, `state.md`).
- Conventions : « B » = milliard de timesteps appris. Fichiers :
  - m = `runs/diag_dense_lr3_s0/metrics.jsonl` : 9 688 lignes, itérations 1 à 9 688 uniques, 500 007 368 appris ;
  - g = `segments.jsonl`, v = `events.jsonl` ;
  - évals hors ligne dans `runs/diag_dense_lr3_offline_eval/`, témoin dans `runs/diag_dense_offline_eval/`.
  - Wilson 95 % bilatéral (z = 1,959964), calculé à partir des JSON.

## 1. Verdict ADR-012 D3 : **NEUTRE**
Commande, commit 86aa57a, `.venv` CPU, 2026-11-22 :
`scripts/eval.py --a checkpoints/diag_dense_lr3_s0/eval_snapshots/ckpt_000006_it00009688.pt --b random --episodes 20000
--n-envs 256 --seed 0 --sprt-elo 100 --sprt-max-decisive 1200`. Sortie : `lr3_s0_final20k_it09688.json`. Verdict :
`verdict_d3.json`.

| Volet | Règle D3 | Mesure | Résultat |
|---|---|---|---|
| SUCCÈS | Wilson bas de « A touche » ≥ 46,0 % **et** SPRT H1 | 29,10 % [28,48 ; 29,74], SPRT H1 | non rempli (le SPRT est rempli, la touche non) |
| NUISIBLE | Wilson haut < 22,4 %, **ou** KL moyenne d'une tranche de 0,1B > 0,02, **ou** STOP | Wilson haut 29,74 % ; KL de tranche max 2,10·10⁻³ ; aucun STOP (g : `error`, `paused`) | non rempli |
| NEUTRE | sinon | | **retenu** |

Suite D3, déjà validée (point 5 des validations d'ADR-012) : audit clos, LR inchangée, pas de phase B. Aucun ADR ent_coef sans
demande explicite. On revient à l'ordre ADR-009 D9 : ADR-007 a été lancé automatiquement ensuite (§5).

## 2. Comparaison au témoin (hors ligne, `--seed 0`, contre l'aléatoire)
| Point | LR ×3 : touche [Wilson] | Témoin : touche [Wilson] | LR ×3 : décisives | Témoin : décisives |
|---|---|---|---|---|
| 0,1B (2 048) | 8,50 [7,36 ; 9,78] | 7,23 [6,18 ; 8,43] | 1,32 % | 1,66 % |
| 0,2B (2 048) | 11,52 [10,21 ; 12,98] | 10,84 [9,57 ; 12,26] | 2,34 % | 1,71 % |
| 0,3B (2 048) | 14,36 [12,90 ; 15,94] | 12,84 [11,46 ; 14,36] | 2,20 % | 2,73 % |
| 0,4B (2 048) | 19,87 [18,20 ; 21,66] | 18,12 [16,51 ; 19,84] | 3,86 % | 3,61 % |
| 0,5B (2 048) | 28,32 [26,41 ; 30,31] | 23,93 [22,13 ; 25,82] | 5,52 % | 3,96 % |
| **0,5B (20 000)** | **29,10 [28,48 ; 29,74]** | **23,04 [22,46 ; 23,62]** | **5,72 % [5,41 ; 6,05]** | 4,50 % [4,22 ; 4,79] |

- À 0,5B, avec 20 000 épisodes, les intervalles de touche sont disjoints : le rapport vaut 1,26, loin du ×2 du seuil SUCCÈS. Le
  témoin à 1B touchait dans 50,4 % des épisodes. Élo décisif +168 [146 ; 191] contre +115 pour le témoin au même point. Victoires
  et défaites : 829 / 315 contre 593 / 306.
- Lecture : la LR ×3 accélère un peu l'apprentissage du contact (écart visible surtout dans la dernière tranche), mais n'est pas
  le goulot à cette échelle. Une graine par bras : la variance entre graines du témoin dense n'est pas mesurée (ADR-012, menaces).
  Un écart de 1,26 reste sous la dispersion entre graines mesurée en GCO (1,60, M2). Il n'est donc pas attribuable à la LR avec
  certitude.

## 3. Diagnostics joints (descriptifs, m des deux runs, moyennes par tranche de 0,1B)
| Tranche | KL ×3 | KL base | Rapport KL | ‖g‖ ×3 / base | clip ×3 / base | entropie ×3 / base | touché/fins (entraînement) ×3 / base |
|---|---|---|---|---|---|---|---|
| 0–0,1B | 2,10·10⁻³ | 1,20·10⁻³ | 1,75 | 0,062 / 0,098 | 0,012 / 0,004 | 4,432 / 4,409 | 0,100 / 0,096 |
| 0,1–0,2B | 1,37·10⁻³ | 6,45·10⁻⁴ | 2,13 | 0,058 / 0,070 | 0,008 / 0,002 | 4,412 / 4,422 | 0,135 / 0,128 |
| 0,2–0,3B | 1,30·10⁻³ | 6,44·10⁻⁴ | 2,02 | 0,067 / 0,073 | 0,008 / 0,002 | 4,380 / 4,394 | 0,187 / 0,175 |
| 0,3–0,4B | 1,42·10⁻³ | 8,05·10⁻⁴ | 1,77 | 0,089 / 0,083 | 0,011 / 0,004 | 4,308 / 4,350 | 0,282 / 0,235 |
| 0,4–0,5B | 1,35·10⁻³ | 9,81·10⁻⁴ | 1,37 | 0,110 / 0,101 | 0,012 / 0,005 | 4,242 / 4,279 | 0,392 / 0,326 |

- **L'hypothèse KL ∝ LR² est réfutée** pour ce run : le rapport mesuré va de 1,4 à 2,1, et non ≈ 9. Le KL a donc crû à peu près
  comme la LR elle-même, voire moins. La règle de repli de D1 (seuil 2,2·10⁻³ fondé sur ×9) était très conservatrice.
- (ln 90 − H) sur les 100 dernières itérations avant 0,5B : 0,271 (×3) contre 0,290 (base). Aucune des deux prédictions de D3
  ne se réalise : ni « touche ≥ 2× et entropie plus basse », ni « touche ≈ 23 % et (ln 90 − H) ≈ 0,22 ».
- Santé : 0 restauration, entropie minimale 4,136 (K2 non déclenché), KL max d'une itération 3,9·10⁻² dans la tranche de
  démarrage (seuil K4 0,05), EV minimale après l'itération 200 : 0,681. Une perte du GPU (R14).

## 4. Limites
Une graine par bras ; diagnostic dense HORS SPEC, sans transfert garanti au GCO ; reprises non bit à bit (2 reprises). Levier
choisi après avoir vu les données, limité par le pré-enregistrement. Le segment 3 tourne sur un commit différent (70c64ed) :
code `aux` inactif, revu. Le SPRT est un rejeu après coup (voir `eval.sprt`).

## 5. Suite
ADR-007 `int_gamma998_s0` a été lancé par l'enchaînement de nuit, après le smoke `configs/smoke.yaml --smoke` sur 86aa57a
(détails dans `state.md`).
