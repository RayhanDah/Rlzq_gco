# Smoke test de la Phase 3 — 1M timesteps (2026-11-13)

Critère de sortie de la Phase 3 (`docs/PLAN.md`) : smoke test de 1M steps sans NaN, avec reprise depuis checkpoint.
**Verdict du script : PASS** (8 critères sur 8). Ce test prouve que la boucle tourne et reprend ; il ne prouve ni
l'apprentissage du jeu ni le débit soutenu.

## Provenance
- Commande (variables `ROCKETGPU_PY_DIR=vendor/build/rocketgpu-cpu-pgo/python` et
  `ROCKETGPU_COLLISION_MESH_DIR=vendor/collision_meshes`) :
  `vendor/venv-xpu/Scripts/python scripts/train.py configs/smoke.yaml --smoke`
- Commit : `ea0e7de`, arbre propre (`git_dirty = false`). RocketGPU @f0508a1 (build CPU PGO).
- Config : `configs/smoke.yaml`, hash de trajectoire `9b3065d2…0cf4de`, seed 0, variante `GCO-std`.
- Matériel : laptop Zenbook 14, Core Ultra 5 125H, GPU intégré Arc (device `xpu`), torch 2.15.0+xpu, Windows 11.
  Sur secteur ; état thermique non contrôlé.
- Date : 2026-11-13T00:28:44Z. Sorties locales, non versionnées : `runs/smoke/` (résumé `smoke_summary.json`,
  `metrics.jsonl`, `events.jsonl`, `segments.jsonl`, `run_meta*.json`) et `checkpoints/smoke/`.

## Déroulé
4 096 arènes en 1v1 (8 192 agents), 7 steps par lot, soit 57 344 timesteps agents par itération ; collecte en
recouvrement (staleness 1). 18 itérations = 1 032 192 timesteps appris (1 089 536 collectés, un lot d'avance).
1. Première partie : itérations 1 à 9, arrêt demandé (`paused`).
2. Reprise dans un nouveau processus logique depuis `ckpt_000002_it00000009.pt`, état du simulateur restauré
   (`exact_env = true`), itérations 10 à 18 (`completed`).

## Critères du verdict
| Critère | Résultat |
|---|---|
| Budget atteint (18 itérations sur 18, ≥ 1 000 000 timesteps) | oui |
| NaN ou Inf détectés | 0 |
| Arrêt demandé par une garde | aucun |
| Restauration par une garde | aucune |
| Reprise au milieu du run (itération 9) | oui |
| Reprise avec simulateur restauré à l'identique | oui |
| Checkpoints écrits | 6 |

## Mesures
| Grandeur | Valeur |
|---|---|
| Durée totale (mur) | 18,9 s, dont 13,3 s d'itérations |
| Débit moyen sur les itérations | 81 853 timesteps/s |
| Débit moyen sur la durée totale (démarrage et checkpoints inclus) | 57 595 timesteps/s |
| Débit par itération, 1re partie hors 1re itération (n = 8) | médiane 97 226 ; min 89 225 ; max 101 420 |
| Débit par itération, 2e partie (n = 9) | médiane 93 573 ; min 82 210 ; max 99 320 |
| 1re itération (démarrage, deux lots collectés) | 39 370 timesteps/s |
| Taille d'un checkpoint (simulateur inclus) | 150 843 032 octets (≈ 143,9 MiB) |
| Capture de l'état | médiane 0,18 s (0,18 à 0,20 s) |
| Écriture atomique avec fsync | médiane 0,71 s (0,68 à 0,82 s) |
| Rotation | 2 fichiers restants sur 6 écrits (`keep: 2`) |

Métriques PPO, de la 1re à la 18e itération (`runs/smoke/metrics.jsonl`) :

| Métrique | Itération 1 | Itération 9 | Itération 18 | Étendue |
|---|---:|---:|---:|---|
| Entropie (ln 90 = 4,4998) | 4,499 | 4,461 | 4,364 | décroît de façon monotone |
| KL approx | 0,0004 | 0,0015 | 0,0013 | max 0,0026 (seuil de garde 0,05) |
| Fraction clippée | 0 | 0 | 0 | max 0,0031 |
| Explained variance | −0,084 | 0,216 | 0,456 | négative à la 1re itération seulement |
| Perte de valeur | 0,0078 | 0,0016 | 0,0011 | — |
| Norme de gradient globale (avant clip à 0,5) | 0,774 | 0,156 | 0,210 | clip actif à la 1re itération seulement |
| Buts par lot | 11 | 9 | 13 | 5 à 14 ; 195 au total |
| `p_kick` | 0,200 | 0,467 | 0,767 | annealing linéaire |

## Lecture
Mesuré :
- La boucle complète (RocketGPU, observations à 109 dimensions, PPO en recouvrement, gardes, checkpoints, reprise)
  tourne sur XPU sans NaN, et la reprise repart de l'itération exacte du checkpoint avec le simulateur restauré.
- Les statistiques PPO restent dans des plages saines : aucune garde ne se déclenche.

Interprété, à ne pas sur-lire :
- **Les 195 buts ne sont pas une preuve d'apprentissage.** Avec une politique quasi aléatoire (entropie à 97 % du
  maximum), ils viennent pour l'essentiel des états initiaux génériques (balle et voitures tirées au hasard). Ils
  montrent seulement que le signal ±1 arrive au learner. Le jalon S1 se juge au run court 4a, contre un adversaire
  gelé.
- **Le débit n'est pas un débit soutenu.** 18,9 s de run sur une machine à l'état thermique inconnu : le plateau
  mesuré sur 57 min avec le prototype est ≈ 50k timesteps/s (`docs/adr/001-infra.md` M4, −35 à −40 % après
  10 à 15 min). Le débit soutenu de la vraie boucle reste à mesurer au run court 4a.
- La hausse de l'explained variance et la baisse de l'entropie sur 18 itérations indiquent que le critique et la
  politique bougent ; 1M timesteps est trop court pour en dire plus.

## Non vérifié par ce test
- L'égalité bit à bit d'une reprise sur XPU contre un run continu sur XPU (prouvée en CPU seulement,
  `tests/integration/test_resume_rocketgpu.py`).
- Le chemin de restauration par une garde sur un vrai run (testé en unitaire avec un simulateur factice).
- La RAM et le comportement sur la durée (throttling, fuite mémoire, taille cumulée des journaux).
- Kaggle : ce smoke test est local. Tout calcul Kaggle exige son propre smoke test sur le même commit.

## Suites
Avant le run court 4a (détail dans `production/sprints/2026-11-11.md`) : détection du jalon S1 pour les gardes,
checkpoint sain à l'itération 0, journal et compteur de buts après une restauration, `p_kick` final à 0,8 ou plus
(0,767 ici : l'annealing s'arrête un cran trop tôt), adversaire gelé et protocole d'évaluation (`src/eval`).
