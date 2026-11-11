# Parité physique RocketGPU @f0508a1 vs RocketSim 2.2.1 (T3, sprint 2026-11-11)

Statut : mesure, pas une décision. Le critère de révision de l'ADR-001 (« écart de physique trop grand ») est discuté
en conclusion ; la décision revient au `tech-director`.

## 1. Conditions de mesure

| Élément | Valeur |
|---|---|
| Commande exacte | `ROCKETGPU_PY_DIR=C:/Users/Rayha/Desktop/RL-GCO/vendor/build/rocketgpu-cpu-pgo/python ROCKETGPU_COLLISION_MESH_DIR=C:/Users/Rayha/Desktop/RL-GCO/vendor/collision_meshes .venv/Scripts/python tests/integration/test_physics_parity.py --report` (aussi `--report --cold`, `--lag`, `--series <scénario>`) ; tests : `.venv/Scripts/python -m pytest -q tests/integration/test_physics_parity.py` |
| Graine | `SEED = 20261111` (contrôles aléatoires, `np.random.default_rng`), tick-skip 8 |
| Commit | `f4f1a45` (`git rev-parse --short HEAD` au moment de la mesure) ; le fichier de test est ajouté par le commit suivant de la branche `worktree-agent-a53417f496a707eeb` |
| Matériel | laptop Zenbook 14, Core Ultra 5 125H, 16 Go (CPU seul) |
| Date | 2026-11-11 |
| RocketSim | 2.2.1 (module `RocketSim` du venv du dépôt, CPU) |
| RocketGPU | build CPU PGO `vendor/build/rocketgpu-cpu-pgo`, source @f0508a1 d'après l'ADR-001 (le hash du clone n'a pas été revérifié ici : accès git à `vendor/` hors périmètre de cet agent) |
| Python / numpy | 3.12.6 / 2.5.4 |
| Durée | ≈ 10 s pour les 87 tests (85 scénarios, deux simulateurs) |

### Conventions vérifiées par introspection (et non devinées)
- Même états initiaux écrits explicitement dans les deux simulateurs (pas de `reset_kickoff` comparé : les positions de
  kickoff pour une même graine diffèrent, voir A6). Kickoff diagonal bleu (-2048, -2560, 17) cap 45 degrés, orange symétrique.
- RocketGPU : quaternion `(w, x, y, z)` ; `rot` = rotation dont les colonnes sont `(forward, right, up)` au sens RocketSim
  (`RotMat.right` de RocketSim est la gauche géométrique : pour cap 90 degrés, `right` = (-1, 0, 0)). Vérifié en comparant
  `basis_forward/right/up` de RocketGPU à `RotMat` de RocketSim sur le même état (écart initial < 1e-3 rad,
  `test_initial_states_identical`).
- Voiture 0 = bleue, voiture 1 = orange dans les deux. Boost en 0..100 dans les deux.
- Pads de boost : actifs au départ dans les deux (34 pads). Voiture 1 inerte placée en (3000, 4000) ramasse le gros pad
  (3072, 4096) dans les deux (même effet des deux côtés dans les scénarios « voiture 1 inerte »).
- Pièges de harnais rencontrés (corrigés dans le test, mesurés) : (i) `upload_car_states([CarState])` **ignore `.rot`**
  quand l'état téléchargé a `basis_authoritative = 1` (l'import recalcule `rot` depuis `basis_*`,
  `include/rocketgpu/detail/math.hpp:1306`) : on écrit donc les tableaux de structures bruts via `upload_state_array`
  (`SeedCarBasisFromRotMat`, `src/rocketsim_compat.cpp:25`) ; (ii) il faut remettre `pad_frame_pos_valid` à 0, sinon un état
  téléporté « balaye » les pads entre l'ancienne et la nouvelle position (mesuré : +24 de boost fantôme) ; (iii) la copie
  des tableaux de structures doit être octet à octet (`tobytes`) : une copie `np.array(...)` laissait des octets non initialisés
  et provoquait un **plantage intermittent (violation d'accès dans `BatchSim.step`, ≈ 2 exécutions sur 3)** ; après correction,
  aucun plantage sur plus de 10 exécutions du banc complet.

## 2. Scénarios

`idle_kickoff` (bruit de référence), `kickoff_*` (voiture 0 pilotée : plein gaz, braquage, boost, marche arrière, boost +
frein à main + braquage ; voiture 1 inerte), `jump_tap`, `jump_hold`, `double_jump`, `flip_forward`, `flip_diag`,
`flip_forward_moving` (voiture à l'arrêt ou à 1000 uu/s), `ball_drop`, `ball_drop_spin`, `ball_wall_ceiling`, `hit_center`,
`hit_offset`, `hit_fast_offset` (voiture lancée qui frappe la balle au repos), `boost_ground_*`, et 8 groupes de 8 tirages de
contrôles aléatoires (`uniform` = distribution de `scripts/benchmark_sim.py` ; `biased` = gaz fréquent, boost 50 %, sauts rares ;
variantes `_tb` : mêmes tirages mais gaz forcé à 1 quand le boost est pressé), à 5 s et 20 s, kickoff diagonal, deux voitures
pilotées. Un tick d'échauffement sans contrôle est exécuté dans les deux simulateurs avant la mesure (voir 4.2).

Métriques (RocketSim moins RocketGPU) : `pos` en uu (norme), `vel` en uu/s, `angvel` en rad/s, `rot` = angle de la rotation
relative en rad, `boost` en points (0..100). Horizons en ticks (120 ticks = 1 s) : 16, 64, 120, 240, 600, 2400 (selon la durée).

## 3. Résultats mesurés

### 3.1 Scénarios déterministes et frappes (voiture 0 ; balle)
| scénario | horizon (ticks) | car0_pos | car0_vel | car0_angvel | car0_rot | car0_boost | ball_pos | ball_vel | ball_angvel |
|---|---|---|---|---|---|---|---|---|---|
| idle_kickoff | 16 | 0.00863 | 0.176 | 0.000556 | 0.000321 | 0 | 0 | 0 | 0 |
| idle_kickoff | 64 | 0.0296 | 0.379 | 0.000106 | 0 | 0 | 0 | 0 | 0 |
| idle_kickoff | 120 | 0.0296 | 0.379 | 0.000107 | 0 | 0 | 0 | 0 | 0 |
| idle_kickoff | 240 | 0.0296 | 0.379 | 0.000108 | 0 | 0 | 0 | 0 | 0 |
| idle_kickoff | 600 | 0.0296 | 0.379 | 0.000107 | 0 | 0 | 0 | 0 | 0 |
| kickoff_throttle | 16 | 1.45 | 9.21 | 0.00061 | 0 | 0 | 0 | 0 | 0 |
| kickoff_throttle | 64 | 3.68 | 2.61 | 0.000233 | 0 | 12 | 0 | 0 | 0 |
| kickoff_throttle | 120 | 3.93 | 1.11 | 0.000608 | 0.000185 | 0 | 0 | 0 | 0 |
| kickoff_throttle | 240 | 1.78 | 2.57 | 0.000755 | 0 | 0 | 0 | 0 | 0 |
| kickoff_throttle | 600 | 1.49 | 0.387 | 0.00084 | 0.000359 | 0 | 0 | 0 | 0 |
| kickoff_throttle_steer | 16 | 1.41 | 9.35 | 0.032 | 0.0081 | 0 | 0 | 0 | 0 |
| kickoff_throttle_steer | 64 | 3.52 | 4.01 | 0.0164 | 0.00508 | 0 | 0 | 0 | 0 |
| kickoff_throttle_steer | 120 | 3.88 | 1.8 | 0.0143 | 0.00244 | 0 | 0 | 0 | 0 |
| kickoff_throttle_steer | 240 | 8.49 | 10.8 | 0.00368 | 0.00988 | 0 | 0 | 0 | 0 |
| kickoff_throttle_steer | 600 | 11.1 | 18.4 | 0.00109 | 0.0152 | 0 | 0 | 0 | 0 |
| kickoff_throttle_boost | 16 | 1.39 | 8.29 | 0.0011 | 0 | 0.282 | 0 | 0 | 0 |
| kickoff_throttle_boost | 64 | 2.8 | 0.15 | 0.000126 | 0.000242 | 0.295 | 0 | 0 | 0 |
| kickoff_throttle_boost | 120 | 1.61 | 3.8 | 0.000611 | 0.000102 | 0.311 | 0 | 0 | 0 |
| kickoff_throttle_boost | 240 | 2.56 | 4.02 | 0.000603 | 0.000201 | 0 | 0 | 0 | 0 |
| kickoff_throttle_boost | 600 | 18.7 | 53.7 | 0.522 | 0.0617 | 0 | 0 | 0 | 0 |
| kickoff_reverse | 16 | 1.73 | 11.2 | 0.000808 | 0.000344 | 0 | 0 | 0 | 0 |
| kickoff_reverse | 64 | 4.54 | 3.89 | 7.9e-06 | 0.000395 | 0 | 0 | 0 | 0 |
| kickoff_reverse | 120 | 5.26 | 0.453 | 0.000491 | 0.000159 | 0 | 0 | 0 | 0 |
| kickoff_reverse | 240 | 3.54 | 2.3 | 0.000699 | 0 | 0 | 0 | 0 | 0 |
| kickoff_reverse | 600 | 2.24 | 11.7 | 0.0217 | 0.00665 | 0 | 0 | 0 | 0 |
| kickoff_boost_steer_drift | 16 | 1.13 | 5.38 | 0.00177 | 0.00264 | 0.282 | 0 | 0 | 0 |
| kickoff_boost_steer_drift | 64 | 2.03 | 0.793 | 0.00604 | 0 | 0.295 | 0 | 0 | 0 |
| kickoff_boost_steer_drift | 120 | 2.07 | 1.07 | 0.00306 | 0.00204 | 0.311 | 0 | 0 | 0 |
| kickoff_boost_steer_drift | 240 | 10.1 | 24.4 | 0.0335 | 0.000432 | 0 | 0 | 0 | 0 |
| kickoff_boost_steer_drift | 600 | 85.5 | 93.3 | 0.347 | 0.0926 | 1.92 | 0 | 0 | 0 |
| jump_tap | 16 | 0.232 | 0.868 | 0.0225 | 0.00365 | 0 | 0.0282 | 0.0719 | 0 |
| jump_tap | 64 | 0.684 | 0.833 | 0.00813 | 0.00911 | 0 | 0.0531 | 0.259 | 0 |
| jump_tap | 120 | 1.27 | 0.93 | 0.00309 | 0.0115 | 0 | 0.219 | 0.482 | 0 |
| jump_tap | 240 | 1.18 | 0.271 | 0.000617 | 0 | 0 | 0.355 | 0.526 | 0 |
| jump_tap | 600 | 1.18 | 0.271 | 0.000617 | 0 | 0 | 0.000908 | 0 | 0 |
| jump_hold | 16 | 0.204 | 1.14 | 0.0225 | 0.00365 | 0 | 0.0282 | 0.0719 | 0 |
| jump_hold | 64 | 0.892 | 1.57 | 0.00813 | 0.00911 | 0 | 0.0531 | 0.259 | 0 |
| jump_hold | 120 | 1.81 | 1.57 | 0.00309 | 0.0115 | 0 | 0.219 | 0.482 | 0 |
| jump_hold | 240 | 3.15 | 0.176 | 0.00099 | 0 | 0 | 0.355 | 0.526 | 0 |
| jump_hold | 600 | 3.15 | 0.271 | 0.000172 | 0.000201 | 0 | 0.000908 | 0 | 0 |
| double_jump | 16 | 0.232 | 0.868 | 0.0225 | 0.00365 | 0 | 0.0282 | 0.0719 | 0 |
| double_jump | 64 | 0.895 | 3.11 | 0.00813 | 0.00911 | 0 | 0.0531 | 0.259 | 0 |
| double_jump | 120 | 2.21 | 3.11 | 0.00309 | 0.0115 | 0 | 0.219 | 0.482 | 0 |
| double_jump | 240 | 4.45 | 2.15 | 0.0153 | 0.00092 | 0 | 0.355 | 0.526 | 0 |
| double_jump | 600 | 4.47 | 0.271 | 0.000617 | 0 | 0 | 0.000908 | 0 | 0 |
| flip_forward | 16 | 0.232 | 0.868 | 0.0225 | 0.00365 | 0 | 0.0282 | 0.0719 | 0 |
| flip_forward | 64 | 1.86 | 0.827 | 4.57e-05 | 0.00228 | 0 | 0.0531 | 0.259 | 0 |
| flip_forward | 120 | 3.1 | 5.66 | 0.000794 | 0.032 | 0 | 0.219 | 0.482 | 0 |
| flip_forward | 240 | 1.5 | 3.08 | 0.000606 | 0.000301 | 0 | 0.355 | 0.526 | 0 |
| flip_forward | 600 | 1.8 | 0.271 | 0.000836 | 0 | 0 | 0.000908 | 0 | 0 |
| flip_diag | 16 | 0.232 | 0.868 | 0.0225 | 0.00365 | 0 | 0.0282 | 0.0719 | 0 |
| flip_diag | 64 | 2.07 | 4.19 | 0.17 | 0.0408 | 0 | 0.0531 | 0.259 | 0 |
| flip_diag | 120 | 2.52 | 6.65 | 0.0988 | 0.0603 | 0 | 0.219 | 0.482 | 0 |
| flip_diag | 240 | 9.82 | 0.274 | 0.000268 | 0.0258 | 0 | 0.355 | 0.526 | 0 |
| flip_diag | 600 | 9.82 | 0.271 | 0.000504 | 0.0258 | 0 | 0.000908 | 0 | 0 |
| flip_forward_moving | 16 | 0.923 | 7.91 | 0.0227 | 0.00363 | 0 | 0.0282 | 0.0719 | 0 |
| flip_forward_moving | 64 | 4.32 | 7.26 | 3.83e-05 | 0.00221 | 0 | 0.0531 | 0.259 | 0 |
| flip_forward_moving | 120 | 7.7 | 8.27 | 0.000793 | 0.032 | 0 | 0.219 | 0.482 | 0 |
| flip_forward_moving | 240 | 9.72 | 2.04 | 0.000504 | 0 | 0 | 0.355 | 0.526 | 0 |
| flip_forward_moving | 600 | 14.6 | 2.04 | 0.000503 | 0 | 0 | 0.000908 | 0 | 0 |
| ball_drop | 16 | 0.00863 | 0.176 | 0.000556 | 0.000321 | 0 | 0.0281 | 0.0719 | 0 |
| ball_drop | 64 | 0.0296 | 0.379 | 0.000106 | 0 | 0 | 0.0536 | 0.259 | 0 |
| ball_drop | 120 | 0.0296 | 0.379 | 0.000107 | 0 | 0 | 0.229 | 0.482 | 0 |
| ball_drop | 240 | 0.0296 | 0.379 | 0.000108 | 0 | 0 | 0.927 | 0.946 | 0 |
| ball_drop | 600 | 0.0296 | 0.379 | 0.000107 | 0 | 0 | 1.63 | 8.45 | 0 |
| ball_drop_spin | 16 | 0.00863 | 0.176 | 0.000556 | 0.000321 | 0 | 0.0302 | 0.0956 | 1.73e-05 |
| ball_drop_spin | 64 | 0.0296 | 0.379 | 0.000106 | 0 | 0 | 0.094 | 0.385 | 1.73e-05 |
| ball_drop_spin | 120 | 0.0296 | 0.379 | 0.000107 | 0 | 0 | 0.367 | 0.768 | 1.73e-05 |
| ball_drop_spin | 240 | 0.0296 | 0.379 | 0.000108 | 0 | 0 | 1.06 | 1.1 | 0.00781 |
| ball_drop_spin | 600 | 0.0296 | 0.379 | 0.000107 | 0 | 0 | 5.87 | 8.96 | 0.025 |
| ball_wall_ceiling | 16 | 0.00863 | 0.176 | 0.000556 | 0.000321 | 0 | 0.00535 | 0.106 | 0 |
| ball_wall_ceiling | 64 | 0.0296 | 0.379 | 0.000106 | 0 | 0 | 0.119 | 0.482 | 0 |
| ball_wall_ceiling | 120 | 0.0296 | 0.379 | 0.000107 | 0 | 0 | 0.634 | 1.99 | 0.00199 |
| ball_wall_ceiling | 240 | 0.0296 | 0.379 | 0.000108 | 0 | 0 | 3 | 2.64 | 0.00199 |
| ball_wall_ceiling | 600 | 0.0296 | 0.379 | 0.000107 | 0 | 0 | 11.4 | 8.9 | 0.00911 |
| hit_center | 16 | 0.019 | 0.32 | 0.000932 | 6.47e-05 | 0 | 0 | 0 | 0 |
| hit_center | 64 | 0.0292 | 0.382 | 0.000132 | 0 | 0 | 0 | 0 | 0 |
| hit_center | 120 | 0.0448 | 0.382 | 0.000135 | 0 | 0 | 0 | 0 | 0 |
| hit_center | 240 | 0.486 | 0.99 | 0.000375 | 0 | 0 | 1.34 | 3.96 | 0.0385 |
| hit_center | 600 | 0.434 | 0.276 | 0.000699 | 3.65e-05 | 0 | 2.8 | 13.8 | 1.31e-05 |
| hit_offset | 16 | 0.019 | 0.32 | 0.000932 | 6.54e-05 | 0 | 0 | 0 | 0 |
| hit_offset | 64 | 0.0292 | 0.382 | 0.000132 | 0 | 0 | 0 | 0 | 0 |
| hit_offset | 120 | 0.0448 | 0.382 | 0.000134 | 0 | 0 | 0 | 0 | 0 |
| hit_offset | 240 | 0.352 | 1.57 | 0.000802 | 0.000911 | 0 | 1.32 | 3.87 | 0.0411 |
| hit_offset | 600 | 4.24 | 1.35 | 0.000627 | 0.000983 | 0 | 4.64 | 9.55 | 0.00739 |
| hit_fast_offset | 16 | 0.0449 | 0.309 | 0.00107 | 0 | 0.282 | 0 | 0 | 0 |
| hit_fast_offset | 64 | 0.0131 | 0.139 | 0.000573 | 0.000399 | 0.296 | 0 | 0 | 0 |
| hit_fast_offset | 120 | 0.245 | 0.139 | 0.000574 | 0.000399 | 0.311 | 0 | 0 | 0 |
| hit_fast_offset | 240 | 0.839 | 0.932 | 0.000725 | 0.000341 | 0.345 | 4.75 | 5.43 | 0.0153 |
| hit_fast_offset | 600 | 149 | 138 | 0.0643 | 0.116 | 0 | 281 | 194 | 1.84 |
| boost_ground_throttle0 | 16 | 21 | 309 | 0.00242 | 0.000388 | 0.282 | 0.0282 | 0.0719 | 0 |
| boost_ground_throttle0 | 64 | 309 | 1.06e+03 | 0.000751 | 0 | 0.296 | 0.0531 | 0.259 | 0 |
| boost_ground_throttle0 | 120 | 946 | 1.61e+03 | 0.000574 | 0.000212 | 0.311 | 0.219 | 0.482 | 0 |
| boost_ground_throttle_neg | 16 | 27.6 | 392 | 0.00268 | 0.000293 | 0.282 | 0.0282 | 0.0719 | 0 |
| boost_ground_throttle_neg | 64 | 386 | 1.32e+03 | 0.000858 | 0.000146 | 0.296 | 0.0531 | 0.259 | 0 |
| boost_ground_throttle_neg | 120 | 1.18e+03 | 2e+03 | 0.000868 | 0.000221 | 0.311 | 0.219 | 0.482 | 0 |
| boost_ground_throttle1 | 16 | 1.37 | 8.29 | 0.000973 | 0.000373 | 0.282 | 0.0282 | 0.0719 | 0 |
| boost_ground_throttle1 | 64 | 2.83 | 0.15 | 0.00013 | 0 | 0.296 | 0.0531 | 0.259 | 0 |
| boost_ground_throttle1 | 120 | 1.69 | 3.73 | 0.000667 | 9.67e-05 | 0.311 | 0.219 | 0.482 | 0 |

### 3.2 Contrôles aléatoires (médiane / max sur 8 tirages ; voiture 0 ; balle)
| scénario | horizon (ticks) | car0_pos | car0_vel | car0_angvel | car0_rot | car0_boost | ball_pos | ball_vel | ball_angvel |
|---|---|---|---|---|---|---|---|---|---|
| random_uniform_5s (médiane / max, n=8) | 16 | 6.66 / 16.9 | 75.4 / 169 | 0.404 / 0.866 | 0.0324 / 0.0836 | 0.142 / 0.282 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_5s (médiane / max, n=8) | 64 | 40 / 90 | 85.2 / 255 | 0.508 / 1.18 | 0.135 / 0.269 | 0.149 / 0.292 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_5s (médiane / max, n=8) | 120 | 74.2 / 217 | 78 / 307 | 0.691 / 2.01 | 0.154 / 0.402 | 0.296 / 0.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_5s (médiane / max, n=8) | 240 | 221 / 571 | 246 / 384 | 0.424 / 2.18 | 0.168 / 0.402 | 0 / 0.318 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_5s (médiane / max, n=8) | 600 | 838 / 2.67e+03 | 279 / 1.61e+03 | 5.08 / 6.97 | 1 / 2.35 | 0 / 0.0675 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_5s (médiane / max, n=8) | 16 | 1.74 / 2.31 | 16.3 / 19.2 | 0.362 / 0.564 | 0.0173 / 0.0435 | 0.142 / 0.282 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_5s (médiane / max, n=8) | 64 | 6.8 / 12.6 | 13.2 / 63.6 | 0.288 / 1.35 | 0.0796 / 0.15 | 0.149 / 12.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_5s (médiane / max, n=8) | 120 | 12.9 / 40.6 | 18.2 / 68.7 | 0.335 / 1.05 | 0.129 / 0.199 | 0.296 / 0.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_5s (médiane / max, n=8) | 240 | 37.9 / 92.2 | 66 / 189 | 0.353 / 0.828 | 0.144 / 0.264 | 0 / 0.318 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_5s (médiane / max, n=8) | 600 | 648 / 1.97e+03 | 453 / 1.19e+03 | 3.24 / 8.19 | 0.688 / 1.83 | 0 / 3.11 | 0 / 67.7 | 0 / 527 | 0 / 4.51 |
| random_biased_5s (médiane / max, n=8) | 16 | 1.38 / 2.62 | 8.64 / 23 | 0.029 / 0.295 | 0.00507 / 0.0312 | 0.28 / 0.282 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_5s (médiane / max, n=8) | 64 | 5.58 / 45.4 | 10.3 / 261 | 0.105 / 0.396 | 0.0498 / 0.116 | 6 / 12.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_5s (médiane / max, n=8) | 120 | 10.4 / 146 | 25.5 / 202 | 0.262 / 1.6 | 0.0765 / 0.219 | 0.158 / 0.303 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_5s (médiane / max, n=8) | 240 | 98.9 / 436 | 150 / 388 | 0.661 / 5.07 | 0.157 / 0.264 | 0.191 / 12 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_5s (médiane / max, n=8) | 600 | 588 / 1.38e+03 | 320 / 765 | 0.271 / 2.55 | 0.174 / 1.37 | 0.444 / 6.73 | 0 / 3.81e+03 | 0 / 1.11e+03 | 0 / 6 |
| random_biased_tb_5s (médiane / max, n=8) | 16 | 1.39 / 2.62 | 8.91 / 23 | 0.029 / 0.295 | 0.00507 / 0.0312 | 0.28 / 0.282 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_tb_5s (médiane / max, n=8) | 64 | 3.83 / 7.01 | 5.45 / 24.4 | 0.0995 / 0.363 | 0.0359 / 0.116 | 6.15 / 12.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_tb_5s (médiane / max, n=8) | 120 | 5.36 / 27.3 | 20.8 / 73 | 0.262 / 1.6 | 0.0634 / 0.219 | 0.158 / 0.303 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_tb_5s (médiane / max, n=8) | 240 | 47.5 / 169 | 154 / 314 | 0.764 / 5 | 0.172 / 0.255 | 0.191 / 12 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_tb_5s (médiane / max, n=8) | 600 | 555 / 1.95e+03 | 329 / 1.22e+03 | 0.257 / 5.15 | 0.232 / 0.826 | 0 / 0.889 | 0 / 4.4e+03 | 0 / 1.42e+03 | 0 / 5.48 |
| random_uniform_20s (médiane / max, n=8) | 16 | 2.25 / 16.4 | 23.1 / 191 | 0.267 / 0.514 | 0.0127 / 0.0489 | 0.00334 / 0.282 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_20s (médiane / max, n=8) | 64 | 27.7 / 81.1 | 82.8 / 138 | 0.49 / 0.803 | 0.116 / 0.217 | 0.284 / 0.288 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_20s (médiane / max, n=8) | 120 | 56.5 / 145 | 97.8 / 145 | 0.528 / 1.07 | 0.115 / 0.245 | 0.296 / 0.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_20s (médiane / max, n=8) | 240 | 182 / 335 | 205 / 270 | 1.06 / 3.21 | 0.376 / 0.901 | 0 / 0.315 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_20s (médiane / max, n=8) | 600 | 769 / 1.88e+03 | 370 / 947 | 2.01 / 5.7 | 0.54 / 2.7 | 0 / 0 | 0 / 3.35e+03 | 0 / 1.24e+03 | 0 / 6 |
| random_uniform_20s (médiane / max, n=8) | 2400 | 4.9e+03 / 8.01e+03 | 728 / 1.58e+03 | 5.86 / 8.4 | 1.69 / 2.95 | 0 / 0 | 0 / 4.61e+03 | 0 / 264 | 0 / 3.09 |
| random_uniform_tb_20s (médiane / max, n=8) | 16 | 1.35 / 2.82 | 16.9 / 24.7 | 0.249 / 0.399 | 0.00876 / 0.0429 | 0.00334 / 0.282 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_20s (médiane / max, n=8) | 64 | 9.63 / 16.5 | 25.3 / 56.7 | 0.348 / 0.833 | 0.0932 / 0.15 | 0.284 / 0.288 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_20s (médiane / max, n=8) | 120 | 19.6 / 47.8 | 39.3 / 88.7 | 0.483 / 0.609 | 0.119 / 0.172 | 0.296 / 0.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_20s (médiane / max, n=8) | 240 | 74.1 / 181 | 116 / 275 | 1.45 / 2.28 | 0.356 / 0.897 | 0 / 0.315 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_uniform_tb_20s (médiane / max, n=8) | 600 | 620 / 3.31e+03 | 683 / 1.15e+03 | 5.25 / 10.9 | 1.33 / 2.94 | 0 / 9.78 | 0 / 2.5e+03 | 0 / 1.07e+03 | 0 / 6 |
| random_uniform_tb_20s (médiane / max, n=8) | 2400 | 4.95e+03 / 8.11e+03 | 1.33e+03 / 1.77e+03 | 5.99 / 8.62 | 2.18 / 2.75 | 0 / 48.9 | 725 / 4.13e+03 | 108 / 545 | 1.19 / 5.99 |
| random_biased_20s (médiane / max, n=8) | 16 | 1.39 / 1.41 | 8.89 / 15.1 | 0.0343 / 0.186 | 0.00241 / 0.0144 | 0.28 / 0.282 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_20s (médiane / max, n=8) | 64 | 4.64 / 147 | 22.6 / 521 | 0.0772 / 0.41 | 0.0119 / 0.288 | 0.289 / 12.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_20s (médiane / max, n=8) | 120 | 20.9 / 366 | 42.2 / 489 | 0.0782 / 0.361 | 0.0323 / 0.297 | 0.16 / 0.301 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_20s (médiane / max, n=8) | 240 | 62.1 / 900 | 71.8 / 678 | 0.264 / 1.09 | 0.127 / 0.415 | 0.175 / 0.322 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_20s (médiane / max, n=8) | 600 | 319 / 2.21e+03 | 489 / 2e+03 | 0.815 / 1.89 | 0.285 / 1.92 | 0 / 0.00777 | 0 / 3.88e+03 | 0 / 1.01e+03 | 0 / 6 |
| random_biased_20s (médiane / max, n=8) | 2400 | 3.54e+03 / 9.18e+03 | 686 / 1.51e+03 | 3.27 / 6.42 | 1.22 / 3.08 | 0 / 18.3 | 0 / 5.28e+03 | 0 / 307 | 0 / 3.77 |
| random_biased_tb_20s (médiane / max, n=8) | 16 | 1.39 / 1.41 | 8.89 / 15.1 | 0.0343 / 0.186 | 0.00241 / 0.0144 | 0.28 / 0.282 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_tb_20s (médiane / max, n=8) | 64 | 3.3 / 8.47 | 4.8 / 65.5 | 0.0419 / 0.274 | 0.00689 / 0.0555 | 0.291 / 12.3 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_tb_20s (médiane / max, n=8) | 120 | 5.54 / 43.7 | 13.8 / 88 | 0.0823 / 0.361 | 0.013 / 0.114 | 0.16 / 0.301 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_tb_20s (médiane / max, n=8) | 240 | 27.9 / 230 | 53.2 / 273 | 0.262 / 0.424 | 0.118 / 0.19 | 0.175 / 0.322 | 0 / 0 | 0 / 0 | 0 / 0 |
| random_biased_tb_20s (médiane / max, n=8) | 600 | 532 / 2e+03 | 747 / 1.82e+03 | 1.56 / 7.26 | 0.715 / 2.28 | 0 / 4.22 | 0 / 1.02e+03 | 0 / 1.08e+03 | 0 / 2.31 |
| random_biased_tb_20s (médiane / max, n=8) | 2400 | 4.32e+03 / 8.35e+03 | 772 / 1.65e+03 | 4.4 / 7.56 | 1.75 / 3.08 | 0 / 51.2 | 0 / 2.52e+03 | 0 / 97.7 | 0 / 1.1 |

### 3.3 Sonde du décalage d'un tick (pas de 1 tick, 30 ticks d'échauffement ; voiture 0)
`lag0` : RocketSim(t) contre RocketGPU(t) ; `lag1` : RocketSim(t) contre RocketGPU(t-1).
| scénario | horizon | pos lag0 | pos lag1 | vel lag0 | vel lag1 |
|---|---|---|---|---|---|
| kickoff_throttle | 16 | 1.58 | 0.0932 | 10.2 | 1.57 |
| kickoff_throttle | 64 | 4.14 | 1.36 | 3.26 | 4.53 |
| kickoff_throttle | 120 | 4.64 | 3.7 | 0.733 | 5.42 |
| kickoff_throttle_steer | 16 | 1.54 | 0.0912 | 10.2 | 1.51 |
| kickoff_throttle_steer | 64 | 3.98 | 1.78 | 4.76 | 8.71 |
| kickoff_throttle_steer | 120 | 4.24 | 6.81 | 1.13 | 19.4 |
| kickoff_throttle_boost | 16 | 1.52 | 1.19 | 9.25 | 9.72 |
| kickoff_throttle_boost | 64 | 3.23 | 5.65 | 0.636 | 11.9 |
| kickoff_throttle_boost | 120 | 2.27 | 11.2 | 3.4 | 11.7 |
| kickoff_reverse | 16 | 1.59 | 0.0796 | 10.2 | 1.56 |
| kickoff_reverse | 64 | 4.11 | 1.38 | 3.26 | 4.53 |
| kickoff_reverse | 120 | 4.59 | 3.75 | 0.755 | 5.45 |
| jump_tap | 16 | 0.178 | 2.2 | 0.5 | 5.9 |
| jump_tap | 64 | 0.426 | 0.294 | 0.351 | 5.74 |
| jump_tap | 120 | 0.732 | 2.94 | 0.655 | 6.05 |
| flip_forward | 16 | 0.178 | 2.2 | 0.5 | 5.9 |
| flip_forward | 64 | 1.83 | 4.7 | 0.155 | 0.155 |
| flip_forward | 120 | 3 | 6.05 | 5.6 | 11 |

### 3.4 Sans échauffement (`--cold`), extrait
| scénario | horizon (ticks) | car0_pos | car0_vel | car0_angvel | car0_rot | car0_boost | ball_pos | ball_vel | ball_angvel |
|---|---|---|---|---|---|---|---|---|---|
| kickoff_throttle | 16 | 1.59 | 10.2 | 0.000665 | 0.000214 | 0 | 0 | 0 | 0 |
| kickoff_throttle | 120 | 4.62 | 0.738 | 0.000657 | 0 | 0 | 0 | 0 | 0 |
| jump_tap | 16 | 0.243 | 1.51 | 0.039 | 0.00617 | 0 | 0.0291 | 0.0658 | 0 |
| jump_tap | 120 | 1.5 | 1.53 | 0.00313 | 0.0183 | 0 | 0.213 | 0.476 | 0 |
| flip_forward | 16 | 0.243 | 1.51 | 0.039 | 0.00617 | 0 | 0.0291 | 0.0658 | 0 |
| flip_forward | 120 | 3.12 | 5.8 | 0.000794 | 0.0348 | 0 | 0.213 | 0.476 | 0 |

### 3.5 Boost au sol sans gaz (voiture à l'arrêt, boost pressé, boost initial 100)
Écart RocketSim moins RocketGPU, voiture 0 (`--series boost_ground_*`) : pos / vel (uu, uu/s).

| scénario | 16 ticks | 64 ticks | 120 ticks |
|---|---|---|---|
| `boost_ground_throttle0` (gaz 0) | 21,0 / 309 | 309 / 1061 | 946 / 1611 |
| `boost_ground_throttle_neg` (gaz -1) | 27,6 / 392 | 386 / 1317 | 1176 / 1998 |
| `boost_ground_throttle1` (gaz 1, contrôle) | 1,37 / 8,3 | 2,83 / 0,15 | 1,69 / 3,7 |

Vitesse de la voiture 0 après 24 ticks (pas d'un tick, départ à l'arrêt, boost 1) : RocketSim 463,7 uu/s quel que soit le gaz
(0, 0,01, 0,1, 1, -1) ; RocketGPU 6,5 (gaz 0 ou 0,01), 228,7 (gaz 0,1), 470,2 (gaz 1), -110,3 (gaz -1). Départ à 500 uu/s,
gaz 0, boost 1, 24 ticks : RocketSim 868,7 ; RocketGPU 588,7 (gaz 1 : 868,7 contre 873,3).

## 4. Écarts anormaux

### A1. Boost au sol sans gaz (important)
- Mesuré : voir 3.5. RocketSim accélère avec le boost indépendamment du gaz (même marche arrière pleine) ; RocketGPU
  applique le frein au sol et annule presque l'accélération du boost quand |gaz| <= 0,01, la module en dessous de 1, et pousse
  vers l'arrière avec gaz -1. Le boost est consommé dans les deux (même consommation, voir A4).
- Scénario minimal : `boost_ground_throttle0` (voiture à l'arrêt, `throttle=0, boost=1`, 120 ticks).
- Localisation dans la source RocketGPU (lecture, pas modification) :
  `include/rocketgpu/detail/step/raycast_wheels_and_contact_prep.hpp:2702-2722` (`real_throttle = controls.throttle`, puis
  `real_brake = 1` si |gaz| <= 0,01 sans considérer le boost) ; en l'air, ligne 3145, `air_throttle_input = is_boosting ? 1 :
  throttle` : le boost y force bien le gaz.
- Interprété (non mesuré) : RocketSim force le gaz à 1 quand la voiture boost, ce qui correspond à l'accélération mesurée
  indépendante du gaz. Je n'ai pas pu lire le source de RocketSim (binaire `.pyd`).
- Effet sur les épisodes aléatoires : forcer le gaz à 1 quand le boost est pressé (variantes `_tb`) réduit l'écart à 120 ticks
  de 74 / 217 (médiane / max, `uniform` 5 s) à 12,9 / 40,6 ; `biased` 20 s : 20,9 / 366 à 5,5 / 43,7. Le reste de l'écart n'est pas expliqué ici.

### A2. Réponse au gaz en retard d'un tick dans RocketSim
- Mesuré : départ à l'arrêt, plein gaz, pas d'un tick : vitesse RocketSim aux ticks 1..4 = 0,13 ; 13,34 ; 26,67 ; 39,89 contre
  13,34 ; 26,55 ; 39,64 ; 52,62 pour RocketGPU, y compris après 30 ticks d'échauffement. La sonde `lag1` ramène l'écart de position
  de 1,58 à 0,09 uu à 16 ticks (3.3), mais pas au-delà (6,8 uu à 120 ticks pour le braquage) : seul le tout début est un décalage pur.
- Scénario minimal : `kickoff_throttle`. Écart résiduel à 120 ticks, famille `kickoff` : 1,6 à 5,3 uu, 0,45 à 3,8 uu/s (3.1).
- Interprété : un décalage d'un tick (8,3 ms) de la réponse au gaz entre les deux simulateurs ; lequel des deux colle au jeu
  n'est pas mesuré ici (la référence du projet est RocketSim, la vérité est le jeu : Phase 5).

### A3. Ordre des pads de boost différent
- Mesuré : `rocketgpu_py.boost_pad_locations()` (34 pads) liste les 28 petits pads puis les 6 gros ; `Arena.get_boost_pads()` de
  RocketSim est dans l'ordre y croissant avec les gros pads intercalés (indices 3, 4, 15, 18, 29, 30). Même ensemble de positions,
  indices différents (par exemple le gros pad (3072, 4096) est l'indice 30 dans RocketSim, 31 dans RocketGPU).
- Conséquence : tout vecteur d'observation indexé par pad doit être permuté (T1/T2).

### A4. Constantes de boost légèrement différentes
- Mesuré (mutateurs lus par introspection) : `boost_spawn_amount` 33,3 (RocketGPU) contre 33,3333 (RocketSim) ; `boost_used_per_second` 33,3
  contre 33,3333. Écart de boost avec boost pressé : 0,282 dès 16 ticks, 0,31 à 120 ticks (3.1) : interprété comme un tick de
  consommation (0,278) plus la dérive de 0,03 pt/s due au débit ; non démontré tick par tick.

### A5. Balle posée en l'air à vitesse exactement nulle
- Mesuré : `Scenario` `ball_zero_vel` (test `test_zero_velocity_midair_ball_known_difference`) : la balle reste à z = 1500 dans
  RocketSim, tombe dans RocketGPU. Avec vz = -0,001 uu/s, les deux tombent (écart 0,2 uu à 120 ticks). Les scénarios de chute
  utilisent donc -0,001. Interprété : RocketSim garde la balle endormie ; RocketGPU la réveille (le test écrit `sleeping = 0`).
  Pas de conséquence attendue en jeu normal.

### A6. États de kickoff différents
- Mesuré : `reset_kickoff(5)` côté RocketSim place le bleu en (0, -4608, 17), côté RocketGPU `reset_to_kickoff(5)` en
  (-2048, -2560, 17,01) : le mapping graine -> spot diffère. Balle au kickoff : z = 93,15 (RocketSim) contre 92,75 (RocketGPU) ;
  voiture z = 17,0 contre 17,012 (RocketGPU « settled »). Les comparaisons ci-dessus écrivent les mêmes états explicitement.

### A7. Divergence tardive après un second événement (`hit_fast_offset`)
- Mesuré : écart voiture <= 2,8 uu jusqu'à 432 ticks, puis 8,4 uu à 456 ticks et 149 uu à 600 ticks, balle 281 uu. L'événement
  déclencheur (second contact avec le décor ou la balle) n'est pas identifié. Scénario : `hit_fast_offset`, `--series hit_fast_offset`.

## 5. Seuils des tests (section citée par le test)
Les scénarios `kickoff`, `jump`, `ball`, `hit` sont testés à 16, 120 et 240 ticks (selon la famille) avec des seuils
fixés **après** cette mesure à environ 2 à 2,5 fois le pire écart mesuré de la famille (table détaillée en commentaire
dans `tests/integration/test_physics_parity.py`, au-dessus de `TOL`). Une transcription erronée du pire écart
(`hit` à 120 ticks : 0,045 au lieu de 0,245 pour `hit_fast_offset`) a fait échouer le test ; le seuil a été recalculé
à partir de la valeur mesurée (0,245 -> 0,6), pas élargi à l'aveugle. Horizons 600 et 2400 ticks non testés : rapportés seulement.
Les scénarios aléatoires n'exigent que des états finis et aucun objet sous le sol. A1 est un `xfail(strict=True)`
(`test_boost_override_on_ground`) avec critère d'exigence 50 uu/s à 120 ticks (non issu de la mesure) : il passe au vert, donc
échoue strictement, le jour où l'écart disparaît.

## 6. Conclusion

**Mesuré.**
- Scénarios simples sans boost seul, à 120 ticks (1 s) : voiture <= 7,7 uu, <= 8,3 uu/s, orientation <= 0,06 rad, vitesse
  angulaire <= 0,1 rad/s ; balle (chute, rebonds sol/mur/plafond) <= 0,63 uu et <= 2 uu/s ; frappe de balle (premier contact) balle <= 4,8 uu
  et <= 5,4 uu/s à 240 ticks. Le chaos s'installe ensuite : à 600 ticks, jusqu'à 11 uu (balle, rebonds) et, avec contacts multiples,
  des centaines à milliers d'uu (3.2, `hit_fast_offset`).
- Contrôles aléatoires : l'écart à 120 ticks est de plusieurs dizaines à centaines d'uu (médiane `uniform` 74 uu, max 217) ; il
  tombe à 13 / 41 uu quand le gaz est forcé avec le boost. À 600 ticks et plus l'écart est du même ordre que la taille du terrain :
  aucune égalité à attendre sur épisodes longs, pour aucun des deux simulateurs.
- **Un écart de physique net : A1** (boost au sol sans gaz). Il apparaît dès le premier tick, dans une situation fréquente
  (une politique tire facilement `boost = 1` avec un gaz quelconque) et son amplitude est de l'ordre de 100 % de l'accélération concernée.

**Interprété (non mesuré).**
- A1 remet en cause la parité de RocketGPU @f0508a1 avec RocketSim 2.2.1 sur ce point précis, pas la qualité globale de la physique :
  les autres écarts mesurés sont petits ou transitoires. Réponse au critère de révision de l'ADR-001 : **partiellement déclenché** ;
  il est contournable côté entraînement/déploiement si `boost = 1` impose `gaz = 1` dans l'espace d'actions (variantes `_tb` : écart divisé
  par 3 à 6 à 120 ticks), à valider par un test de parité de l'espace d'actions (T1) et, surtout, contre le jeu (Phase 5), car
  **la référence est RocketSim, pas le jeu** : on ne sait pas ici lequel colle au jeu sur A1 et A2.
- Un correctif dans RocketGPU (ou un fork) relèverait d'une décision `tech-director` ; je n'ai rien modifié sous `vendor/`.

**Non vérifié.**
- Fidélité au jeu (aucune mesure contre Rocket League). Autres voitures que l'Octane par défaut, autres modes (2v2, hoops,
  dropshot), démolitions, buts et reset de manche, ramassage de pads au-delà du comportement incident (un décalage d'un step
  sur le ramassage d'un petit pad apparaît à 64 ticks dans `kickoff_throttle`, 12 points de boost, résorbé à 120 ticks),
  orientation de la balle (`no_ball_rot = 1` côté RocketGPU : seules position, vitesse et vitesse angulaire sont comparées),
  build CUDA (non compilable, ADR-001), reproductibilité sur d'autres machines, stabilité sur plus de 8 tirages par groupe.
