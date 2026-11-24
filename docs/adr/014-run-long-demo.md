# ADR-014 — Run long de démo : GCO-S, 25B, tout le compute du laptop

- Statut : proposé (G3 : GO utilisateur requis)
- Date : 2026-11-24
- Décideurs : utilisateur (orientation du 2026-11-24, GO G3 à venir), agent principal

## Contexte
Il reste environ 10 jours avant la fin du projet. Aucun agent n'est encore assez fort pour une démo. Les quatre interventions
mesurées sont NEUTRES : ADR-012 (LR ×3), ADR-007 (γ 0,998), ADR-010 (tâche auxiliaire), ADR-009 (SIL v1). L'utilisateur
abandonne ADR-013 (2026-11-24) : « on n'en a plus du tout [de temps] », et demande de donner tout le compute à un seul agent.

## Options
| Option | Pour | Contre |
|---|---|---|
| A. GCO-S, run neuf, 25B (retenue) | Meilleur agent mesuré à 1B (4,31 % de victoires contre 2,40 % pour std s0, IC disjoints, une graine) ; encore en progrès entre 0,2B et 1B (1,81 % → 4,31 %) ; GCO strict ; débit plein (≈ 51k/s, pas de SIL) ; calendriers LR et entropie cohérents sur tout le run | Une seule graine ; aucune mesure au-delà de 1B |
| B. Reprendre le checkpoint GCO-S s0 à 1B | Gagne ≈ 5,5 h | Pas de démarrage à chaud dans `train.py` ; calendriers déjà annelés à 1B ; le changement de budget change le hash et casse la reprise exacte |
| C. SIL c_sil 0,1 longtemps | Pic mesuré à 0,7B (8,8 % de décisives) | Débit 32,6k/s (−36 %) ; gain neutre à 1B ; risque RAM (KS2) ; C2 impose un fondu |
| D. Kaggle T4 en relais de jour (retenue en complément, 2026-11-24) | Laptop libre le jour pour RLBot ; ≈ 55k env-steps/s mesurés en sim sur T4 | Kernel d'entraînement encore factice ; reprise XPU ↔ CUDA refusée aujourd'hui (device dans le hash) ; quota ≈ 30 h/semaine par compte (tiers, non vérifié) |

## Mesures
- Base : `docs/reports/gcos_s0.md` (GCO-S s0, 1B, `run_4a_s_s0`, éval hors ligne 20 000 épisodes).
- Débit : ≈ 51k appris/s soutenus (A7, `docs/reports/run_4a.md`), 1B ≈ 5,5 h. **25B ≈ 136 h ≈ 5,7 jours, estimé, non mesuré**
  à cette durée. Pertes du GPU : 8 en ≈ 53 h (R14), reprise automatique.
- Disque : 40 Go libres (2026-11-24). Checkpoints de reprise : 2 gardés. Instantanés d'éval ≈ 0,7 Mo × 51.

## Décision
1. Config `configs/long_run.yaml` : copie de `run_4a_s.yaml`, seuls `name` et `budget.total_timesteps` = 25B changent
   (484 393 itérations), testé (`tests/unit/test_run_long.py`). GCO strict : aucun bloc `sil` ni `aux` (refus du lanceur).
2. Éval en cours de run `configs/long_eval.yaml` : protocole S1-seed v2, un point tous les 0,5B (au lieu de 0,1B).
   Chaque point garde un instantané : c'est le vivier où choisir l'agent de démo.
3. Lancement : `vendor/venv-xpu/Scripts/python scripts/run_4a.py --variant long --seed 0 --max-hours <H>`, avec H = heures
   restantes jusqu'à l'arrêt horaire (D4). Run `long_s0`, reprise automatique après perte du GPU, veille bloquée.
   `--max-hours` compte depuis le lancement du superviseur, à travers les reprises (`supervise_run`, `scripts/run_4a.py`).
4. **Critères d'arrêt** (dans la config, ADR-005 D6) : K2 (entropie < 10 % du max, alerte à 50 %), K3 (EV ≤ 0 sur 50 it.
   après S1), K4 (KL > 0,05 sur 5 it., NaN : restauration), K5 (plus de 3 restaurations, ou 2 de suite : STOP). K7 : alerte
   seulement. Un STOP n'est pas repris sans décision humaine (`--resume-after-stop`).
   **Arrêt horaire** : le 2026-12-01 à 20 h au plus tard, que le budget soit atteint ou non. Il reste ainsi 3 jours pour
   l'export, RLBot et la démo.
5. **Critère de succès** (hors ligne, après le run, `scripts/eval.py`, 20 000 épisodes au plus) : l'agent de démo retenu
   bat GCO-S s0 à 1B (`run_4a_s_s0`, it. 19 376) en tête-à-tête, avec un SPRT +100 Elo accepté (α = β = 0,05). Le choix parmi
   les instantanés se fait par tête-à-tête, car ADR-009 §7 bis a montré un oubli possible en fin de run.
6. **Alternance (utilisateur, 2026-11-24)** : la nuit sur le laptop (XPU), le jour sur Kaggle T4 (learner CUDA, sim
   RocketGPU CPU sur 4 vCPU), un seul run `long_s0` relayé par checkpoint. Tant que le relais n'a pas de smoke vert sur T4,
   seules les nuits tournent. Le laptop sert le jour au travail RLBot.
   Préalables du relais, chacun testé :
   - (a) reprise inter-device : `device` sort de la comparaison de reprise, le reste du hash est inchangé ;
   - (b) état sim portable Windows → Linux, sinon reprise avec une sim neuve et une trace dans `segments.jsonl` ;
   - (c) kernel réel : clone du dépôt public au commit, build RocketGPU CPU (88 s mesurés, `docs/research/kaggle_probe.md`),
     meshes, `run_4a.py --variant long --max-hours <H>`, checkpoint et journaux en sortie ;
   - (d) aller-retour du checkpoint (≈ 225 Mo, mesuré) par dataset privé ;
   - (e) smoke T4 ≤ 30 min.
   La reprise n'est plus bit à bit entre machines. Le débit T4 du run complet n'est pas mesuré.
   Implémentation (2026-11-24) :
   - (a) et (b) : 6df2b00 (`checkpoint.device_change`, `versions.platform`) ;
   - (c) : `kaggle/train/long_kernel.py`. Le kernel clone le commit, installe, compile RocketGPU CPU, restaure le run,
     lance `run_4a.py --device cuda --max-hours H` (fin 45 min avant la session de 11 h 50), archive le run dans tous
     les cas ;
   - (d) : `scripts/kaggle_long.py` (`pack`, `start`, `status`, `fetch`, `sync`). Le pack fait 269 Mo à l'it. 23 544
     (mesuré, à blanc). `sync` refuse si le run local a changé depuis le pack, sauvegarde `runs/long_s0` avant d'écrire
     et ne supprime rien.
   Revue `code-reviewer` (6df2b00, 8c8730f) : REQUEST CHANGES pour un bloquant, corrigé :
   - B1 : au timeout, le kernel arrête tout le groupe de processus (SIGINT, puis SIGKILL) avant d'archiver. Re-revue
     (92c2bb1), le cas du superviseur mort avant l'enfant est corrigé : groupe vérifié vide après chaque `wait`, grâce
     ≤ marge / 4. Ce chemin ne s'exécute que sous Linux et n'est pas testé localement (pas de `killpg` sous Windows).
   Mineurs corrigés :
   - générateur du collecteur ré-amorcé (graine, itération) sur changement de device ;
   - `pack --assume-stopped` ;
   - rapport protégé si le disque est plein ;
   - `sync` exige que l'empreinte d'entrée de la sortie soit celle du pack ;
   - temporaires exclus de l'archive.
   Mineurs acceptés, sans perte possible :
   - m1 : `auto` ↔ `xpu` jette la sim ;
   - m5 : sortie bloquée si une session locale a tourné après le pack ;
   - m6 : collision de nom si un checkpoint local non sain dépasse le dernier sain ;
   - m8 : copie partielle refusée à la relance.
   Essai avant la 1re journée : `start --trial N`, reprise réelle du pack sur T4 pour N itérations, session ≤ 30 min.
   **Essai T4 PASS** (2026-11-24 08 h 28, kernel `long_s0-t4` v1, commit 1f03f3b, `start --trial 50`) :
   - Kaggle a décompressé `bundle.tar` à l'import : le kernel vérifie alors chaque fichier contre le manifest ;
   - setup en 94 s (build RocketGPU CPU 86 s) ;
   - reprise de `ckpt_000095_it00023544.pt` (`device xpu -> cuda`, `exact_env=False`), 50 itérations, checkpoint sain
     écrit, 0 NaN ;
   - Python 3.13.16, torch 2.11.0+cu130, 2 T4 vus, le learner sur un seul ;
   - débit médian 70,1k pas agents/s sur 50 it., contre 62,6k pour le laptop sur la nuit 1 (même métrique
     `perf/agent_steps_per_s`). Débit soutenu sur 11 h non mesuré.
   La sortie de l'essai n'est pas réintégrée : la 1re journée repart du même pack.
   Non vérifié avant le smoke T4 : torch 2.11+cu130 et Python 3.13 sur Kaggle, contre torch 2.15 et Python 3.12 en local ;
   chargement de l'état du générateur XPU sur CUDA ; débit.
7. Chaque session se relance avec un nouveau H. Les tests RLBot dans le vrai jeu se font le jour, sans le run. Une session
   reprend du dernier checkpoint : au plus 250 it. rejouées, ≈ 4 min (ADR-001 M8).

## Conséquences
- Plus d'ADR-013, plus de graines 1 et 2. H1 reste non concluante : ce run est un livrable de démo, pas une preuve.
- Si le succès n'est pas atteint, la démo utilise quand même le meilleur instantané, et le rapport final le dit.
- Revue `code-reviewer` (e60ba8d) : APPROVE, trois mineurs.
  - `truncate_metrics` (`src/training/run.py:265`) lit tout `metrics.jsonl` à chaque reprise : ≈ 0,8 Go en fin de run,
    estimé. À rendre en flux avant la 2e semaine.
  - K3 ne s'arme qu'après S1, donc au plus tôt à 1B avec des points tous les 0,5B.
  - Si l'arrêt horaire coupe le run avant 25B, les calendriers LR et entropie ne sont pas finis : l'agent de démo garde un
    LR et une entropie plus hauts.
- Révision si : STOP K5 ; débit < 35k/s soutenu (25B hors délai, le budget devient l'arrêt horaire) ; disque < 10 Go.
