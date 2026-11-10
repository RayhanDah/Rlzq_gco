# Relais Kaggle entre les 3 membres (protocole, 2026-11-10)

Outil : `scripts/kaggle_run.py` (stdlib), modèle de kernel `kaggle/train/train_kernel.py`, tests
`tests/unit/test_kaggle_run.py`. Contexte et décisions : `docs/research/kaggle.md` §5 et §6.

**Statut : rien de ce protocole n'a été exécuté contre Kaggle.** Il a été écrit pendant un gel Kaggle (des runs de
l'utilisateur tournaient) : aucune commande `kaggle`, aucun import du module `kaggle`, aucune requête vers kaggle.com.
Les options du CLI ont été vérifiées dans le source installé (`kaggle` 2.2.5, `.venv/Lib/site-packages/kaggle/`) ;
le comportement serveur ne l'est pas (§4).

## 1. Principe
- Un run PPO = un **dataset Kaggle privé** `<propriétaire>/<slug>`, partagé en collaborateurs avec les deux autres
  membres. Il contient, à plat : `manifest.json` (run, dernier checkpoint, SHA-256, steps, historique des sessions),
  `lease.json` (verrou) et le dernier checkpoint.
- Chaque session est un kernel **du membre qui la lance**, sur son compte, avec son jeton, depuis son PC. Le kernel monte
  le dataset en entrée (`dataset_sources`) et **n'a aucun identifiant** : il ne publie rien, il écrit dans
  `/kaggle/working`.
- Le membre publie la session suivante depuis son PC (`finish`) : `kernels output`, vérifications, `datasets version`.
- Chaque version du dataset est un **instantané complet** (checkpoint + manifest + verrou), car on suppose qu'une
  version remplace tout le jeu de fichiers (non vérifié, §4). Les anciennes versions restent accessibles par
  `<propriétaire>/<slug>/<n>` : c'est l'historique de rollback.

## 2. Commandes (à blanc par défaut, `--push` pour agir)
Arguments communs : `--user <pseudo>` (le vôtre), `--run <run_id>`, `--acc t4|tpu`, `--dataset <propriétaire>/<slug>`.
`--dataset` s'ajoute à la conception initiale : le propriétaire du dataset ne se déduit ni du pseudo ni du run.

| Étape | Qui | Effet (`--push`) |
|---|---|---|
| `init` | le propriétaire, une fois | `datasets create` (privé : pas de `-u`) avec manifest vide et verrou libre. Puis, **dans l'interface web**, ajouter les 2 autres membres en collaborateurs (le CLI 2.2.5 n'a pas de commande pour ça). |
| `start` | le membre de relais | git propre et `HEAD == origin/main` (après `git fetch origin main`) ; lecture du dataset ; refus si verrou actif d'un autre (ou déjà à soi) ; nouvelle version avec notre verrou (précontrôle et relecture, §3) ; `kernels push`. Si le push échoue, le verrou est libéré automatiquement. |
| `status` | n'importe qui | lit `lease.json` et `manifest.json` ; `kernels status` si le verrou est à soi (actif ou expiré). |
| `finish` | le détenteur du verrou (actif, ou expiré sans reprise) | `kernels status` (refus si en cours ; erreur ⇒ `--salvage` requis) ; `kernels output` ; vérifie `relay-out.json` (run, nonce du verrou, numéro de session, checkpoint parent, SHA-256) ; nouvelle version : checkpoint, manifest, verrou libéré ; succès annoncé seulement après relecture de cette version. |
| `release` | le détenteur du verrou (actif, ou expiré sans reprise) | nouvelle version avec verrou libéré, sans nouveau checkpoint (le checkpoint courant est ré-envoyé tel quel). N'arrête pas le kernel (l'annuler dans l'interface web). |

`init` est une sous-commande ajoutée à la liste initiale (`start`, `status`, `finish`, `release`) : sans elle, le premier
manifest et le premier verrou devraient être écrits à la main.

Commandes CLI produites, options vérifiées dans `kaggle/cli.py` (2.2.5) :
- `kaggle datasets download <ds> -p <dir> --unzip -o [-f <fichier>]` : `cli.py:1140-1165` ;
- `kaggle datasets version -p <dir> -m <message>` : `cli.py:1198-1233` (`-m` requis ; sous-dossiers ignorés par défaut) ;
- `kaggle datasets create -p <dir>` : `cli.py:1168-1195` (privé sauf `-u`) ;
- `kaggle kernels push -p <dir> --accelerator <ID> -t <s>` : `cli.py:1414-1430` ;
- `kaggle kernels status <owner/slug>` : `cli.py:1485-1495` ; format de sortie `kaggle_api_extended.py:7674-7678` ;
- `kaggle kernels output <owner/slug> -p <dir> -o` : `cli.py:1454-1483` ; pagination suivie par l'API
  (`kaggle_api_extended.py:7575-7584`).

**Codes de sortie du CLI.** `kaggle` 2.2.5 ne sort en code non nul que sur exception (`cli.py:106-121`). Plusieurs
refus serveur sont seulement affichés, avec le code 0 : `Kernel push error` (`kaggle_api_extended.py:7300`, `:7335`),
`Dataset version creation error` (`:5955`, `:5964`), `Dataset creation error` (`:6237`). `KaggleCli` traite ces
messages comme des échecs. De plus, `datasets version` est asynchrone (« is being created », `:5962`) : un code 0 ne
prouve pas que la version est visible, d'où la relecture systématique (§3).

Métadonnées : `dataset-metadata.json` (`kaggle_api_extended.py:1143`, `id` suffit pour une version, `:5864-5868` ;
création : titre 6–50 caractères et une seule licence, `:6133-6139`) ; `kernel-metadata.json` (slug = slugify(titre),
`:7197-7204` ; `enable_tpu` lu, `:7267` ; `dataset_sources`, `:7224-7226`).

## 3. Verrou (`lease.json`) : au mieux, pas d'exclusion stricte
Contenu : `run_id`, `user`, `kernel`, `commit` (complet), `accelerator`, `nonce`, `acquired_at`, `expires_at`,
`released`. Expiration = attente en file (0 h en T4, **40 h en TPU**, d'après la file ≥ 35 h rapportée, non mesurée)
\+ timeout de session (12 h T4, 9 h TPU) + 2 h de marge.

États : `free` (absent ou libéré), `mine`, `other`, `expired`. `start` refuse `other` et `mine` ; il reprend un verrou
`expired` (le kernel de l'ancien détenteur ne pourra plus publier : `finish` exige que le verrou soit à soi et porte le
même kernel).

Un verrou à soi expiré mais **jamais repris** reste utilisable par `finish` et `release` : la dernière version porte
encore notre nonce, donc publier ne prend la place de personne. Dès qu'un autre membre l'a repris, `finish` est refusé
et la session non publiée est perdue (elle reste téléchargeable par `kaggle kernels output`).

**Le versioning d'un dataset n'est pas atomique.** Kaggle n'offre ni compare-and-swap ni version conditionnelle. Deux
membres qui lancent `start` presque en même temps peuvent tous deux lire un verrou libre. Mitigations, appliquées à
**chaque** publication (`start`, `finish`, `release`, libération automatique) :
1. **précontrôle** : juste avant `datasets version`, on relit `lease.json` et `manifest.json` ; si la tête du dataset a
   changé depuis notre lecture (un `finish` d'un autre membre, par exemple), on refuse au lieu d'écraser sa version ;
2. **relecture après écriture** : on relit la tête (jusqu'à 5 fois, 30 s d'intervalle) ; succès seulement si elle porte
   exactement notre verrou et notre manifest. `start` ne pousse le kernel qu'après cette confirmation ; s'il voit une
   autre version, il abandonne sans pousser ;
2. **nonce de session** embarqué dans le kernel et recopié dans `relay-out.json` : `finish` refuse une sortie dont le
   nonce diffère du verrou courant. Un kernel « perdant » ne peut donc pas publier ;
3. **chaîne de checkpoints** : le kernel vérifie que le checkpoint monté a l'empreinte attendue (sinon il échoue), et
   `finish` vérifie le `parent_sha256`. Une session ne peut pas être publiée sur une autre branche de la chaîne ;
4. règle humaine : annoncer le relais dans le canal du groupe avant `start`.

Fenêtres qui restent possibles (aucune ne détruit de données, car chaque version reste dans l'historique du dataset) :
- **deux kernels poussés** si les deux relectures voient chacune leur propre version (délai de propagation inégal). Un
  seul `finish` passe (nonce) ; l'autre session est perdue (quota gaspillé) ;
- **checkpoint masqué** : si le `finish` d'un membre et le `start` d'un autre tombent entre le précontrôle et la
  version de ce dernier, la version du `start` (ancien manifest) peut passer devant celle du `finish`. Le nouveau
  checkpoint n'est alors plus dans la version courante, et la session suivante repart de l'ancien. La relecture du
  `finish` ne le signale que si elle voit la version du `start` (« une autre version a remplacé la nôtre ») ; si elle
  a vu la sienne avant, rien ne l'alerte. Détection a posteriori : le manifest courant a moins de sessions que
  l'historique. Restauration à la main depuis l'historique (`<propriétaire>/<slug>/<n>`). Fenêtre de quelques
  secondes, estimée, non mesurée ; la règle humaine (4) la ferme en pratique ;
- **horloges** : l'expiration est calculée avec l'horloge du PC qui prend le verrou et comparée à celle du PC qui le
  lit. Une dérive de quelques minutes est couverte par la marge de 2 h.

## 4. Mécanismes non vérifiés (à tester plus tard, avec l'accord explicite de l'utilisateur)
Chaque point se teste sur un dataset jetable et un kernel `--train-seconds` court, une fois le gel Kaggle levé.
1. **Droits d'un collaborateur** : un collaborateur peut-il `datasets version` sur le dataset d'un autre propriétaire
   (rôle « éditeur ») ? Peut-il `datasets download` et le monter en `dataset_sources` de son propre kernel privé ?
2. **Remplacement des fichiers** : une version créée par `datasets version` contient-elle uniquement les fichiers
   envoyés (hypothèse retenue), ou fusionne-t-elle avec la version précédente ?
3. **Délai de disponibilité** : combien de temps entre `datasets version` et la visibilité de la nouvelle version par
   `datasets download` (traitement serveur) ? Le paramétrage de la relecture (5 × 30 s) en dépend.
4. **Version montée** : `dataset_sources` sans numéro monte-t-il la dernière version au moment du push, ou au démarrage
   du kernel (important en TPU, après des heures de file) ? Le contrôle d'empreinte du kernel détecte une version
   périmée, sans la corriger.
5. **Chemin de montage** : `/kaggle/input/<slug>/` ou une variante (le kernel parcourt tout `/kaggle/input`).
6. **Texte du statut** : `kernels status` affiche-t-il `KernelWorkerStatus.COMPLETE` ou `complete` ? Le parseur accepte
   les deux ; un statut inconnu bloque `finish`.
7. **Sorties d'un kernel en erreur ou au timeout** : `kernels output` renvoie-t-il les fichiers de `/kaggle/working`
   d'une session tuée par le timeout ? (`--salvage` en dépend.)
8. **Licence `other`** acceptée par `datasets create` pour un dataset privé.
9. **Durée réelle** des sessions et de la file TPU (R7, R12), qui fixe l'expiration du verrou.
10. **Concurrence** : le même membre peut-il avoir un kernel T4 et un kernel TPU actifs (R12) ? Ce relais suppose un
    seul kernel actif par run.

## 5. Risques
- **R-relais-1 (course au verrou)** : §3. Coût borné à une session perdue.
- **R-relais-2 (chaîne cassée)** : un membre ne lance pas `finish`. Le verrou expire, un autre membre reprend depuis le
  dernier checkpoint publié ; la session non publiée est alors perdue pour la chaîne, car son `finish` est refusé.
  Sans reprise, le `finish` tardif reste accepté (§3).
- **R-relais-3 (fichiers remplacés)** : si l'hypothèse §4.2 est fausse (fusion), les anciens checkpoints s'accumulent
  dans les versions ; sans gravité pour la correction, car le manifest désigne le bon fichier.
- **R-relais-4 (taille)** : chaque `start` et `release` ré-envoie le checkpoint. Négligeable pour un MLP 3×256
  (quelques Mo, estimé, non mesuré) ; à revoir si le pool self-play grossit.
- **R-relais-5 (jeton)** : le kernel, les métadonnées, le verrou et les commandes affichées sont filtrés par
  `SECRET_RE` (refus avant écriture) ; la sortie brute du CLI est affichée ligne à ligne, et toute ligne qui
  correspond à `SECRET_RE` est masquée. Côté studio, `.claude/hooks/guard_shell.py:23` refuse les commandes qui
  mentionnent le jeton. Le kernel n'a jamais d'identifiant.
- **R-relais-6 (CGU)** : chaque membre opère son propre compte (décision R11). Le pooling de quotas reste une zone grise.
- **R-relais-7 (squelette)** : le kernel actuel n'entraîne rien (`--train-seconds` 120 s par défaut, checkpoint
  factice). Le vrai learner devra respecter `.claude/rules/training.md` (contenu complet du checkpoint, reprise exacte,
  sauvegarde toutes les 20 min et avant timeout).
- **R-relais-8 (verrou bloqué)** : un `start` interrompu entre la prise du verrou et le push (coupure réseau, PC
  éteint) laisse un verrou actif sans kernel. La libération automatique ne couvre que l'échec signalé du push ; sinon
  le détenteur lance `release`, ou les autres attendent l'expiration (14 h en T4, 51 h en TPU).
