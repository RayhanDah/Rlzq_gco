---
name: train
description: "Lance ou reprend un run d'entraînement (Kaggle ou laptop) en processus détaché, journalisé. Réservé à l'utilisateur."
argument-hint: "<config.yaml> [--resume <checkpoint>] [--target kaggle-tpu|kaggle-t4|local]"
disable-model-invocation: true
---
# /train

Préconditions, toutes obligatoires, sinon refuser :
- `/smoke-test` PASS sur le commit courant ; arbre git propre ;
- pour un run > 1B steps ou > 1 session Kaggle : Gate G3 validé par l'utilisateur (voir `production/state.md`) ;
- la config contient seed, kill criteria et le hash de reward.
Étapes :
1. Vérifier que `scripts/train.py` (local) ou `scripts/kaggle_train.py` (Kaggle) existe. Si le script cité n'existe pas encore (phase pas atteinte), le dire et s'arrêter : ne pas improviser un substitut.
2. Kaggle : `python scripts/kaggle_train.py --config $ARGUMENTS` (pousse le kernel, passe le checkpoint précédent en source,
   enregistre le slug). Local : lancer `python scripts/train.py --config ...` en arrière-plan, jamais dans un subagent.
3. Mettre à jour la section « Expériences en cours » de `production/state.md` (id du run, commit, config, cible, date).
4. Ne pas attendre la fin : `/status` suit le run.
