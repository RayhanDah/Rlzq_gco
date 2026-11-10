---
name: eval
description: "Évalue un checkpoint contre des adversaires de référence avec IC et test séquentiel ; rend un verdict de promotion."
argument-hint: "<checkpoint> [--opponents random,atba,...] [--sprt]"
---
# /eval

1. Vérifier que `scripts/eval.py` existe. Si le script cité n'existe pas encore (phase pas atteinte), le dire et s'arrêter : ne pas improviser un substitut.
2. Lancer `python scripts/eval.py $ARGUMENTS` (seeds fixes, côtés alternés bleu/orange).
3. Rapporter : nombre d'épisodes, buts pour/contre, Elo/TrueSkill avec IC 95 % (bootstrap), décision SPRT si demandée.
4. Promotion seulement si le test est significatif (protocole `docs/01_papers.md` §1.9). Une seule seed ⇒ pas de conclusion.
5. Consigner dans `docs/reports/eval-<checkpoint>.md`.
