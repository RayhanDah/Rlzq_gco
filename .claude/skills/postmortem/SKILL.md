---
name: postmortem
description: "Rédige le postmortem d'un run ou d'un incident : faits, mesures, cause, décision, actions."
argument-hint: "<run id ou incident>"
---
# /postmortem

1. Rassembler : config, commit, journal `production/run_log.jsonl`, courbes TensorBoard, évals.
2. Déléguer l'analyse chiffrée à `experiment-analyst`.
3. Écrire `docs/postmortems/<date>-<id>.md` : hypothèse testée, résultat (avec IC), kill criteria atteints ou non,
   cause probable (séparer prouvé / supposé), décision, actions avec responsable.
4. Mettre à jour `production/state.md` (risques, prochaines actions).
