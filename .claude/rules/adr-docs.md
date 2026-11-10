---
paths:
  - "docs/**"
  - "production/**"
---
# Règles : documentation et ADR
- ADR : `docs/adr/NNN-titre.md` d'après `docs/adr/TEMPLATE.md` (contexte, options, mesures, décision, conséquences).
- Chaque chiffre porte sa provenance : mesure (commande, commit, matériel, date) ou source (`fichier:ligne`, URL).
  Sinon il est marqué « estimé, non mesuré ».
- `production/state.md` reste court (< 80 lignes) : phase, gate, runs en cours, risques, prochaines actions.
