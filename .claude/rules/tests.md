---
paths:
  - "tests/**"
---
# Règles : tests
- Structure Arrange / Act / Assert. Pas de réseau, pas de Kaggle, pas de GPU dans `tests/unit` ; total < 60 s.
- Un bug corrigé = un test de régression vu ÉCHOUER avant le fix (le noter dans le message de commit).
- Ne jamais affaiblir une assertion pour faire passer un test. Une tolérance numérique a une justification écrite.
- Tests de propriété pour les invariants (somme nulle de la reward, miroir, reprise exacte).
