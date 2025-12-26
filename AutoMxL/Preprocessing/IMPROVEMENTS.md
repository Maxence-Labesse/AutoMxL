# Améliorations proposées — Zone Preprocessing

## Priorité 1 : Import wildcard (Outliers.py)
- Quoi : Remplacer `from AutoMxL.Utils.Display import *` par un import explicite
- Pourquoi : Les imports `*` rendent les dépendances implicites (§2 conventions)
- Fichiers impactés : `Outliers.py` ligne 9

## Priorité 2 : Préfixe `l_` pour les listes
- Quoi : Les variables `l_var`, `l_num`, `l_cat`, `l_str` utilisent le préfixe hongrois `l_`
- Pourquoi : Convention datée, préférer des noms explicites (`columns`, `numeric_cols`, `categorical_cols`)
- Fichiers impactés : Tous les fichiers de la zone

## Priorité 3 : Renommer `top_NA_*` → `was_na_*`
- Quoi : Le préfixe `top_NA_` pour les colonnes de tracking n'est pas explicite
- Pourquoi : `was_na_` ou `is_imputed_` serait plus clair
- Fichiers impactés : `Missing_Values.py` ligne 130

## Priorité 4 : Séparateurs visuels
- Quoi : Remplacer les blocs `"""---"""` par des commentaires simples `# ---`
- Pourquoi : Les docstrings vides sont inhabituelles, un commentaire suffit
- Fichiers impactés : Tous les fichiers de la zone
