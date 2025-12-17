# Améliorations proposées — Start

## Priorité 1 : Gestion d'erreurs silencieuse
- Quoi : `import_data()` retourne `None` sans lever d'exception pour les formats non supportés. `get_delimiter()` fait un `print` au lieu de lever une erreur.
- Pourquoi : Contraire aux conventions (§8 : les erreurs ne sont pas silencieuses)
- Fichiers impactés : `Load.py`

## Priorité 2 : Support JSON non implémenté
- Quoi : `import_data()` déclare le support JSON mais le code est un `pass` (ligne 27)
- Pourquoi : Code mort qui induit en erreur, retourne `None` silencieusement
- Fichiers impactés : `Load.py`

## Priorité 3 : Paramètres `min`/`max` masquent les built-ins
- Quoi : `range_to_target()` utilise `min` et `max` comme noms de paramètres
- Pourquoi : Masque les fonctions built-in Python, source de bugs potentiels
- Fichiers impactés : `Encode_Target.py`

## Priorité 4 : Absence de docstrings
- Quoi : Aucune fonction n'a de docstring
- Pourquoi : Contraire aux conventions (§7 : les points d'entrée publics doivent avoir une docstring)
- Fichiers impactés : `Load.py`, `Encode_Target.py`

## Priorité 5 : Nommage incohérent
- Quoi : Variable `myCsvfile` en camelCase alors que le reste du code est en snake_case
- Pourquoi : Incohérence de style (§3.2)
- Fichiers impactés : `Load.py`
