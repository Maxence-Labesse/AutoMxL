# Améliorations proposées — Start

## Priorité 1 : Gestion d'erreurs silencieuse
- Quoi : `import_data()` retourne `None` silencieusement pour les formats non supportés. `get_delimiter()` fait un `print` au lieu de lever une erreur.
- Pourquoi : Contraire aux conventions (§8 : les erreurs ne sont pas silencieuses)
- Fichiers impactés : `Load.py`

## Priorité 2 : Paramètres `min`/`max` masquent les built-ins
- Quoi : `range_to_target()` utilise `min` et `max` comme noms de paramètres
- Pourquoi : Masque les fonctions built-in Python, source de bugs potentiels
- Fichiers impactés : `Encode_Target.py`

## Priorité 3 : Code mort à supprimer
- Quoi : Le bloc `elif file.endswith('.json'): pass` ne fait rien
- Pourquoi : Code mort qui induit en erreur (refacto simple)
- Fichiers impactés : `Load.py`
