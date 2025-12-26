# Améliorations proposées — Utils

## Priorité 1 : Utiliser `functools.wraps` pour le décorateur
- Quoi : Remplacer la copie manuelle de `__name__` et `__doc__` par `@functools.wraps(func)`
- Pourquoi : Pattern standard Python, préserve aussi `__module__`, `__annotations__`, etc.
- Fichiers impactés : `Decorators.py`

## Priorité 2 : Renommer `print_title1`
- Quoi : Renommer en `print_title` ou `print_section_header`
- Pourquoi : Le "1" suggère d'autres versions qui n'existent pas
- Fichiers impactés : `Display.py`, et tous les fichiers qui l'utilisent (Explore, Preprocessing, Modelisation, Core)
