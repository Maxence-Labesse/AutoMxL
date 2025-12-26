# Tests — Utils

## Ce qui est testé
- `timer` : mesure du temps d'exécution et retour du résultat
- `random_from_dict` : sélection aléatoire dans les listes
- `random_from_dict` : conservation des valeurs non-listes

## Ce qui n'est pas testé
- `Display.py` : fonctions de print — tester que `print()` fonctionne n'apporte pas de valeur

## Lancer les tests
```bash
PYTHONPATH=. pytest tests/test_utils.py -v
```
