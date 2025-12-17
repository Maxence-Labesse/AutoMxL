# Tests — Start

## Ce qui est testé
- `import_data()` : chargement d'un CSV en DataFrame
- `import_data()` : retour `None` pour format non supporté
- `category_to_target()` : création de target binaire depuis une catégorie
- `range_to_target()` : création de target binaire depuis une plage numérique
- `range_to_target()` : fonctionnement avec une seule borne (min ou max)

## Ce qui n'est pas testé
- `get_delimiter()` : fonction interne utilisée par `import_data()`, testée indirectement
- Chargement XLSX : nécessiterait une dépendance openpyxl dans les tests
- Chargement JSON : non implémenté dans le code

## Lancer les tests
```bash
pytest tests/test_start.py -v
```
