# Tests — Zone Explore

## Ce qui est testé

1. **test_get_features_type_returns_all_keys** : `get_features_type()` retourne un dictionnaire complet avec toutes les clés de types attendues

2. **test_is_date_detects_datetime_column** : `is_date()` détecte correctement les colonnes datetime et rejette les colonnes numériques

3. **test_is_boolean_detects_binary_column** : `is_boolean()` détecte les colonnes à exactement 2 valeurs uniques

4. **test_is_categorical_detects_low_cardinality** : `is_categorical()` détecte les colonnes à faible cardinalité

5. **test_low_variance_features_detects_constant** : `low_variance_features()` détecte les colonnes constantes (variance nulle)

## Ce qui n'est pas testé

- `explore()` (nécessite le package complet avec torch)
- `is_identifier()` et `is_verbatim()` (cas d'usage moins fréquents)
- Cas avec valeurs manquantes (NA)
- Seuils personnalisés (th != 0.95)

## Lancer les tests

```bash
python -m pytest tests/test_explore.py -v
```
