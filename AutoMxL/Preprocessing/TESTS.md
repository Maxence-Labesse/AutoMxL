# Tests — Zone Preprocessing

## Ce qui est testé

1. **test_date_encoder_converts_to_anciennete** : `date_to_anc()` convertit les dates en années et renomme avec préfixe `anc_`

2. **test_na_encoder_fills_numerical_with_mean** : `fill_numerical()` remplit les NA numériques avec la moyenne

3. **test_na_encoder_fills_categorical_with_NR** : `fill_categorical()` remplit les NA catégoriels avec 'NR'

4. **test_na_encoder_tracks_na_columns** : `fill_numerical()` crée les colonnes `top_NA_*` quand `track_num_NA=True`

5. **test_outliers_encoder_caps_extreme_values** : `replace_extreme_values()` cap les valeurs aux bornes

6. **test_outliers_encoder_aggregates_rare_categories** : `get_cat_outliers()` identifie les catégories rares

## Ce qui n'est pas testé

- `CategoricalEncoder` : nécessite PyTorch (Deep_Encoder)
- `DateEncoder` classe complète : testé via `date_to_anc()` directement
- Cas avec données vides

## Lancer les tests

```bash
python -m pytest tests/test_preprocessing.py -v
```
