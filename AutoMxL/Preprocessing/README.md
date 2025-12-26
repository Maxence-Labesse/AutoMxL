# Preprocessing

## Responsabilité

Nettoyage et transformation des données avant modélisation. Chaque encoder suit le pattern sklearn `fit()` / `transform()` / `fit_transform()`.

## Fichiers

- `Date.py` : Conversion des dates en ancienneté (années depuis une date de référence)
- `Missing_Values.py` : Remplissage des valeurs manquantes (median/mean/zero pour num, 'NR' pour cat)
- `Outliers.py` : Traitement des outliers (agrégation des catégories rares, cap des valeurs extrêmes)
- `Categorical.py` : Encodage des variables catégorielles (one-hot ou deep embeddings)
- `Deep_Encoder.py` : Réseau de neurones PyTorch pour générer les embeddings catégoriels

## Flux

```
DataFrame brut
    │
    ├── DateEncoder ──────► Dates → ancienneté (anc_*)
    │
    ├── NAEncoder ────────► NA remplis + colonnes top_NA_* (tracking)
    │
    ├── OutliersEncoder ──► Catégories rares → 'outliers', valeurs extrêmes cappées
    │
    └── CategoricalEncoder ► Catégories → one-hot OU embeddings NN
                                │
                                └── Deep_Encoder (si method='deep_encoder')
```

## Points d'attention

- Tous les encoders modifient une copie du DataFrame (pas d'effet de bord)
- `DateEncoder` : la date de référence par défaut est `datetime.now()` — penser à fixer une date pour la reproductibilité
- `NAEncoder` : l'option `track_num_NA=True` crée des colonnes `top_NA_*` pour tracer les valeurs imputées
- `CategoricalEncoder` avec `deep_encoder` nécessite une target binaire et PyTorch
- Les paramètres du deep encoder sont dans `param_config.py`
