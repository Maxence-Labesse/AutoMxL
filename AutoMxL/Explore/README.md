# Explore

## Responsabilité
Première étape du pipeline AutoMxL : analyse automatique d'un DataFrame pour détecter les types de features. Le résultat guide le preprocessing (quelles colonnes encoder, supprimer, transformer).

## Fichiers
- `Explore.py` : point d'entrée `explore()` et détection des features à faible variance
- `Features_Type.py` : fonctions de détection de type (`is_date`, `is_identifier`, etc.)

## Flux
```
DataFrame
    |
    v
explore()
    |
    ├── low_variance_features() --> features à variance nulle
    |
    └── get_features_type()
            |
            └── features_from_type() pour chaque type
                    |
                    └── is_date(), is_identifier(), is_verbatim(),
                        is_boolean(), is_categorical()
    |
    v
Dictionnaire { 'date': [...], 'identifier': [...], 'categorical': [...], ... }
```

## Points d'attention
- Le seuil `th=0.95` définit quand une colonne est considérée comme "unique" (identifiant/verbatim)
- L'ordre de détection est important : date → identifier → verbatim → boolean → categorical → numerical
- `is_date()` échantillonne 10 valeurs pour tester le parsing (performance)
