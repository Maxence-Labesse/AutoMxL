# Start

## Responsabilité
Utilitaires standalone pour préparer les données AVANT l'instanciation de la classe AML. Ce module permet de charger des fichiers de données et de créer une target binaire à partir d'une variable existante.

## Fichiers
- `Load.py` : chargement de fichiers (CSV, TXT, XLSX) en DataFrame
- `Encode_Target.py` : création de target binaire (0/1) à partir d'une variable

## Flux
```
Fichier brut --> import_data() --> DataFrame
                                      |
                                      v
                     category_to_target() ou range_to_target()
                                      |
                                      v
                          DataFrame + nom de la target
                                      |
                                      v
                                 AML(df, target)
```

## Points d'attention
- Ce module est standalone, à utiliser AVANT d'instancier AML
- Les fonctions d'encodage suppriment la variable source du DataFrame et retournent le nom de la nouvelle target
