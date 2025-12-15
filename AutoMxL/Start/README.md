# Start

## Responsabilite
Utilitaires standalone pour preparer les donnees AVANT l'instanciation de la classe AML. Ce module permet de charger des fichiers de donnees et de creer une target binaire a partir d'une variable existante.

## Fichiers
- `Load.py` : chargement de fichiers (CSV, TXT, XLSX) en DataFrame
- `Encode_Target.py` : creation de target binaire (0/1) a partir d'une variable

## Fonctions

### Load.py
- `get_delimiter(file)` : detecte le delimiteur (`;` ou `,`) d'un fichier CSV/TXT
- `import_data(file, index_col, verbose)` : charge un fichier en DataFrame

### Encode_Target.py
- `category_to_target(df, var, cat)` : cree une target binaire depuis une categorie (1 si valeur == cat, 0 sinon)
- `range_to_target(df, var, min, max, verbose)` : cree une target binaire depuis une plage numerique (1 si min <= valeur <= max, 0 sinon)

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
- `import_data()` : le support JSON est declare mais non implemente (ligne `pass`)
- `get_delimiter()` : ne detecte que `;` et `,`, pas de gestion des autres delimiteurs (tab, pipe, etc.)
- `get_delimiter()` : ne retourne rien si le fichier n'est pas CSV/TXT (pas de return explicite)
- Les deux fonctions de `Encode_Target.py` suppriment la variable source du DataFrame (`del df_local[var]`)
