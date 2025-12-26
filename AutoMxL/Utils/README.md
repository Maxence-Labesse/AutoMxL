# Utils

## Responsabilité
Fonctions utilitaires transverses utilisées par les autres modules : affichage console formaté, mesure de temps d'exécution, et sélection aléatoire dans des dictionnaires.

## Fichiers
- `Display.py` : fonctions d'affichage console (titres, couleurs, dictionnaires)
- `Decorators.py` : décorateur `timer` pour mesurer le temps d'exécution
- `Utils.py` : sélection aléatoire de valeurs dans un dictionnaire (pour le random search)

## Flux
Pas de flux interne — ces fonctions sont appelées indépendamment par les autres modules.

## Points d'attention
- Ce module est transverse, utilisé par presque tous les autres modules
- `Utils.py` n'est pas exporté dans `__init__.py` (contrairement à `Display` et `Decorators`)
