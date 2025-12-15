# Améliorations proposées — Start

## Priorité 1 : Support JSON non implémenté
- Quoi : `import_data()` déclare le support JSON mais le code est un `pass`
- Pourquoi : Code mort qui induit en erreur, retourne `None` silencieusement
- Fichiers impactés : `Load.py`

## Priorité 2 : `get_delimiter()` ne retourne rien pour les fichiers non CSV/TXT
- Quoi : Si le fichier n'est pas `.csv` ou `.txt`, la fonction print un message mais ne retourne rien (retour implicite `None`)
- Pourquoi : Comportement silencieux, pas de gestion d'erreur explicite
- Fichiers impactés : `Load.py`

## Priorité 3 : Délimiteurs limités à `;` et `,`
- Quoi : `get_delimiter()` ne détecte que `;` et `,`, pas les autres délimiteurs courants (tab, pipe, etc.)
- Pourquoi : Limite la flexibilité du chargement de données
- Fichiers impactés : `Load.py`

## Priorité 4 : Gestion d'erreur silencieuse dans `import_data()`
- Quoi : Si le format n'est pas supporté, retourne `None` sans lever d'exception
- Pourquoi : Erreurs silencieuses contraires aux conventions (CONVENTIONS.md §8)
- Fichiers impactés : `Load.py`

## Priorité 5 : Paramètres `min`/`max` masquent les built-ins Python
- Quoi : `range_to_target()` utilise `min` et `max` comme noms de paramètres
- Pourquoi : Masque les fonctions built-in Python `min()` et `max()`
- Fichiers impactés : `Encode_Target.py`
