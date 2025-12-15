# CONVENTIONS.md

## 1. LANGUE

- Tous les identifiants sont écrits en anglais.
- Le mélange de langues dans les identifiants est interdit.

---

## 2. STRUCTURE DES FICHIERS ET MODULES

Ordre :

1. Imports / includes / dépendances
2. Configuration et constantes
3. État global ou au niveau du module
4. Logique d'initialisation
5. Points d'entrée publics ou externes
6. Logique de haut niveau
7. Détails d'implémentation
8. Utilitaires et helpers

---

## 3. CONVENTIONS DE NOMMAGE

### 3.1 Règles générales

- Les noms sont explicites et descriptifs.
- Les abréviations sont évitées sauf si universellement connues.

Identifiants interdits :
- `tmp`, `lst`, `obj`, `data`, `val`, `foo`, `bar`

---

### 3.2 Format de nommage par type

- Le format de nommage est cohérent au sein d'un même projet ou module.
- Un seul format principal est utilisé pour les variables et fonctions.

| Type d'identifiant              | Format de nommage                         |
|--------------------------------|-------------------------------------------|
| Variable                       | `snake_case` ou `camelCase`               |
| Constante                      | `UPPER_SNAKE_CASE`                        |
| Fonction / Méthode             | `snake_case` ou `camelCase`               |
| Fonction booléenne             | `is*`, `has*`, `can*`, `should*`          |
| Classe / Struct / Type         | `PascalCase`                              |
| Module / Fichier               | `snake_case` ou `kebab-case`              |
| Valeur d'énumération           | `UPPER_SNAKE_CASE`                        |

---

### 3.3 Fonctions / Méthodes

- Les noms utilisent un verbe suivi d'un complément explicite.
- Les fonctions retournant un booléen commencent par `is`, `has`, `can`, `should`.

Noms de fonctions interdits :
- `process`, `handle`, `execute`, `run`, `launch`, `do`, `doStuff`

---

## 4. TAILLE DES FONCTIONS

- Taille recommandée : 10–60 lignes
- Au-delà de 60 lignes, un découpage est envisagé

---

## 5. ORCHESTRATION ET DÉCOUPAGE

- Les fonctions point d'entrée décrivent le flux de traitement.
- Les détails d'implémentation sont délégués à des fonctions de plus bas niveau.

---

## 6. VALEURS MAGIQUES

- Les nombres magiques et valeurs codées en dur sont évités.
- Des constantes nommées sont utilisées à la place lorsque pertinent.

---

## 7. COMMENTAIRES ET DOCSTRINGS

- La langue des commentaires et docstrings est cohérente au sein d'un même projet.
- Les commentaires expliquent le pourquoi, pas le quoi.
- Les commentaires redondants avec le code sont évités.
- Les points d'entrée publics ou externes ont un commentaire ou une docstring décrivant :
  - objectif
  - entrées et sorties attendues
  - effets de bord significatifs

---

## 8. GESTION DES ERREURS

- Les erreurs ne sont pas silencieuses.
- Les blocs de gestion d'erreur vides sont interdits.
- Lorsqu'une erreur est interceptée, une action explicite est réalisée :
  - propagation
  - retour d'un état d'erreur explicite
  - log avec contexte minimum
- Les valeurs d'absence (`null`, `None`, codes de retour) sont traitées explicitement.

---

## 9. LOGS ET SORTIES

- Les sorties et logs sont centralisés lorsque applicable.
- Les appels directs dispersés sont évités.

---

## 10. TESTS (NOMMAGE)

- Les tests sont nommés de manière descriptive.
- Les noms de tests indiquent :
  - l'unité testée
  - la condition
  - le résultat attendu

Formats recommandés (choisir un format cohérent par projet) :
- `test_<unit>_<condition>_<expected>`
- `<unit>_when_<condition>_then_<expected>`

Les fichiers de tests utilisent un préfixe ou suffixe explicite, cohérent par projet :
- `test_*.py`, `*_test.c`, `*Test.java`, `*_test.*`

---

## 11. CHECKLIST

- Identifiants en anglais uniquement
- Format de nommage cohérent par projet
- Nommage explicite
- Fonctions raisonnablement découpées
- Valeurs magiques limitées
- Flux principal lisible dans les points d'entrée
- Gestion d'erreur explicite
- Logs et sorties structurés
