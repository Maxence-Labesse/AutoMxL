# Architecture du projet AutoMxL

## Presentation
AutoMxL est une bibliotheque Python d'Auto Machine Learning orientee classification. Elle permet de creer rapidement des POC en automatisant le pipeline complet : exploration des donnees, preprocessing, selection de features et modelisation.

## Structure globale
```
AutoMxL/
├── __main__.py        # Classe principale AML (herite de DataFrame)
├── __init__.py
├── param_config.py    # Configuration des parametres
├── Explore/           # Analyse et typage des features
├── Preprocessing/     # Nettoyage et encodage des donnees
├── Select_Features/   # Selection de features
├── Modelisation/      # Entrainement et selection de modeles
├── Start/             # Chargement et encodage de la target
└── Utils/             # Utilitaires (affichage, decorateurs)
```

## Zones principales

### Zone 1 : Core (`__main__.py`)
- Role : Classe principale `AML` qui orchestre le pipeline complet
- Herite de `pd.DataFrame`
- Methodes principales : `explore()`, `preprocess()`, `select_features()`, `model_train_test()`

### Zone 2 : Explore
- Role : Analyse du dataset pour detecter les types de features
- Fichiers : `Explore.py`, `Features_Type.py`
- Detecte : dates, identifiants, verbatims, booleens, categoriques, numeriques, NA, low_variance

### Zone 3 : Preprocessing
- Role : Nettoyage et transformation des donnees
- Fichiers : `Date.py`, `Missing_Values.py`, `Outliers.py`, `Categorical.py`, `Deep_Encoder.py`
- Encoders : DateEncoder, NAEncoder, OutliersEncoder, CategoricalEncoder

### Zone 4 : Select_Features
- Role : Reduction du nombre de features
- Fichiers : `Select_Features.py`
- Methodes : PCA (a confirmer)

### Zone 5 : Modelisation
- Role : Entrainement, hyperparameter tuning et selection du meilleur modele
- Fichiers : `HyperOpt.py`, `Bagging.py`, `Utils.py`
- Classifiers supportes : XGBOOST (autres a confirmer)

### Zone 6 : Start
- Role : Chargement des donnees et encodage de la target
- Fichiers : `Load.py`, `Encode_Target.py`
- Statut : A clarifier (semble peu utilise dans __main__.py)

### Zone 7 : Utils
- Role : Fonctions utilitaires
- Fichiers : `Display.py`, `Decorators.py`, `Utils.py`

## Dependances entre zones
- Core depend de toutes les autres zones
- Preprocessing depend de Explore (pour les listes de features par type)
- Modelisation est independante (recoit les donnees preprocessees)

## Points d'attention
- A completer apres exploration

## Ouvertures
- A completer apres exploration
