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
- Role : Detection automatique des types de features
- Fichiers : `Explore.py`, `Features_Type.py`
- Types detectes :
  - `date` : dates (via parsing)
  - `identifier` : IDs uniques (>95% uniques, meme longueur)
  - `verbatim` : texte libre (>95% uniques, longueurs variables)
  - `boolean` : exactement 2 valeurs
  - `categorical` : categories (type object ou <5 valeurs si numerique)
  - `numerical` : tout le reste
  - `low_variance` : variance nulle apres MinMaxScaler
  - `NA` : features avec valeurs manquantes

### Zone 3 : Preprocessing
- Role : Nettoyage et transformation des donnees
- Fichiers : `Date.py`, `Missing_Values.py`, `Outliers.py`, `Categorical.py`, `Deep_Encoder.py`
- Encoders :
  - `DateEncoder` : convertit les dates en timedelta depuis une date de reference
  - `NAEncoder` : remplit les NA (median/mean/zero pour num, 'NR' pour cat)
  - `OutliersEncoder` : agrege categories rares, cap les valeurs extremes (X std)
  - `CategoricalEncoder` : `one_hot` ou `deep_encoder` (embeddings via NN PyTorch)

### Zone 4 : Select_Features
- Role : Reduction de dimensionnalite
- Fichiers : `Select_Features.py`
- Methodes : `pca` (StandardScaler + PCA) ou `no_rescale_pca` (PCA seul)
- Garde les dimensions expliquant >95% de la variance

### Zone 5 : Modelisation
- Role : Entrainement, hyperparameter tuning et selection du meilleur modele
- Fichiers : `HyperOpt.py`, `Bagging.py`, `Utils.py`
- Classifiers : `RF` (RandomForest) et `XGBOOST` (XGBClassifier)
- Fonctionnalites :
  - Random search sur hyperparametres
  - Bagging optionnel
  - Selection du best model via metrique (F1, AUC) + controle delta AUC train/test

### Zone 6 : Start
- Role : Utilitaires standalone pour preparer les donnees AVANT l'instanciation de AML
- Fichiers : `Load.py`, `Encode_Target.py`
- Fonctions :
  - `import_data()` : charge fichiers csv/xlsx/txt en DataFrame
  - `category_to_target()` : transforme une categorie en target binaire (0/1)
  - `range_to_target()` : transforme une plage numerique en target binaire

### Zone 7 : Utils
- Role : Fonctions utilitaires transverses
- Fichiers : `Display.py`, `Decorators.py`, `Utils.py`
- Contenu : fonctions d'affichage console (couleurs, titres), decorateur timer

### Zone 8 : Configuration (`param_config.py`)
- Role : Parametres par defaut centralises
- Contenu :
  - Parametres deep encoder (batch_size=124, n_epoch=20, learning_rate=0.001)
  - Parametres bagging par defaut
  - Grilles d'hyperparametres RF et XGBoost

## Dependances entre zones
```
Start (standalone) --> AML (Core)
                         |
                         v
                      Explore --> Preprocessing --> Select_Features --> Modelisation
                         |              |                                    |
                         v              v                                    v
                       Utils         Utils                               Utils
                                   param_config                       param_config
```

## Points d'attention
- Vieux code (2020) : conventions et patterns potentiellement datés
- Dossier `build/` present : artefact de build, code dupliqué
- Deep Encoder utilise PyTorch : dépendance lourde
- Classification binaire uniquement (target 0/1)

## Ouvertures
- Pas de support multiclass explicite
- Pas de regression
- Seulement 2 classifiers (RF, XGBoost)
- Sélection features uniquement via PCA (pas de feature importance, recursive elimination, etc.)
