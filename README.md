
<p align="center">
  <img width="400" height="250" src="docs/image.jpg">
</p>


# Presentation 

The main purpose of this package is to provide a Python AutoML class named AML that covers the complete pipeline of a binary classification project 
from a raw dataset to a deployable model.
It can be used as well as a functions/classes catalogue to ease and speed-up Data Scientists repetitive dev tasks.

You can find the whole code documentation on [MLBG59 Readthedoc documentation](https://mlbg59.readthedocs.io/en/latest/)

# Getting Started
### Prerequisites
- Python 3.7
- pandas 1.0.1
- torch 
- xgboost

### Installation
Since this package is uploaded to PyPI, it can be installed with pip using the terminal :
```
$ pip install MLBG59
```

# AML class tutorial
AML is built as a class inherited from pandas DataFrame. Each Machine Learning step corresponds to a method that can be called with default or filled parameters.

Note : 
For each method, verbose parameter allows you to get logging informations.

### Import and target encoding

If needed, you can find in <span style="color: orange"> Start </span> sub-package functions that facilitate data loading and target encoding.
```python
# import package
from MLBG59 import *

# import data into DataFrame with delimiter auto-detection for csv and txt files
df_raw = import_data('data/bank-additional-full.csv', verbose=False)

# set "yes" category from variable "y" as the classification target.
# => get modified dataset and new target name
df, target = category_to_target(df_raw, var='y' , cat='yes')

# instantiate AML object with dataset and target name
auto_df = AML(df, target=target)
```

### Explore

<span style="color: orange">explore</span> method gives you global information about the dataset and automatically
identify features types (booleans, dates, verbatims, categoricals, numericals). This information is stored in "d_features" attribute.

```python
auto_df.explore(verbose=False)

print(auto_df.d_features.keys())
> output : dict_keys(['date', 'identifier', 'verbatim', 'boolean', 'categorical', 'numerical', 'NA', 'low_variance'])
```

### Preprocess
<span style="color: orange">preprocess</span> method prepares the data before feeding it to the model :

- removes features with low variance and features identified as verbatims and identifiers
- transforms date features to numeric data (timedelta, ...)
- fills missing values
- processes categorical data (using one hot encoding or Pytorch §NN embedding encoder)
- processes outliers (optional)

```python
auto_df.preprocess(process_outliers=False, cat_method='encoder', verbose=False)
```

### Select Features (optional)
<span style="color: orange">select_features</span> method reduces the features dimension to speed up the modelisation execution time 
(may increase model performance aswell).

```python

auto_df.select_features(verbose=False)
```

### Model Train Test
model_train_test method trains and test models with random search.

- creates models with random hyper-parameters combinations from HP grid
- splits (random 80/20) train/test sets to fit/apply models
- identifies valid models |(auc(train)-auc(test)|<0.03
- gets the best model in respect of a selected metric among valid model

Available classifiers : Random Forest, XGBOOST (and bagging).

```python
d_fitted_models, l_valid_models, best_model_idx, df_model_res = auto_df.model_train_test(verbose=False)
```
output :

- d_fitted_models: dict containing models and information on test set
- l_valid_models: valid model indexes
- best_model_idx: best model index
- df_model_res: models information and metrics stored in DataFrame

\
Note : if you prefer to train and test your model separately, you can also use the following modelisation methods:
```python
auto_df.model_train(verbose=False)
d_fitted_models, l_valid_models, best_model_idx, df_model_res = auto_df.model_apply(df_sel, verbose=False)
```

### Application methods
Once you have applied preprocess and select_features, you can apply the same transformations to any iso-structure dataset using following methods:

```python
df_prep = auto_df.preprocess_apply(df, verbose=False)
df_sel = auto_df.select_features_apply(df_prep, verbose=False)
```

### Other methods
Since AML is pandas DataFrame inherited class, you can apply any DataFrame methods on it.

Note : copy() method applied on AML object will return a DataFrame. If you need to make a copy of AML object, use duplicate() method instead.


# Information
#### Release History
- 1.0.0 : First proper release 

#### Next steps
- Regression and multi-class classification

#### Licence
Distributed under the MIT license. See License.txt for more information

#### Author
Maxence Labesse - maxence.labesse@yahoo.fr

https://github.com/Maxence-Labesse/MLBG59

#### Contributors

