
<p align="center">
  <img width="400" height="250" src="docs/image.jpg">
</p>


# Presentation 

The main purpose of this package is to provide a Python AutoML class named AML that covers the complete pipeline of a binary classification project 
from a raw dataset to a deployable model.
It can be used as well as a functions/classes catalogue to ease and speed-up Data Scientists repetitive dev tasks.

You can fin the whole code documentation on [Readthedoc documentation](https://mlbg59.readthedocs.io/en/latest/)

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

# AutoML tutorial
AML is built as a class inherited from pandas DataFrame. Except for data import, each step corresponds to a class method that can be 
called with default parameters or filled ones.

Note : 
verbose parameter allows you to get logging informations for all the methods

### Import and target encoding

If needed, you can find in <span style="color: orange"> Start </span> sub-package functions that facilitate data loading and target encoding.
```python
# import package
from MLBG59 import *

# import data into DataFrame with delimiter auto-detection for csv and txt files
df_raw = import_data('data/bank-additional-full.csv', verbose=False)

# set "yes" category from variable "y" as the classification target.
# => get modified dataset and new target name
new_df, new_target = category_to_target(df_raw, var='y' , cat='yes')

# instantiate AutoML object with dataset and target name
auto_df = AML(new_df.copy(), target=new_target)
```

### Explore

<span style="color: orange">explore</span> method gives you global information about the dataset (dimensions, NA containing features, low variance feature) and automatically
identify features types (booleans, dates, verbatims, categorical, numerical). This information is stored in "d_features" attribute".

```python
auto_df.explore(verbose=False)

print(auto_df.d_features.keys())
> output : dict_keys(['date', 'identifier', 'verbatim', 'boolean', 'categorical', 'numerical', 'NA', 'low_variance'])
```

### Preprocess
<span style="color: orange">preprocess</span> method prepares the data before feeding it to the model :

- remove features with low variance and features identified as verbatims and identifiers
- transform date features to numeric data (timedelta, week of the year, ...)
- fill missing values
- process categorical data
- process outliers (optional)

For categorical encoding, you can choose classical one-hot-encoding method or pytorch NN embedding encoder.

Features Transformations are stored in d_preprocess attribute.
```python
auto_df.preprocess(process_outliers=False, cat_method='encoder', verbose=False)
```

### Select Features (optional)
<span style="color: orange">select_features</span> method reduces the features dimension to speed up the modelisation execution time 
(may increase model performance aswell).

Features reduction is store in "features_selector" attribute.
```python

auto_df.select_features(verbose=True)
```

### Model Train Test
model_train_test method trains and test models with random search.

- Create models with random hyper-parameters combinations from HP grid
- split (random 80/20) train/test sets to fit/apply models
- identify valid models |(auc(train)-auc(test)|<0.03
- get the best model in respect of a selected metric among valid model

Available classifiers : Random Forest, XGBOOST (and bagging).

Models, metrics and outputs are stored in "d_hyperopt" attribute

```python
d_fitted_models, l_valid_models, best_model_idx, df_model_res = auto_df.model_train_test(verbose=False)
```
output :

- d_fitted_moels: dict containing models and information on test set: {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics', 'metrics', 'output'}
- l_valid_models: valid model indexes
- best_model_idx: best model index
- df_model_res: models information and metrics stored in DataFrame

### Application methods
Since features transformations and models are stored as class attributes, you can apply it to any iso-structure dataset :
- preprocess_apply
- select_features_apply
- model_apply

```python
df_prep = auto_df.preprocess_apply(df)
df_sel = auto_df.select_features_apply(df_prep)
```

### Other methods
Since AML is pandas DataFrame inherited class, you can apply any DataFrame methods.

Note : copy() method applied on you AML object will return a DataFrame. If you need to make a copy of AML object, use duplicate() method instead.


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

