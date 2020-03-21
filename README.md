<p align="center">
  <img width="400" height="250" src="docs/image.jpg">
</p>


# Presentation 

The main purpose of this package is to provide a Python AutoML class that covers the complete pipeline of a binary classification project 
from a raw dataset to a deployable model.
It can be used as well as a functions/classes catalogue to ease and speed-up the Data Scientists repetitive dev tasks.

You can fin the whole code documentation on [Readthedoc documentation](https://mlbg59.readthedocs.io/en/latest/)

# Getting Started
### Prerequisites
- Python 3.7


### Installation
Since this package is uploaded to PyPI, it can be installed with pip using the terminal :
```
$ pip install MLBG59
```

# AutoML tutorial
AutoML is built as a class inherited from pandas DataFrame. Except for data import, each step corresponds to a class method that can be 
called with only default parameters or filled ones.

Note : you can specify verbose=True in each method to get logging informations.

### Import and target encoding :

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
auto_df = AutoML(new_df.copy(), target=new_target)
```

### Data exploration :

<span style="color: orange">explore</span> method allows you to get global information about the dataset (features types,
missing values, low variance variables). This information is also stored in object attribute dict "d_features".

```python
auto_df.explore(verbose=False)

print(auto_df.d_features.keys())
> output : dict_keys(['numerical', 'date', 'categorical', 'NA', 'low_variance'])
```
\
You can also call <span style="color: orange">get_outliers</span> method to perform an outliers analysis on categorical 
(categories frequency) or numerical features (extrem values). Information is stored in object attributes dict d_num_outliers and d_cat_outliers.
```python
auto_df.get_outliers(verbose=False)
```

### Preprocessing
<span style="color: orange">preprocess</span> method prepares the data before feeding it to the model :

- remove low variance features
- transform date features to timedelta
- fill missing values
- process categorical data
- replace outliers (optional)

For categorical encoding, you can choose classical one-hot-encoding method or pytorch nn embedding encoder
```python
auto_df.preprocess(process_outliers=False, cat_method='encoder', verbose=False)
```

### Features Selection
<span style="color: orange">select_features</span> method reduce the features dimension to speed up the modelisation execution time 
(may increase model performance aswell): 


```python
auto_df.select_features(verbose=True)
```

### Modelisation
Model hyper-optimisation with random search.

- Creates random hyper-parameters combinations from HP grid
- train and test a model for each combination
- get the best model in respect of a selected metric among valid model


Available classifiers : Random Forest, XGBOOST (and bagging)
```python
auto_df.train_model(verbose=False)
```
outputs :
- dict : {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics', 'metrics', 'output'}
- int : best model index
- DataFrame : Models information and metrics

### Methods configuration
All the methods parameters and hyper-parameters are stored in the file config.py

# Catalogue
TODO


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

