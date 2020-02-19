<p align="center">
  <img width="400" height="250" src="docs/image.jpg">
</p>


# Presentation 

The main purpose of ths package is to provide a Python AutoML class that covers the complete pipeline of a classification problem 
from the raw dataset to a deployable model.
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
AutoML is built as a class inherited frm pandas DataFrame. Except for data import, each step corresponds to a class method that can be 
called with only default parameters or chosen ones.

Note : you can specify verbose=True in each method to get logging informations.

####  Import and target encoding :

If needed, you can find in <span style="color: orange"> Start </span> sub-package functions to facilitate data loading and target encoding.
```python
# import package
from MLBG59 import *

# import data into DataFrame with delimiter identification for csv and txt files
df_raw = import_data('data/bank-additional-full.csv', verbose=False)

# set "yes" category from variable "y" as the classification target.
# => get modified dataset and new target name
new_df, new_target = category_to_target(df_raw, var='y' , cat='yes')

# instantiate AutoML object with dataset and target name
auto_df = AutoML(new_df.copy(), target=new_target)
```

#### Data exploration :

<span style="color: orange">recap</span> method allows to get global informations about the dataset (features types,
missing values, low variance variables). This information is also stored in object attribute dict "d_features".

```python
auto_df.recap(verbose=False)

print(auto_df.d_features.keys())
output : dict_keys(['numerical', 'date', 'categorical', 'NA', 'low_variance'])
```
\
You can also call <span style="color: orange">get_outliers</span> method to perform an outliers analysis on categorical 
(categories frequency) or numerical features (extrem values). Information is stored in object attributes dict d_num_outliers and d_cat_outliers
```python
auto_df.get_outliers(verbose=False)
```

#### Preprocessing

<span style="color: orange">preprocess</span> method

### Release History

- 1.0.0 : First proper release

# Information 
### Licence
Distributed under the MIT license. See License.txt for more information

### Author
Maxence Labesse - maxence.labesse@yahoo.fr
https://github.com/Maxence-Labesse/MLBG59

### Contributors

n.

