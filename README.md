<p align="center">
  <img width="400" height="250" src="docs/image.jpg">
</p>


# Presentation 

The main purpose of ths package is to provide a Python AutoML class that covers the complete pipeline of a classification problem from the raw dataset to a deployable model.
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

# Example of usage
### AutoML class
AutoML is built as a class inherited frm pandas DataFrame. Except for data import, each step corresponds to a class method that can be called with only default parameters or chosen ones:

In the following example, our objective is to predict 
- Import and target encoding :

If needed, you can find in <span style="color: orange"> Start </span> sub-package functions to facilitate data loading and target encoding.
```python
# import package
from MLBG59 import *

# import data into DataFrame with delimiter identification for csv and txt files
df_raw = import_data('data/bank-additional-full.csv', verbose=False)

# set "yes" category from variable "y" as the classification target.
# => get modified dataset and new target name
new_df, new_target = category_to_target(df_raw, var='y' , cat='yes')
```

### Release History

- 1.0.0 : First proper release


### Meta
Maxence Labesse - maxence.labesse@yahoo.fr
https://github.com/Maxence-Labesse/MLBG59

Distributed under the MIT license. See License.txt for more information.

