# AutoML

The ambition of this project is to provide a Python library that automate the differents steps of a Machine Learning classification usecase
- Load 
- Audit
- Clean and process
- Model

It can be used aswell as a catalog of functions to ease and speed up repetitive Data Scientists tasks

AutoML is built as a class inherited from pandas dataframe one, allowing user to get some more attributes that describes the datas.

## Getting Started



### Prerequisites

- Python 3.7


### Exemple of use

```
# Load data
df_raw = load_data('data/data')

# Create autoML object
df = AutoML(df_raw.copy(),target = 'y_yes')

# automated classification task from audit to modeling
df.audit(verbose=1)

df.get_outliers(verbose=1)

df.preprocess(process_outliers=True, verbose=1)

df.train_predict(verbose=1)
```