import pandas as pd
import numpy as np

def category_to_target(df, var, cat):
    df_local = df.copy()

    if var in df._get_numeric_data().columns:
        df_local[var] = df_local[var].apply(str)
        cat = str(cat)

    target_dummies = pd.get_dummies(df_local[var])
    target_dummies[var + '_' + cat] = target_dummies[cat]

    df_local = pd.concat((df_local, target_dummies[var + '_' + cat]), axis=1)

    del df_local[var]

    return df_local, var + '_' + cat

def range_to_target(df, var, min=None, max=None, verbose=False):
    assert min is not None or max is not None, 'fill at least one limit parameter (lower,upper)'

    df_local = df.copy()

    if var not in df_local._get_numeric_data().columns:
        df_local[var] = pd.to_numeric(df_local[var], errors='coerce')

    if min is None:
        min = -float("inf")
    if max is None:
        max = float("inf")

    target_name = var + '_' + str(min) + '_' + str(max)

    df_local[target_name] = np.where((df_local[var] >= min) & (df_local[var] <= max), 1, 0)

    if verbose:
        print("Created target : ", target_name)
        print(df_local[target_name].value_counts().rename_axis('values').to_frame('counts'))

    del df_local[var]

    return df_local, target_name
