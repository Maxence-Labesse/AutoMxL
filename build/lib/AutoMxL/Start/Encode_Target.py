"""Target encoding functions :

- category_to_target : create a target variable (1/0) from a selected category
- range_to_target : create a target variable (1/0) from a selected range
"""
import pandas as pd
import numpy as np


def category_to_target(df, var, cat):
    """Create a target variable (1/0) from a selected category

    Parameters
    ----------
    df : DataFrame
        input dataset
    var : string
        variable containing the target category
    cat : string
         target category

    Returns
    -------
    DataFrame : modified dataset
    string : new target name (var+'_'+cat)
    """
    df_local = df.copy()

    # transform variable to string if numerical
    if var in df._get_numeric_data().columns:
        df_local[var] = df_local[var].apply(str)
        cat = str(cat)

    # one hot encoding
    target_dummies = pd.get_dummies(df_local[var])
    # select cat feature
    target_dummies[var + '_' + cat] = target_dummies[cat]

    # add encoded cat feature to dataset
    df_local = pd.concat((df_local, target_dummies[var + '_' + cat]), axis=1)

    # remove var
    del df_local[var]

    return df_local, var + '_' + cat


"""
-----------------------------------------------------------------------------------------------------
"""


def range_to_target(df, var, min=None, max=None, verbose=False):
    """Create a target variable (1/0) from a selected range

    Parameters
    ----------
    df : DataFrame
        input dataset
    var : string
        variable containing the target range
    min : float
        lower limit.
        If None, no min
    max : float
        upper limit.
        If None, no max
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame : modified dataset
    string : new target name (var+'_'+lower+'_'+upper)
    """
    assert min is not None or max is not None, 'fill at least one limit parameter (lower,upper)'

    df_local = df.copy()

    # transform variable to numeric if string
    if var not in df_local._get_numeric_data().columns:
        df_local[var] = pd.to_numeric(df_local[var], errors='coerce')

    # handle None limits : replace by infinity
    if min is None:
        min = -float("inf")
    if max is None:
        max = float("inf")

    # define target name, using lower and upper values
    target_name = var + '_' + str(min) + '_' + str(max)

    # encode target
    df_local[target_name] = np.where((df_local[var] >= min) & (df_local[var] <= max), 1, 0)

    if verbose:
        print("Created target : ", target_name)
        print(df_local[target_name].value_counts().rename_axis('values').to_frame('counts'))

    # remove var
    del df_local[var]

    return df_local, target_name

