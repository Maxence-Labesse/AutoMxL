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
    # transform variable to string if numerical
    if var in df._get_numeric_data().columns:
        df[var] = df[var].apply(str)
        cat = str(cat)

    # one hot encoding
    target_dummies = pd.get_dummies(df[var])
    # select cat feature
    target_dummies[var + '_' + cat] = target_dummies[cat]

    # add encoded cat feature to dataset
    df_local = pd.concat((df, target_dummies[var + '_' + cat]), axis=1)

    # remove var
    del df_local[var]

    return df_local, var + '_' + cat


"""
-----------------------------------------------------------------------------------------------------
"""


def range_to_target(df, var, lower=None, upper=None, verbose=False):
    """Create a target variable (1/0) from a selected range

    Parameters
    ----------
    df : DataFrame
        input dataset
    var : string
        variable containing the target range
    lower : float
        lower limit.
        If None, no lower limit
    upper : float
        upper limit.
        If None, no upper limit
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame : modified dataset
    string : new target name (var+'_'+lower+'_'+upper)
    """
    assert lower is not None or upper is not None, 'fill at least one limit parameter (lower,upper)'

    df_local = df.copy()

    # transform variable to numeric if string
    if var not in df_local._get_numeric_data().columns:
        df_local[var] = pd.to_numeric(df_local[var], errors='coerce')

    # handle None limits : replace by infinity
    if lower is None:
        lower = -float("inf")
    if upper is None:
        upper = float("inf")

    # define target name, using lower and upper values
    target_name = var + '_' + str(lower) + '_' + str(upper)

    # encode target
    df_local[target_name] = np.where((df_local[var] >= lower) & (df_local[var] <= upper), 1, 0)

    if verbose:
        print("Created target : ", target_name)
        print(df_local[target_name].value_counts().rename_axis('values').to_frame('counts'))

    # remove var
    del df_local[var]

    return df_local, target_name
