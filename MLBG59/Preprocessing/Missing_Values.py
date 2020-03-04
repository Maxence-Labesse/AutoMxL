""" Missing values handling functions :

 - fill_numerical: replace missing values for numerical features
 - fill_categorical: replace missing values for categorical features
"""
import pandas as pd
import numpy as np


def fill_numerical(df, var_list=None, method='median', top_var_NA=True, verbose=False):
    """Fill missing values for selected/all numerical features.
    top_var_NA parameter allows to create a variable to keep track of missing values.

    Available methods : replace with zero, median or mean (Default = median)
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var_list : list (Default : None)
        names of the features to fill.
        If None, all the numerical features
    method : string (Default : 'median')
        Method used to fill the NA values :

        - zero : replace with zero
        - median : replace with median
        - mean : replace with mean

    top_var_NA : boolean (Defaut : True)
        If True, create a boolean column to keep track of missing values
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame
        Modified dataset
    """
    assert method in ['zero', 'median', 'mean'], method + ' invalid method : choose zero, median or mean'

    # if var_list = None, get all num features
    # else, remove features from var_list whose type is not num
    l_num = df._get_numeric_data().columns.tolist()

    if var_list is None:
        var_list = l_num
    else:
        var_list = [col for col in var_list if col in l_num]

    df_local = df.copy()

    # values to fill NA
    if method == 'median':
        fill_value = df_local[var_list].mean()
    elif method == 'mean':
        fill_value = df_local[var_list].mean()
    elif method == 'zero':
        fill_value = pd.Series([0] * len(var_list), index=var_list)

    for var in var_list:
        if top_var_NA:
            # keep track of NA values in Top_var_NA
            df_local['top_NA_' + var] = df_local.apply(lambda x: 1 if np.isnan(x[var]) else 0, axis=1)
        # fill NA
        df_local[var] = df_local[var].fillna(fill_value[var])

    if verbose:
        print('  > method: ' + method)
        print('  > filled features:', df[var_list].isna().sum().loc[df[var_list].isna().sum() > 0].index.tolist())

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def fill_categorical(df, var_list=None, method='NR', verbose=False):
    """Fill missing values for selected/all categorical features.
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var_list : list (Default : None)
        list of the features to fill.
        If None, contains all the categorical features
    method : string (Default : 'NR')
        Method used to fill the NA values :

        - NR : replace NA with 'NR'

    verbose : boolean (Default False)
        Get logging information
    
    Returns
    -------
    DataFrame
        Modified dataset
    """
    assert method in ['NR'], method + ' invalid method : choose NR '

    # if var_list = None, get all categorical features
    # else, remove features from var_list whose type is not categorical
    l_cat = [col for col in df.columns.tolist() if df[col].dtype == 'object']

    if var_list is None:
        var_list = l_cat
    else:
        var_list = [col for col in var_list if col in l_cat]

    df_local = df.copy()

    # values to fill NA
    if method in ['NR']:
        fill_value = 'NR'

    for var in var_list:
        df_local[var] = df_local[var].fillna(fill_value)

    if verbose:
        print('  > method: ' + method)
        print('  > filled features:', df[var_list].isna().sum().loc[df[var_list].isna().sum() > 0].index.tolist())

    return df_local