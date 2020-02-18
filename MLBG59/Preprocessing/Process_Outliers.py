""" Outliers handling :

 - replace_category : Replace multiple categories of a categorical variable
 - process_num_outliers
"""
import pandas as pd
import numpy as np
from MLBG59.Utils.Utils import get_type_features


def replace_category(df, var, categories, replace_with='other', verbose=True):
    """Replace multiple categories of a categorical variable
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var : string
        variable to modify
    categories : list(string)
        categories to replace
    replace_with : string
        word to replace categories with
    verbose : boolean (Default False)
        Get logging information
        
    Returns
    -------
    DataFrame
        Modified dataset
    """
    df_local = df.copy()

    for cat in categories :
        df_local.loc[df_local[var].isin(categories), var] = 'other'

    if verbose:
        print('  ** categories replaced with \''+replace_with+'\' for the variable '+var+': ')
        print(categories)

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def process_num_outliers(df, var_list, xstd=3, verbose=1):
    """Replace outliers for a list of num features
    
    Parameters
    ----------
    df : DataFrame
        Modified dataset
    var_list : list (Default : None)
        List of the features to process
        If None, contains all the numeric features
    xstd : int (Default : 3)
        Coefficient ... ?
    verbose : int (0/1) (Default : 1)
        Fet more operations information
        
    Returns
    -------
    DataFrame
        Modified dataset
    """
    # if var_list = None, get all num features
    # else, exclude features from var_list whose type is not categorical
    var_list = get_type_features(df, 'num', var_list)

    df_local = df.copy()

    # compute features upper and lower limit (deviation from the mean > x*std dev (x=3 by default))
    data_std = np.std(df_local[var_list])
    data_mean = np.mean(df_local[var_list])
    anomaly_cut_off = data_std * xstd
    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off

    df_outliers = pd.DataFrame()

    # mask (1 if outlier, else 0)
    for col in var_list:
        df_outliers[col] = np.where((df_local[col] < lower_limit[col]) | (df_local[col] > upper_limit[col]), 1, 0)

    # get features containing outliers
    outlier_var = df_outliers.sum().loc[df_outliers.sum() > 0].index.tolist()

    # replace outliers with upper_limit and lower_limit
    for col in outlier_var:
        df_local.loc[df_local[col] > upper_limit[col], col] = upper_limit[col]
        df_local.loc[df_local[col] < lower_limit[col], col] = lower_limit[col]

    if verbose > 0:
        print('  ** x outliers if  |x - mean| > '+str(xstd)+' * var')
        print('  > processed features: ',outlier_var)

    return df_local
