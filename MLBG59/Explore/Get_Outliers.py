""" Outliers detection functions :

 - get_cat_outliers : identify categorical features containing outliers
 - get_num_outliers : identify numerical features containing outliers
"""
import pandas as pd
import numpy as np
from MLBG59.Utils.Display import *


def get_cat_outliers(df, var_list=None, threshold=0.05, verbose=False):
    """Outliers detection for selected/all categorical features.

    Method : Modalities with frequency <x% (Default 5%)

    Parameters
    ----------
     df : DataFrame
        Input dataset
     var_list : list (Default : None)
        Names of the features
        If None, all the categorical features
     threshold : float (Default : 0.05)
        Minimum modality frequency
     verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    dict
        {variable : list of categories considered as outliers}
    """
    # if var_list = None, get all categorical features
    # else, remove features from var_list whose type is not categorical
    l_cat = [col for col in df.columns.tolist() if df[col].dtype == 'object']

    if var_list is None:
        var_list = l_cat
    else:
        var_list = [col for col in var_list if col in l_cat]

    df_local = df[var_list].copy()

    # dict containing value_counts for each variable
    d_freq = {col: pd.value_counts(df[col], dropna=False, normalize=True) for col in var_list}

    # if features contain at least 1 outlier category (frequency <threshold)
    # store outliers categories in dict
    d_outliers = {k: v[v < threshold].index.tolist()
                  for k, v in d_freq.items()
                  if len(v[v < threshold]) > 0}

    if verbose:
        color_print('cat features outliers identification (frequency<' + str(threshold) + ')')
        print('  > features : ', df_local.columns, )
        print("  > containing outliers", list(d_outliers.keys()))

    return d_outliers


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def get_num_outliers(df, var_list=None, xstd=3, verbose=False):
    """Outliers detection for selected/all numerical features.

    Method : x outlier <=> abs(x - mean) > xstd * var
    
    Parameters
    ----------
     df : DataFrame
        Input dataset
     var_list : list (Default : None)
        Names of the features
        If None, all the num features
     xstd : int (Default : 3)
        Variance gap coef
     verbose : boolean (Default False)
        Get logging information
            
    Returns
    -------
    dict
        {variable : [lower_limit, upper_limit]}
    """
    # if var_list = None, get all num features
    # else, remove features from var_list whose type is not num
    l_num = df._get_numeric_data().columns.tolist()

    if var_list is None:
        var_list = l_num
    else:
        var_list = [col for col in var_list if col in l_num]

    if verbose:
        color_print('num features outliers identification ( x: |x - mean| > ' + str(xstd) + ' * var)')
        print('  > features : ', var_list, )

    df_local = df[var_list].copy()

    # compute features upper and lower limit (abs(x - mean) > xstd * var (x=3 by default))
    data_std = np.std(df_local)
    data_mean = np.mean(df_local)
    anomaly_cut_off = data_std * xstd
    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    data_min = np.min(df_local)
    data_max = np.max(df_local)

    # store variables and lower/upper limits
    d_outliers = {col: [lower_limit[col], upper_limit[col]]
                  for col in df_local.columns.tolist()
                  if (data_min[col] < lower_limit[col] or data_max[col] > upper_limit[col])}

    return d_outliers
