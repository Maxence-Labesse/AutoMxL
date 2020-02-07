import pandas as pd
import numpy as np
from MLBG59.Utils.Utils import get_type_features


def remove_bind(df, col, method='percent', threshold=0.05):
    """
    replace least represented values with 'other' for a list of categorical features
    
    input
    -----
     > df : DataFrame
         dataset
     > col : string
         name of the feature
     > method : string (Default : 'percent')
         methode for values selection
     > threshold : float (Default : 0.05)
         threshold used for method

    return
    ------
     > df_local : DataFrame
         dataset modifié
    """
    df_local = df.copy()

    # select value according to method
    if method == 'percent':
        count = pd.value_counts(df_local[col], dropna=False) / len(df_local[col])
        mask = df_local[col].isin(count[count > threshold].index)
    elif method == 'number':
        count = pd.value_counts(df_local[col], dropna=False)
        mask = df_local[col].isin(count[count > threshold].index)
    elif method == 'nbvar':
        count = pd.value_counts(df_local[col], dropna=False)
        mask = df_local[col].isin(count.head(threshold - 1).index)
    else:
        raise ValueError("Invalid parameters, sorry")

    # replace selected values with 'other'
    outliers_nb = (~mask).sum()
    df_local.loc[(~mask), col] = 'other'

    return df_local, outliers_nb


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def process_cat_outliers(df, var_list, method="percent", threshold=0.05, verbose=1):
    """
    replace outliers for a list of categorical features
    
    input
    -----
     > df : DataFrame
         dataset
     > var_list : list (Default : None)
         list of the features to process
         if None, contains all the cat features
     > method : string (Default : 'percent')
         methode for values selection
     > threshold : float (Default : 0.05)
         threshold used for method
     > verbose : int (0/1) (Default : 1)
          get more operations information
        
    return
    ------
    df_local : DataFrame
        dataset modifié
    """
    # if var_list = None, get all categorical features
    # else, exclude features from var_list whose type is not categorical
    var_list = get_type_features(df, 'cat', var_list)

    df_local = df.copy()

    outliers_list = list()
    for col in var_list:
        # apply remove_bind (replace least represented values with 'other')
        df_local, outliers_num = remove_bind(df_local, col, method, threshold)
        if outliers_num > 0: outliers_list.append(col)
    if verbose > 0:
        print('  ** method: ' + method)
        print('  > processed features:',outliers_list)
    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def process_num_outliers(df, var_list, xstd=3, verbose=1):
    """
    replace outliers for a list of num features
    
    input
    -----
     > df : DataFrame
          dataset
     > var_list : list (Default : None)
          list of the features to process
          if None, contains all the numeric features
     > xstd : int (Default : 3)
          coefficient ... ?
     > verbose : int (0/1) (Default : 1)
          get more operations information
        
    return
    ------
     > df_local : DataFrame
          dataset modifié
        
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
