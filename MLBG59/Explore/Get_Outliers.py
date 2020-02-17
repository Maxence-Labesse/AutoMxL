""" Outliers detection :

 - get_cat_outliers : identify categorical features containing outliers and store their names in a list
 - get_num_outliers : identify numerical features containing outliers and store their names in a list data
"""
import pandas as pd
import numpy as np
from MLBG59.Utils.Display import *
from MLBG59.Utils.Utils import get_type_features


def get_cat_outliers(df, var_list=None, threshold=0.05, verbose=1):
    """outliers detection for categorical features

    Parameters
    ----------
     df : DataFrame
        Input dataset
     var_list : list (Default : None)
        list of the features to analyze.
        If None, contains all the categorical features
     threshold : float (Default : 0.05)
        Minimum modality frequency
     verbose : int (0/1) (Default : 1)
        Get more operations information

    Returns
    -------
     outlier_dict : dict
          {feature : list of modalities considered as outliers}
    """
    # if var_list = None, get all categorical features
    # else, exclude features from var_list whose type is not categorical
    var_list = get_type_features(df, 'cat', var_list)

    df_local = df[var_list].copy()

    if verbose > 0:
        color_print('cat features outliers identification (frequency<' + str(threshold) + ')')
        print('  > features : ', var_list,)

    # initialize output dict
    outlier_dict = {}

    # value count (frequency as number and percent for each modality) for features in var_list
    for col in df_local.columns:
        # percent
        freq_perc = pd.value_counts(df[col], dropna=False) / len(df[col])

        # if feature contain modalities with frequency < trehshold, store in outlier_dict
        if len(freq_perc.loc[freq_perc < threshold]) > 0:
            outlier_dict[col] = freq_perc.loc[freq_perc < threshold].index.tolist()

    if verbose > 0:
        print("  > containing outliers", list(outlier_dict.keys()))

    return outlier_dict


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def get_num_outliers(df, var_list=None, xstd=3, verbose=0):
    """outliers detection for num features
    
    Parameters
    ----------
     df : DataFrame
        Input dataset
     var_list : list (Default : None)
        List of the features to analyze.
        If None, contains all the num features
     xstd : int (Default : 3)
        coefficient (TODO)
     verbose : int (0/1) (Default : 1)
        Get more operations information
            
    Returns
    -------
     outlier_dict : dict
         {feature : index of outliers}
    """
    # if var_list = None, get all num features
    # else, exclude features from var_list whose type is not num
    var_list = get_type_features(df, 'num', var_list)

    df_bis = df[var_list].copy()

    if verbose > 0:
        color_print('num features outliers identification ( x: |x - mean| > '+str(xstd)+' * var)')
        print('  > features : ', var_list, )

    # initialize output dict
    outlier_dict = {}

    # compute features upper and lower limit (deviation from the mean > x*std dev (x=3 by default))
    data_std = np.std(df_bis)
    data_mean = np.mean(df_bis)
    anomaly_cut_off = data_std * xstd
    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off

    df_outliers = pd.DataFrame()

    # mask (1 if outlier, else 0)
    for col in df_bis.columns:
        df_outliers[col] = np.where((df_bis[col] < lower_limit[col]) | (df_bis[col] > upper_limit[col]), 1, 0)

    # for features containing outliers
    for col in df_outliers.sum().loc[df_outliers.sum() > 0].index.tolist():
        # store features and outliers index in outlierèdict
        outlier_dict[col] = [lower_limit[col], upper_limit[col]]

    if verbose > 0:
        print("  > containing outliers", list(outlier_dict.keys()))

    return outlier_dict
