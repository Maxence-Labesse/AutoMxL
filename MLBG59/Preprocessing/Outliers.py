""" Outliers handling functions

 - OutliersEncoding (class) : identify and replace outliers
 - get_cat_outliers (funct): identify categorical features containing outliers
 - get_num_outliers (func): identify numerical features containing outliers
 - replace_category (func): replace categories of a categorical variable
 - replace_extreme_values (func): replace extreme values (oh!)
"""
import pandas as pd
import numpy as np
from MLBG59.Utils.Display import *


class OutliersEncoder(object):
    """
    Parameters
    ----------

    """

    def __init__(self,
                 cat_threshold=0.02,
                 num_xstd=4
                 ):

        self.cat_threshold = cat_threshold,
        self.num_xstd = num_xstd
        self.is_fitted = False
        self.l_var_num = []
        self.l_var_cat = []
        self.d_num_outliers = {}
        self.d_cat_outliers = {}

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit(self, df, l_var, verbose=False):
        """Fit encoder

        Parameters
        ----------
        df : DataFrame
            input dataset
        l_var : list
            features to encode.
            If None, all features identified as dates (see Features_Type module)
        verbose : boolean (Default False)
            Get logging information
        """
        l_num = [col for col in df.columns.tolist() if df[col].dtype != 'object']
        l_str = [col for col in df.columns.tolist() if df[col].dtype == 'object']

        if l_var is None:
            self.l_var_cat = [col for col in l_str]
            self.l_var_num = [col for col in l_num]
        else:
            self.l_var_cat = [col for col in l_var if col in l_str and df[col].nunique()>2]
            self.l_var_num = [col for col in l_var if col in l_num and df[col].nunique()>2]

        if len(self.l_var_cat) > 0:
            self.d_cat_outliers = get_cat_outliers(df, l_var=self.l_var_cat, threshold=self.cat_threshold,
                                                   verbose=False)

        if len(self.l_var_num) > 0:
            self.d_num_outliers = get_num_outliers(df, l_var=self.l_var_num, xstd=self.num_xstd, verbose=False)

        self.is_fitted = True

        # verbose
        if verbose:
            print(" **method cat: frequency<" + str(self.cat_threshold)
                  + " / num:( x: |x - mean| > " + str(self.num_xstd) + "* var)")
            print("  >", len(self.d_cat_outliers.keys()) + len(self.d_num_outliers.keys()), "features with outliers")
            if len(self.d_cat_outliers.keys()) > 0:
                print("  - cat", list(self.d_cat_outliers.keys()))
            if len(self.d_num_outliers.keys()) > 0:
                print("  - num", list(self.d_num_outliers.keys()))

    """
    ----------------------------------------------------------------------------------------------
    """

    def transform(self, df, verbose=False):
        """Transform dataset categorical features using the encoder.
        Can be done only if encoder has been fitted

        Parameters
        ----------
        df : DataFrame
            dataset to transform
        verbose : boolean (Default False)
            Get logging information
        """
        assert self.is_fitted, 'fit the encoding first using .fit method'
        df_local = df.copy()

        if len(list(self.d_cat_outliers.keys())) > 0:
            if verbose :
                print(" - cat aggregated values:")
            for col in self.d_cat_outliers.keys():
                df_local = replace_category(df_local, col, self.d_cat_outliers[col], replace_with='outliers',
                                            verbose=verbose)

        if len(list(self.d_num_outliers.keys())) > 0:
            if verbose:
                print(" - num values replaces:")
            for col in self.d_num_outliers.keys():
                df_local = replace_extreme_values(df_local, col, self.d_num_outliers[col][0],
                                                  self.d_num_outliers[col][1], verbose=verbose)

        if len(list(self.d_cat_outliers.keys())) + len(list(self.d_num_outliers.keys())) == 0:
            print("  > no outlier to replace")

        return df_local

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit_transform(self, df, l_var, verbose):
        """Fit and transform dataset with encoder

        Parameters
        ----------
        df : DataFrame
            input dataset
        l_var : list
            features to encode.
            If None, all features identified as dates (see Features_Type module)
        verbose : boolean (Default False)
            Get logging information
        """
        df_local = df.copy()
        self.fit(df_local, l_var=l_var, verbose=verbose)
        df_local = self.transform(df_local, verbose=verbose)

        return df_local


"""
----------------------------------------------------------------------------------------------
"""


def get_cat_outliers(df, l_var=None, threshold=0.05, verbose=False):
    """Outliers detection for selected/all categorical features.

    Method : Modalities with frequency <x% (Default 5%)

    Parameters
    ----------
     df : DataFrame
        Input dataset
     l_var : list (Default : None)
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

    if l_var is None:
        l_var = l_cat
    else:
        l_var = [col for col in l_var if col in l_cat]

    df_local = df[l_var].copy()

    # dict containing value_counts for each variable
    d_freq = {col: pd.value_counts(df[col], dropna=False, normalize=True) for col in l_var}

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


def get_num_outliers(df, l_var=None, xstd=3, verbose=False):
    """Outliers detection for selected/all numerical features.

    Method : x outlier <=> abs(x - mean) > xstd * var

    Parameters
    ----------
     df : DataFrame
        Input dataset
     l_var : list (Default : None)
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

    if l_var is None:
        l_var = l_num
    else:
        l_var = [col for col in l_var if col in l_num]

    df_local = df[l_var].copy()

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

    if verbose:
        color_print('num features outliers identification ( x: |x - mean| > ' + str(xstd) + ' * var)')
        print('  > features : ', l_var)
        print("  > containing outliers", list(d_outliers.keys()))

    return d_outliers


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def replace_category(df, var, categories, replace_with='outliers', verbose=False):
    """Replace categories of a categorical variable
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var : string
        variable to modify
    categories : list(string)
        categories to replace
    replace_with : string (Default : 'outliers')
        word to replace categories with
    verbose : boolean (Default False)
        Get logging information
        
    Returns
    -------
    DataFrame
        Modified dataset
    """
    df_local = df.copy()

    # replace categories
    df_local.loc[df_local[var].isin(categories), var] = replace_with

    if verbose:
        print('  > ' + var + ' ', categories)

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def replace_extreme_values(df, var, lower_th=None, upper_th=None, verbose=False):
    """Replace extrem values : > upper threshold or < lower threshold
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var : string
        variable to modify
    lower_th : int/float (Default=None)
        lower threshold
    upper_th : int/float (Default=None)
        upper threshold
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame
        Modified dataset
    """
    assert (lower_th is not None or upper_th is not None), 'specify at least one limit value'
    df_local = df.copy()

    # replace values with upper_limit and lower_limit
    if upper_th is not None:
        df_local.loc[df_local[var] > upper_th, var] = upper_th
    if lower_th is not None:
        df_local.loc[df_local[var] < lower_th, var] = lower_th

    if verbose:
        print('  > ' + var + ' < ' + str(round(lower_th, 4)) + ' or > ' + str(
            round(upper_th, 4)))

    return df_local
