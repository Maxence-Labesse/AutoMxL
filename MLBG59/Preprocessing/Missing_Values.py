""" Missing values handling functions :

 - NAEncoder (class): encoder that replaces missing values
 - fill_numerical (func): replace missing values for numerical features
 - fill_categorical (func): replace missing values for categorical features
"""
import pandas as pd
import numpy as np


class NAEncoder(object):
    """ Encoder that replaces missing values

    Available methods to replace missing values
    - num : metdian/mean/zero
    - cat : 'NR'

    Parameters
    ----------
    replace_num_with: string
        method used to replace numerical missing values
    replace_cat_with: string
        method used to replace categorical missing values
    """

    def __init__(self,
                 replace_num_with='median',
                 replace_cat_with='NR',
                 ):

        self.replace_num_with = replace_num_with
        self.replace_cat_with = replace_cat_with
        self.l_var_cat = []
        self.l_var_num = []
        self.is_fitted = False

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit(self, df, l_var, verbose=False):
        """fit encoder

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
        l_num = [col for col in df.column.tolist() if df[col].dtype != 'object']
        l_str = [col for col in df.column.tolist() if df[col].dtype == 'object']

        if l_var is None:
            self.l_var_cat = l_str
            self.l_var_num = l_num
        else:
            self.l_var_cat = [col for col in l_var if col in l_str]
            self.l_var_num = [col for col in l_var if col in l_num]

        if len(self.l_var_cat) > 0 or len(self.l_var_num) > 0:
            self.is_fitted = True

        if verbose:
            print("features to encode")
            print("cat :", self.l_var_cat)
            print("num :", self.l_var_num)

    """
    ----------------------------------------------------------------------------------------------
    """

    def transform(self, df, verbose=False):
        """ transform dataset categorical features using the encoder.
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

        if len(self.l_var_cat) > 0:
            df_local = fill_categorical(df_local, var_list=self.l_var_cat, method=self.replace_cat_with,
                                        verbose=verbose)

        if len(self.l_var_num) > 0:
            df_local = fill_numerical(df, var_list=self.l_var_num, method=self.replace_num_with, top_var_NA=True,
                                      verbose=verbose)

        return df_local

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit_transform(self, df, l_var, verbose=False):
        """fit and transform dataset with encoder

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
