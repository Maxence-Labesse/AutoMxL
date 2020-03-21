""" Missing values handling functions :

 - NAEncoder (class): encoder that replaces missing values
 - fill_numerical (func): replace missing values for numerical features
 - fill_categorical (func): replace missing values for categorical features
"""
import pandas as pd
import numpy as np


class NAEncoder(object):
    """ Missing values filling

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
                 track_num_NA=True
                 ):

        assert replace_num_with in ['median', 'mean', 'zero'], 'invalid method, select median/mean/zero'
        assert replace_cat_with in ['NR'], 'invalid method, select NR'
        self.replace_num_with = replace_num_with
        self.replace_cat_with = replace_cat_with
        self.track_num_NA = track_num_NA
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
            If None, all features
        verbose : boolean (Default False)
            Get logging information
        """
        # get num and categorical columns
        l_num = [col for col in df.columns.tolist() if df[col].dtype != 'object']
        l_str = [col for col in df.columns.tolist() if df[col].dtype == 'object']

        # get list of valid features (containing NA)
        if l_var is None:
            self.l_var_cat = [col for col in l_str if df[col].isna().sum() > 0]
            self.l_var_num = [col for col in l_num if df[col].isna().sum() > 0]
        else:
            self.l_var_cat = [col for col in l_var if col in l_str and df[col].isna().sum() > 0]
            self.l_var_num = [col for col in l_var if col in l_num and df[col].isna().sum() > 0]

        # Fitted !
        self.is_fitted = True

        # verbose
        if verbose:
            print(" **method cat:", self.replace_cat_with, " / num:", self.replace_num_with)
            print("  >", len(self.l_var_cat) + len(self.l_var_num), "features to fill")
            if len(self.l_var_cat) > 0:
                print("  - cat", self.l_var_cat)
            if len(self.l_var_num) > 0:
                print("  - num", self.l_var_num)

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

        # categorical features filling
        if len(self.l_var_cat) > 0:
            df_local = fill_categorical(df_local, l_var=self.l_var_cat, method=self.replace_cat_with,
                                        verbose=verbose)

        # numerical features filling
        if len(self.l_var_num) > 0:
            df_local = fill_numerical(df_local, l_var=self.l_var_num, method=self.replace_num_with,
                                      track_num_NA=self.track_num_NA, verbose=verbose)

        # if no feature to fill
        if len(self.l_var_cat) + len(self.l_var_num) == 0 and verbose:
            print("  > no transformation to apply")

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
        # fit
        self.fit(df_local, l_var=l_var, verbose=verbose)
        # transform
        df_local = self.transform(df_local, verbose=verbose)

        return df_local

    """
    ----------------------------------------------------------------------------------------------
    """


def fill_numerical(df, l_var=None, method='median', track_num_NA=True, verbose=False):
    """Fill missing values for selected/all numerical features.
    top_var_NA parameter allows to create a variable to keep track of missing values.

    Available methods : replace with zero, median or mean (Default = median)

    Parameters
    ----------
    df : DataFrame
        Input dataset
    l_var : list (Default : None)
        names of the features to fill.
        If None, all the numerical features
    method : string (Default : 'median')
        Method used to fill the NA values :

        - zero : replace with zero
        - median : replace with median
        - mean : replace with mean

    track_num_NA : boolean (Defaut : True)
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

    if l_var is None:
        l_var = l_num
    else:
        l_var = [col for col in l_var if col in l_num]

    df_local = df.copy()

    # values to fill NA
    if method == 'median':
        fill_value = df_local[l_var].mean()
    elif method == 'mean':
        fill_value = df_local[l_var].mean()
    elif method == 'zero':
        fill_value = pd.Series([0] * len(l_var), index=l_var)

    for var in l_var:
        if track_num_NA:
            # keep track of NA values in Top_var_NA
            df_local['top_NA_' + var] = df_local.apply(lambda x: 1 if np.isnan(x[var]) else 0, axis=1)
        # fill NA
        df_local[var] = df_local[var].fillna(fill_value[var])

    if verbose:
        print('  > method: ' + method)
        print('  > filled features:', df[l_var].isna().sum().loc[df[l_var].isna().sum() > 0].index.tolist())

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def fill_categorical(df, l_var=None, method='NR', verbose=False):
    """Fill missing values for selected/all categorical features.

    Parameters
    ----------
    df : DataFrame
        Input dataset
    l_var : list (Default : None)
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

    if l_var is None:
        l_var = l_cat
    else:
        l_var = [col for col in l_var if col in l_cat]

    df_local = df.copy()

    # values to fill NA
    if method in ['NR']:
        fill_value = 'NR'

    for var in l_var:
        df_local[var] = df_local[var].fillna(fill_value)

    if verbose:
        print('  > method: ' + method)
        print('  > filled features:', df[l_var].isna().sum().loc[df[l_var].isna().sum() > 0].index.tolist())

    return df_local
