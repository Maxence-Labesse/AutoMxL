""" Missing values handling:

 - fill_num/fill_all_num: Replace missing values for numerical features
 - fill_cat/fill_all_cat: Replace missing values for categorical features
"""
from MLBG59.Utils.Utils import get_type_features


def fill_numerical(df, var_list=None, method='median', top_var_NA=True, verbose=True):
    """Fill missing values for numerical features from a list.
    You can also chose to keep track of which values were missing.

    Available methods : replace with zero, median or mean (Default = median)
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var_list : list (Default : None)
        names of the features to fill.
        If None, all the num features
    method : string (Default : 'median')
        Method used to fill the NA values :

        - zero : replace NA with zero
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
    # if var_list = None, get all numerical features
    # else, exclude features from var_list whose type is not numerical
    var_list = get_type_features(df, 'num', var_list)

    df_local = df.copy()

    if verbose:
        print('  > method: ' + method)
        print('  > filled features:', df[var_list].isna().sum().loc[df[var_list].isna().sum() > 0].index.tolist())

    if method in ['zero', 'median', 'mean']:
        # fill Na values for each feature in var_list
        for var in var_list:

            # keep track of NA values in Top_var_NA
            if top_var_NA is True and df[var].isna().sum() > 0:
                var_na = 'top_NA_' + var
                df_local[var_na] = 0
                df_local.loc[df_local[var].isna(), var_na] = 1

            # 'zero' method
            if method == 'zero':
                df_local[var] = df_local[var].fillna(0)
            # 'median' method
            elif method == 'median':
                df_local[var] = df_local[var].fillna(df_local[var].median())
            # 'mean' method
            elif method == 'mean':
                df_local[var] = df_local[var].fillna(df_local[var].mean())

        return df_local

    else:
        print(method + ' invalid method : choose zero, median or mean ')


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def fill_categorical(df, var_list=None, method='NR', verbose=1):
    """Fill missing values for categorical features from a list
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var_list : list (Default : None)
        list of the features to fill
        If None, contains all the cat features
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
    # if var_list = None, get all categorical features
    # else, exclude features from var_list whose type is not categorical
    var_list = get_type_features(df, 'cat', var_list)

    df_local = df.copy()

    if verbose > 0:
        print('  > method: ' + method)
        print('  > filled features:', df[var_list].isna().sum().loc[df[var_list].isna().sum() > 0].index.tolist())

    if method in ['NR']:
        # fill Na values for each feature in var_list
        for var in var_list:
            # 'NR' method
            if method == 'NR':
                df_local[var] = df_local[var].fillna('NR')

        return df_local

    else:
        print(method + ' invalid method : choose NR ')
