""" Outliers handling :

 - replace_category : Replace multiple categories of a categorical variable
 - replace_extreme_values : Replace values upper/lower than thresholds
"""


def replace_category(df, var, categories, replace_with='other', verbose=False):
    """Replace multiple categories of a categorical variable
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var : string
        variable to modify
    categories : list(string)
        categories to replace
    replace_with : string (Default : 'other')
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
    df_local.loc[df_local[var].isin(categories), var] = 'other'

    if verbose:
        print('  ** categories replaced with \'' + replace_with + '\' for the variable ' + var + ': ')
        print(categories)

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def replace_extreme_values(df, var, lower_th=None, upper_th=None, verbose=False):
    """Replace values with lower/upper thresholds limits
    
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
    df_local = df.copy()

    # replace values with upper_limit and lower_limit
    if upper_th is not None:
        df_local.loc[df_local[var] > upper_th, var] = upper_th
    if lower_th is not None:
        df_local.loc[df_local[var] < lower_th, var] = lower_th

    if verbose:
        print('  ** Values replaced for variable ' + var + ' :')
        print(' values lower than', lower_th)
        print(' values upper than', upper_th)

    return df_local
