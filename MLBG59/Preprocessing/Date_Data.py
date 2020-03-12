""" Date Features processing functions:

 - all_to_date : detect dates from num/cat features and transform them to datetime format.
 - date_to_anc : Transform datetime features to timedelta according to a ref date
"""
import pandas as pd
from datetime import datetime


def all_to_date(df, var_list=None, verbose=False):
    """Detect dates from selected/all features and transform them to datetime format.
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var_list : list (Default : None)
        Names of the features
        If None, all the features
    verbose : boolean (Default False)
        Get logging information
        
    Return
    -------
    DataFrame
        Modified dataset
    """
    # if var_list = None, get all df features
    # else, exclude features if not in df
    if var_list is None:
        var_list = df.columns.tolist()
    else:
        var_list = [col for col in var_list if col in df.columns.tolist()]

    df_local = df.copy()

    if verbose:
        print('  > features : ', var_list)
        print('  > features conversion to date using "try .to_datetime')

    # for each feature in var_list, try to convert to datetime
    for col in var_list:
        try:
            if df_local[col].dtype == 'object':
                df_local[col] = pd.to_datetime(df_local[col], errors='raise')
            else:
                df_smpl = df.loc[~df[col].isna()].copy()
                df_smpl[col] = pd.to_datetime(df_smpl[col].astype('Int32').astype(str), errors='raise')
                df_local[col] = pd.to_datetime(df_local[col].astype('Int32').astype(str), errors='coerce')
        except ValueError:
            pass
        except OverflowError:
            pass
        except TypeError:
            pass

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def date_to_anc(df, var_list=None, date_ref=None, verbose=False):
    """Transform selected/all datetime features to timedelta according to a ref date
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    var_list : list (Default : None)
        List of the features to analyze.
        If None, contains all the datetime features
    date_ref : string '%d/%m/%y' (Default : None)
        Date to compute timedelta.
        If None, today date
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame
        Modified dataset
        
    list
        New timedelta features names
    """
    # if date_ref is None, use today date
    if date_ref is None:
        date_ref = datetime.now()
    else:
        date_ref = datetime.strptime(date_ref, '%d/%m/%Y')

    # if var_list = None, get all datetime features
    # else, exclude features from var_list whose type is not datetime
    l_date = df.dtypes[df.dtypes == 'datetime64[ns]'].index.tolist()
    if var_list is None:
        var_list = l_date
    else:
        var_list = [col for col in var_list if col in l_date]

    df_local = df.copy()

    # new variables names
    l_new_var_names = ['anc_' + col for col in var_list]
    # compute time delta for selected dates variables
    df_local = df_local.apply(lambda x: (date_ref - x).dt.days / 365 if x.name in var_list else x)
    # rename columns
    df_local = df_local.rename(columns=dict(zip(var_list, l_new_var_names)))

    if verbose:
        print('  ** Reference date for timelapse computing : ', date_ref)
        list(map(lambda x, y: print("  >", x + ' -> ' + y), var_list, l_new_var_names))

    return df_local, l_new_var_names
