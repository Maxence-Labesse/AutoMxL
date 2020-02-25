""" Date Features processing functions:

 - all_to_date : detect dates from num/cat features and transform them to datetime format.
 - date_to_anc : Transform datetime features to timedelta according to a ref date
"""
import pandas as pd
from datetime import datetime
from MLBG59.Utils.Utils import get_type_features


def all_to_date(df, var_list=None, verbose=1):
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
    var_list = get_type_features(df, 'all', var_list)
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


def date_to_anc(df, var_list=None, date_ref=None, verbose=1):
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
    var_list = get_type_features(df, 'date', var_list)

    df_local = df.copy()

    if verbose > 0:
        print('  ** Reference date for timelapse computing : ', date_ref)

    # initialisation
    new_var_list = []

    for col in var_list:
        # new feature name
        var_name = 'anc_' + col
        df_local[var_name] = (date_ref - df_local[col]).dt.days / 365
        del df_local[col]
        new_var_list.append(var_name)

        if verbose:
            print("  >", col + ' -> ' + var_name)

    return df_local, new_var_list
