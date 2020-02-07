import pandas as pd
import numpy as np
from datetime import datetime
from MLBG59.Utils.Utils import get_type_features


def all_to_date(df, var_list=None, verbose=1):
    """
    convert string date features to datetime
    
    input
    -----
     > df : DataFrame
     > var_list : list (Default : None)
          list of the features to analyze
          if None, contains all the num features
     > verbose : int (0/1) (Default : 1)
          get more operations information
        
    return
    ------
    > df_local : DataFrame
         modified dataset
    """
    # if var_list = None, get all df features
    # else, exclude features if not in df
    var_list = get_type_features(df, 'all', var_list)
    df_local = df.copy()

    if verbose > 0:
        print('  > features : ', var_list)
        print('  > features conversion to date using "try .to_datetime')

    # for each feature in var_list, try to convert to datetime
    for col in var_list:
        try:
            if df_local[col].dtype == 'object':
                df_local[col] = pd.to_datetime(df_local[col])
            else:
                df_local[col] = pd.to_datetime(df_local[col].astype('Int32').astype(str))
                print(np.dtype(df_local[col]))
        except:
            pass
    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def date_to_anc(df, var_list=None, date_ref=None, verbose=1):
    """
    convert string date features to timelapse according to a ref date
    
    input
    -----
     > df : DataFrame
          dataset
     > var_list : list (Default : None)
          list of the features to analyze
          if None, contains all the num features
     > date_ref : string '%d/%m/%y' (Default : None)
          date to compute timelapse
          if None, today date
     > verbose : int (0/1) (Default : 1)
          get more operations information

    return
    ------
    > df_local : DataFrame
         modified dataset
        
    new_var_list : list
        contains new features name
        
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

        if verbose > 0:
            print("  >",col + ' -> ' + var_name)

    return df_local, new_var_list
