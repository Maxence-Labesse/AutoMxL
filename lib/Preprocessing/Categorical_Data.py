import pandas as pd
from lib.Utils.Utils import get_type_features


def dummy_all_var(df, var_list=None, prefix_list=None, keep=False, verbose=1):
    """
    replace categorical features from a list with dummified ones 
    
    input
    -----
     > df : datraframe
     > var_list : list (Default : None)
          list of the features to dummify
          if None, contains all the num features
     > prefix_list : list (default : None)
          prefix to add before new features name
     > keep : boolean (Default = False)
          if True, delete the original feature
     > verbose : int (0/1) (Default : 1)
          get more operations information
        
    return
    ------
     > df_local : dataframe
          le dataframe modifié
    
    """
    # if var_list = None, get all categorical features
    # else, exclude features from var_list whose type is not categorical
    var_list = get_type_features(df, 'cat', var_list)

    df_local = df.copy()

    if verbose > 0:
        print('  ** method : one hot encoding')

    for col in var_list:
        # if prefix_list == None, add column name as prefix, else add prefix_list
        if prefix_list == None:
            pref = col
        else:
            pref = prefix_list[var_list.index(col)]

        # dummify
        df_cat = pd.get_dummies(df_local[col], prefix=pref)
        # concat source DataFrame and new features
        df_local = pd.concat((df_local, df_cat), axis=1)

        # if keep = False, delete original features
        if keep == False:
            df_local = df_local.drop(col, axis=1)
        if verbose > 0:
            print('  > '+col+' ->',df_cat.columns.tolist())

    return df_local



