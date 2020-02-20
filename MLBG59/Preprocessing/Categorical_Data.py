""" Categorical features processing

 - dummy_all_var : get one hot encoded vector for each category of a categorical features list
 - label encoding : coming soon
"""
import pandas as pd
from MLBG59.Utils.Utils import get_type_features


def dummy_all_var(df, var_list=None, prefix_list=None, keep=False, verbose=1):
    """Get one hot encoded vector for selected/all categorical features
    
    Parameters
    ----------
     df : DatraFrame
        Input dataset
     var_list : list (Default : None)
        Names of the features to dummify
        If None, all the num features
     prefix_list : list (default : None)
        Prefix to add before new features name (prefix+'_'+cat).
        It None, prefix=variable name
     Keep : boolean (Default = False)
        If True, delete the original feature
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

    if verbose:
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
        if verbose:
            print('  > ' + col + ' ->', df_cat.columns.tolist())

    return df_local
