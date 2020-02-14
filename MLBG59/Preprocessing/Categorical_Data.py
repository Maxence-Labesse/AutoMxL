""" Categorical Features processing

 - dummy_all_var : get one hot encoded vector for each modality of each categorical features
"""
import pandas as pd
from MLBG59.Utils.Utils import get_type_features


def dummy_all_var(df, var_list=None, prefix_list=None, keep=False, verbose=1):
    """Replace categorical features from a list with dummified ones
    
    Parameters
    ----------
     df : DatraFrame
        Input dataset
     var_list : list (Default : None)
        List of the features to dummify
        If None, contains all the num features
     prefix_list : list (default : None)
        Prefix to add before new features name
     Keep : boolean (Default = False)
        If True, delete the original feature
     verbose : int (0/1) (Default : 1)
        Get more operations information
        
    Returns
    -------
     df_local : DataFrame
          Modified dataset
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



