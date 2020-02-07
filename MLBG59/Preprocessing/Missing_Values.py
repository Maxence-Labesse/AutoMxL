from MLBG59.Utils.Utils import get_type_features


def fill_num(df, var, method='median', top_var_NA=True):
    """
    fill missing values of a num feature
    
    input
    -----
     > df : DataFrame
          dataset
     > var : string
          feature to fill
     > method : string (Default : 'median')
          method used to fill the NA values :
            - zero : replace NA with zero
            - median : replace with median
            - mean : replace with mean
            - regression : replace with regression predictions
     > top_var_NA : boolean (Defaut : True)
          if True, create a boolean column to identify replaced observations

    return
    ------
    df_local : DataFrame
        modified dataset
    """
    df_local = df.copy()

    if method in ['zero', 'median', 'mean', 'reg']:
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
        # 'regression' method
        elif method == 'regression':
            print('coming soon')

    else:
        print(method + ' méthode invalide : choisir entre zero/mean/median/reg ')

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def fill_all_num(df, var_list=None, method='median', top_var_NA=True, verbose=1):
    """
    fill missing values for numerical features from a list
    
    input
    -----
     > df : DataFrame
          dataset
     > var_list : list (Default : None)
          list of the features to fill
          if None, contains all the num features
     > method : string (Default : 'median')
          method used to fill the NA values :
              - zero : replace NA with zero
              - median : replace with median
              - mean : replace with mean
              - regression : replace with regression predictions
     > top_var_NA : boolean (Defaut : True)
          if True, create a boolean column to identify replaced observations
     > verbose : int (0/1) (Default : 1)
          get more operations information

    return
    ------
    df_local : DataFrame
        modified dataset
    """
    # if var_list = None, get all numerical features
    # else, exclude features from var_list whose type is not numerical
    var_list = get_type_features(df, 'num', var_list)

    df_local = df.copy()

    if verbose > 0:
        print('  > method: ' + method)
        print('  > filled features:',df[var_list].isna().sum().loc[df[var_list].isna().sum()>0].index.tolist())

    # fill Na values for each feature in var_list
    for j in var_list:
        df_local = fill_num(df_local, j, method, top_var_NA)

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def fill_cat(df, var, method='NR'):
    """
    fill missing values of a categorical feature
    
    input
    -----
     > df : Dataframe
         dataset
     > var : string
         feature to fill
     > method : string (Default : 'NR')
         method used to fill the NA values :
             - NR : replace NA with 'NR'
             - regression : replace with regression predictions
     > top_var_NA : boolean (Defaut : True)
         if True, create a boolean column to identify replaced observations

    return
    ------
     > df_local : DataFrame
         modified dataset
    
    """
    df_local = df.copy()

    if method in ['NR', 'reg']:

        # 'NR' method
        if method == 'NR':
            df_local[var] = df_local[var].fillna('NR')
        # 'regression' method
        elif method == 'regression':
            pass

    else:
        print(method + ' méthode invalide : choisir entre NR/reg ')

    return df_local


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def fill_all_cat(df, var_list=None, method='NR', verbose=1):
    """
    fill missing values for categorical features from a list
    
    input
    -----
    > df : DataFrame
        dataset
    > var_list : list (Default : None)
         list of the features to fill
         if None, contains all the cat features
    > method : string (Default : 'NR')
         method used to fill the NA values :
            - NR : replace NA with 'NR'
            - regression : replace with regression predictions (coming soon)
    > verbose : int (0/1) (Default : 1)
        get more operations information
    
    return
    ------
    > df_local : DataFrame
        modified dataset
    
    """
    # if var_list = None, get all categorical features
    # else, exclude features from var_list whose type is not categorical
    var_list = get_type_features(df, 'cat', var_list)

    df_local = df.copy()

    if verbose > 0:
        print('  > method: ' + method)
        print('  > filled features:',df[var_list].isna().sum().loc[df[var_list].isna().sum()>0].index.tolist())
    # fill Na values for each feature in var_list
    for j in var_list:
        df_local = fill_cat(df_local, j, method)

    return df_local
