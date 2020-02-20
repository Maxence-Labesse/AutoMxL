""" Global dataset information functions :

 - recap : get global information about the dataset (NA, features type, low variance features, ...)
 - is_date : test if a variable is as date
 - get_all_dates : identify date features
 - low variance features : identify features with low variance
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from MLBG59.Utils.Display import *
from MLBG59.Utils.Utils import get_type_features


def recap(df, verbose=False):
    """Get global information about the dataset

    - Variables type (num, cat, date)
    - NA values
    - low variance variables
        
    Parameters
    ----------
    df : DataFrame
        input dataset
    verbose : boolean (Default False)
        Get logging information
            
    Returns
    -------
    dict
        {x : list of variables names}

        - x = numerical : numerical features
        - x = categorical : categorical features
        - x = date : date features
        - x = NA : features which contains NA values
        - x = low_variance : list of the features with low variance
    """
    # dataset dimensions
    if verbose:
        color_print("Dimensions : ")
        print("  > row number : ", df.shape[0], "\n  > col number : ", df.shape[1])

    #################
    # features type #
    #################
    # numerical
    num_columns = df._get_numeric_data().columns.tolist()
    # date
    date_columns = get_all_dates(df)
    # categorical
    cat_columns = [x for x in df.columns if (x not in num_columns) and (x not in date_columns)]

    if verbose:
        color_print("Features type identification : ")
        print("  > cat : " + str(len(cat_columns)) + ' (' + str(round(len(cat_columns) / df.shape[1] * 100)) + '%)',
              '\n  > num : ' + str(len(num_columns)) + ' (' + str(round(len(num_columns) / df.shape[1] * 100)) + '%)',
              '\n  > dates: ' + str(len(date_columns)) + ' (' + str(
                  round(len(date_columns) / df.shape[1] * 100)) + ' %)')

    ######################
    # NA values analysis
    ######################
    df_col = pd.DataFrame(df.columns.values, columns=['variables'])
    df_col['Nbr NA'] = df.isna().sum().tolist()
    df_col['Taux NA'] = df_col['Nbr NA'] / df.shape[0]
    # features containing NA values
    NA_columns = df_col.loc[df_col['Nbr NA'] > 0].sort_values('Nbr NA', ascending=False).variables.tolist()
    col_des = df_col['Taux NA'].describe()

    if verbose:
        color_print(str(len(NA_columns)) + " features containing NA")
        print('  > Taux NA moyen : ' + str(round(col_des['mean'] * 100, 2)) + '%',
              '\n  >           min : ' + str(round(col_des['min'] * 100, 2)) + '%',
              '\n  >           max : ' + str(round(col_des['max'] * 100, 2)) + '%')

    #########################
    # Low variance features
    #########################
    if verbose:
        color_print('Low variance features')
    low_var_columns = \
        low_variance_features(df, var_list=num_columns, threshold=0, rescale=True, verbose=verbose).index.tolist()

    # store into DataFrame
    d_features = {'numerical': num_columns,
                  'date': date_columns,
                  'categorical': cat_columns,
                  'NA': NA_columns,
                  'low_variance': low_var_columns}

    return d_features


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def is_date(df, col):
    """Test if a variable is as date.

    Method : try to apply to_datetime

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name

    Returns
    -------
    res : boolean
        test result
        test result
    """
    # if col is datetime type, res = True
    if df[col].dtype == 'datetime64[ns]':
        return True

    # if col is object type, try apply to_datetime
    elif df[col].dtype == 'object':
        try:
            df_smpl = df.sample(100).copy()
            pd.to_datetime(df_smpl[col])
            return True
        except ValueError:
            return False
        except OverflowError:
            return False


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def get_all_dates(df):
    """Identify dates variables

    Method : try to apply to_datetime
    
    Parameters
    ----------
    df : DataFrame
        input DataFrame

    Returns
    -------
    list
        features identified as date
    """
    date_list = list()

    for col in df.columns:
        # if col is recognized as date
        if is_date(df, col): date_list.append(col)

    return date_list


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def low_variance_features(df, var_list=None, threshold=0, rescale=True, verbose=1):
    """Identify  features with low variance (< threshold).
    Possible to rescale feature before computing.

    Parameters
    ----------
     df : DataFrame
        input DataFrame
     var_list : list (default : None)
        names of the variables to test variance
     threshold : float (default : 0)
        variance threshold
     rescale : bool (default : true)
        enable  MinMaxScaler before computing variance

    Returns
    -------
    list
       Names of the variables with low variance
    """
    # if var_list = None, get all numerical features
    # else, exclude features from var_list whose type is not numerical
    var_list = get_type_features(df, 'num', var_list)

    df_bis = df.copy()

    if rescale:
        scler = MinMaxScaler()
        df_bis[var_list] = scler.fit_transform(df_bis[var_list].astype('float64'))

    selected_var = df_bis[var_list].var().loc[df_bis.var() <= threshold]

    if verbose > 0:
        # print('features : ',list(var_list))
        if rescale: print('  **MinMaxScaler [0,1]')
        print('  ', str(len(selected_var)) + ' feature(s) with  variance <= threshold (' + str(threshold) + ')')

    return selected_var.sort_values(ascending=True)
