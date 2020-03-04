""" Global dataset information functions :

 - recap : get global information about the dataset (NA, features type, low variance features, ...)
 - low variance features : identify features with low variance
"""
from sklearn.preprocessing import MinMaxScaler
from MLBG59.Explore.Features_Type import *
from MLBG59.Utils.Display import *


def explore(df, verbose=False):
    """Get global information about the dataset

    - Variables type :
        - date
        - identifier
        - verbatim
        - boolean
        - categorical
        - numerical
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
        {x : variables names list }

        - date : date features
        - identifier : identifier features
        - verbatim : verbatim features
        - boolean : boolean features
        - categorical : categorical features
        - numerical : numerical features
        - categorical : categorical features
        - date : date features
        - NA : features which contains NA values
        - low_variance : list of the features with low variance
    """
    # dataset dimensions
    if verbose:
        color_print("Dimensions :")
        print("  > row number :", df.shape[0], "\n  > col number : ", df.shape[1])

    #################
    # features type #
    #################
    d_features = get_features_type(df, var_list=None, th=0.95)

    if verbose:
        color_print("Features type identification : ")
        list(map(lambda typ :
                 print("  > " + typ + " : " + str(len(d_features[typ])) + ' (' + str(
                     round(len(d_features[typ]) / df.shape[1] * 100)) + '%)'),
                 d_features.keys()))

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
        low_variance_features(df, var_list=df._get_numeric_data().columns.tolist(), threshold=0, rescale=True,
                              verbose=verbose).index.tolist()

    # store into DataFrame
    d_features['NA'] = NA_columns
    d_features['low_variance'] = low_var_columns

    return d_features


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def low_variance_features(df, var_list=None, threshold=0, rescale=True, verbose=False):
    """Identify  features with low variance : (< threshold).
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
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    list
       Names of the variables with low variance
    """
    # if var_list = None, get all num features
    # else, remove features from var_list whose type is not num
    l_num = df._get_numeric_data().columns.tolist()

    if var_list is None:
        var_list = l_num
    else:
        var_list = [col for col in var_list if col in l_num]

    df_bis = df.copy()

    if rescale:
        scler = MinMaxScaler()
        df_bis[var_list] = scler.fit_transform(df_bis[var_list].astype('float64'))

    selected_var = df_bis[var_list].var().loc[df_bis.var() <= threshold]

    if verbose:
        # print('features : ',list(var_list))
        if rescale:
            print('  **MinMaxScaler [0,1]')
        print('  ', str(len(selected_var)) + ' feature(s) with  variance <= threshold (' + str(threshold) + ')')

    return selected_var.sort_values(ascending=True)
