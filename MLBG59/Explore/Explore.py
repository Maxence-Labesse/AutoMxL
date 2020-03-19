""" Global dataset information functions :

 - explore (func): Identify variables types and gives global information about the dataset (NA, low variance features)
 - low variance features (func): identify features with low variance
 - - get_features_type (func): get all features per type
"""
from sklearn.preprocessing import MinMaxScaler
from MLBG59.Explore.Features_Type import *
from MLBG59.Utils.Display import *


def explore(df, verbose=False):
    """Identify variables types and gives global information about the dataset

    - Variables type :
        - date
        - identifier
        - verbatim
        - boolean
        - categorical
        - numerical
    - variables containing NA values
    - low variance and unique values variables

    See get_features_type function doc for type identification heuristics

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

    #########################
    # Low variance features
    #########################
    if verbose:
        color_print('Low variance features')

    l_low_var = \
        low_variance_features(df, var_list=df._get_numeric_data().columns.tolist(), threshold=0, rescale=True,
                              verbose=verbose).index.tolist()

    # categorical features with unique values
    l_unique = [col for col in df.columns.tolist() if df[col].dtype == 'object' and df[col].nunique(dropna=True) == 1]

    l_low_var = l_low_var + l_unique
    df_valid = df.drop(l_low_var, axis=1).copy()

    #################
    # features type #
    #################
    d_features = get_features_type(df_valid, l_var=None, th=0.95)

    if verbose:
        color_print("Features type identification : ")
        list(map(lambda typ:
                 print("  > " + typ + " : " + str(len(d_features[typ])) + ' (' + str(
                     round(len(d_features[typ]) / df_valid.shape[1] * 100)) + '%)'),
                 d_features.keys()))

    ######################
    # NA values analysis
    ######################
    df_col = pd.DataFrame(df_valid.columns.values, columns=['variables'])
    df_col['Nbr NA'] = df_valid.isna().sum().tolist()
    df_col['Taux NA'] = df_col['Nbr NA'] / df_valid.shape[0]
    # features containing NA values
    NA_columns = df_col.loc[df_col['Nbr NA'] > 0].sort_values('Nbr NA', ascending=False).variables.tolist()
    col_des = df_col['Taux NA'].describe()

    if verbose:
        color_print(str(len(NA_columns)) + " features containing NA")
        print('  > Taux NA moyen : ' + str(round(col_des['mean'] * 100, 2)) + '%',
              '\n  >           min : ' + str(round(col_des['min'] * 100, 2)) + '%',
              '\n  >           max : ' + str(round(col_des['max'] * 100, 2)) + '%')

    # store into DataFrame
    d_features['NA'] = NA_columns
    d_features['low_variance'] = l_low_var

    return d_features


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def get_features_type(df, l_var=None, th=0.95):
    """ Get all features per type :

    - date : try to apply to_datetime
    - identifier :
        - #(unique values)/#(total values) > threshold (default 0.95)
        - AND length is the same for all values (for non NA)
    - verbatim :
        - #(unique values)/#(total values) >= threshold (default 0.95)
        - AND length is NOT the same for all values (for non NA)
    - boolean : #(distinct values) = 2
    - categorical :
        - not a date
        - #(unique values)/#(total values) < threshold (default 0.95)
        - AND #(uniques values)>2
        - AND for num values #(unique values)<30
    - numerical : others

    Parameters
    ----------
    df : DataFrame
        input dataset
    l_var : list (Default  : None)
        variable names
    th : float (Default : 0.95)
        threshold used to identify identifiers/verbatims variables

    Returns
    -------
    dict
        { type : variables name list}
    """
    d_output = {}

    if l_var is None:
        df_local = df.copy()
    else:
        df_local = df[l_var].copy()

    l_col = df_local.columns.tolist()

    for typ in ['date', 'identifier', 'verbatim', 'boolean', 'categorical']:
        d_output[typ] = features_from_type(df_local, typ, l_var=l_col, th=th)
        l_col = [x for x in l_col if (x not in d_output[typ])]

    d_output['numerical'] = l_col

    return d_output


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def low_variance_features(df, var_list=None, threshold=0, rescale=True, verbose=False):
    """Identify numerical features with low variance : (< threshold).
    Possible to rescale feature before computing.

    Parameters
    ----------
     df : DataFrame
        input DataFrame
     var_list : list (default : None)
        names of the variables to check variance
        if None : all the numerical features
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
