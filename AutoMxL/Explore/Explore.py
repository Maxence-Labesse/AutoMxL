from sklearn.preprocessing import MinMaxScaler
from AutoMxL.Explore.Features_Type import *
from AutoMxL.Utils.Display import *

def explore(df, verbose=False):
    """
    Analyse un DataFrame et détecte automatiquement les types de features.

    Première étape du pipeline AutoMxL. Identifie les colonnes par type
    (date, identifier, verbatim, boolean, categorical, numerical) et détecte
    les features à faible variance et les valeurs manquantes.

    Args:
        df: DataFrame à analyser
        verbose: Affiche les détails de l'analyse

    Returns:
        Dictionnaire avec les listes de colonnes par type
    """
    if verbose:
        color_print("Dimensions :")
        print("  > row number :", df.shape[0], "\n  > col number :", df.shape[1])

    if verbose:
        color_print('Low variance features')

    l_low_var = \
        low_variance_features(df, var_list=df._get_numeric_data().columns.tolist(), threshold=0, rescale=True,
                              verbose=verbose).index.tolist()

    l_unique = [col for col in df.columns.tolist() if df[col].dtype == 'object' and df[col].nunique(dropna=True) == 1]

    l_low_var = l_low_var + l_unique

    df_valid = df.drop(l_low_var, axis=1).copy()

    d_features = get_features_type(df_valid, l_var=None, th=0.95)

    if verbose:
        color_print("Features type identification : ")
        list(map(lambda typ:
                 print("  > " + typ + " : " + str(len(d_features[typ])) + ' (' + str(
                     round(len(d_features[typ]) / df_valid.shape[1] * 100)) + '%)'),
                 d_features.keys()))

    df_col = pd.DataFrame(df_valid.columns.values, columns=['variables'])
    df_col['Nbr NA'] = df_valid.isna().sum().tolist()
    df_col['Taux NA'] = df_col['Nbr NA'] / df_valid.shape[0]
    NA_columns = df_col.loc[df_col['Nbr NA'] > 0].sort_values('Nbr NA', ascending=False).variables.tolist()
    col_des = df_col['Taux NA'].describe()

    if verbose:
        color_print(str(len(NA_columns)) + " features containing NA")
        print('  > Taux NA moyen : ' + str(round(col_des['mean'] * 100, 2)) + '%',
              '\n  >           min : ' + str(round(col_des['min'] * 100, 2)) + '%',
              '\n  >           max : ' + str(round(col_des['max'] * 100, 2)) + '%')

    d_features['NA'] = NA_columns
    d_features['low_variance'] = l_low_var

    return d_features

"""
-------------------------------------------------------------------------------------------------------------------------
"""

def get_features_type(df, l_var=None, th=0.95):
    """
    Classifie les colonnes d'un DataFrame par type.

    Applique les détecteurs dans l'ordre : date → identifier → verbatim →
    boolean → categorical. Les colonnes restantes sont considérées numerical.

    Args:
        df: DataFrame à analyser
        l_var: Liste des colonnes à analyser (toutes si None)
        th: Seuil d'unicité pour identifier/verbatim (défaut: 0.95)

    Returns:
        Dictionnaire {type: [liste de colonnes]}
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
    """
    Détecte les features numériques à faible variance.

    Ces features n'apportent pas d'information discriminante et seront
    supprimées lors du preprocessing.

    Args:
        df: DataFrame à analyser
        var_list: Liste des colonnes à analyser (toutes les numériques si None)
        threshold: Seuil de variance (défaut: 0 = variance nulle)
        rescale: Normalise en [0,1] avant calcul (défaut: True)
        verbose: Affiche les détails

    Returns:
        Series des variances pour les features sous le seuil
    """
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
        if rescale:
            print('  **MinMaxScaler [0,1]')
        print('  ', str(len(selected_var)) + ' feature(s) with  variance <= threshold (' + str(threshold) + ')')

    return selected_var.sort_values(ascending=True)
