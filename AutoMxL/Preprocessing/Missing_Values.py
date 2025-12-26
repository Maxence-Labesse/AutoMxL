"""
Gestion des valeurs manquantes (NA).

Remplit les NA avec des stratégies différentes selon le type :
- Numériques : median, mean ou zero
- Catégorielles : 'NR' (Non Renseigné)
"""
import pandas as pd
import numpy as np


class NAEncoder(object):
    """
    Remplit les valeurs manquantes d'un DataFrame.

    Options :
    - replace_num_with : 'median', 'mean', 'zero'
    - replace_cat_with : 'NR'
    - track_num_NA : crée des colonnes top_NA_* pour tracer les imputations
    """

    def __init__(self,
                 replace_num_with='median',
                 replace_cat_with='NR',
                 track_num_NA=True
                 ):

        assert replace_num_with in ['median', 'mean', 'zero'], 'invalid method, select median/mean/zero'
        assert replace_cat_with in ['NR'], 'invalid method, select NR'
        self.replace_num_with = replace_num_with
        self.replace_cat_with = replace_cat_with
        self.track_num_NA = track_num_NA
        self.l_var_cat = []
        self.l_var_num = []
        self.is_fitted = False

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit(self, df, l_var, verbose=False):
        """
        Identifie les colonnes avec des NA à remplir.

        Sépare les colonnes numériques et catégorielles.
        """
        l_num = [col for col in df.columns.tolist() if df[col].dtype != 'object']
        l_str = [col for col in df.columns.tolist() if df[col].dtype == 'object']

        if l_var is None:
            self.l_var_cat = [col for col in l_str if df[col].isna().sum() > 0]
            self.l_var_num = [col for col in l_num if df[col].isna().sum() > 0]
        else:
            self.l_var_cat = [col for col in l_var if col in l_str and df[col].isna().sum() > 0]
            self.l_var_num = [col for col in l_var if col in l_num and df[col].isna().sum() > 0]

        self.is_fitted = True

        if verbose:
            print(" **method cat:", self.replace_cat_with, " / num:", self.replace_num_with)
            print("  >", len(self.l_var_cat) + len(self.l_var_num), "features to fill")
            if len(self.l_var_cat) > 0:
                print("  - cat", self.l_var_cat)
            if len(self.l_var_num) > 0:
                print("  - num", self.l_var_num)

    """
    ----------------------------------------------------------------------------------------------
    """

    def transform(self, df, verbose=False):
        """
        Remplit les NA selon les stratégies configurées.

        Returns:
            DataFrame avec les NA remplis (+ colonnes top_NA_* si track_num_NA=True)
        """
        assert self.is_fitted, 'fit the encoding first using .fit method'

        df_local = df.copy()

        if len(self.l_var_cat) > 0:
            df_local = fill_categorical(df_local, l_var=self.l_var_cat, method=self.replace_cat_with,
                                        verbose=verbose)

        if len(self.l_var_num) > 0:
            df_local = fill_numerical(df_local, l_var=self.l_var_num, method=self.replace_num_with,
                                      track_num_NA=self.track_num_NA, verbose=verbose)

        if len(self.l_var_cat) + len(self.l_var_num) == 0 and verbose:
            print("  > no transformation to apply")

        return df_local

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit_transform(self, df, l_var=None, verbose=False):
        """Fit puis transform en une seule opération."""
        df_local = df.copy()
        self.fit(df_local, l_var=l_var, verbose=verbose)
        df_local = self.transform(df_local, verbose=verbose)

        return df_local

    """
    ----------------------------------------------------------------------------------------------
    """

def fill_numerical(df, l_var=None, method='median', track_num_NA=True, verbose=False):
    """
    Remplit les NA des colonnes numériques.

    Si track_num_NA=True, crée une colonne top_NA_* par variable (1 si NA, 0 sinon).
    """
    assert method in ['zero', 'median', 'mean'], method + ' invalid method : choose zero, median or mean'

    l_num = df._get_numeric_data().columns.tolist()

    if l_var is None:
        l_var = l_num
    else:
        l_var = [col for col in l_var if col in l_num]

    df_local = df.copy()

    if method == 'median':
        fill_value = df_local[l_var].mean()
    elif method == 'mean':
        fill_value = df_local[l_var].mean()
    elif method == 'zero':
        fill_value = pd.Series([0] * len(l_var), index=l_var)

    for var in l_var:
        if track_num_NA:
            df_local['top_NA_' + var] = df_local.apply(lambda x: 1 if np.isnan(x[var]) else 0, axis=1)
        df_local[var] = df_local[var].fillna(fill_value[var])

    if verbose:
        print('  > method: ' + method)
        print('  > filled features:', df[l_var].isna().sum().loc[df[l_var].isna().sum() > 0].index.tolist())

    return df_local

"""
-------------------------------------------------------------------------------------------------------------------------
"""

def fill_categorical(df, l_var=None, method='NR', verbose=False):
    """Remplit les NA des colonnes catégorielles avec 'NR' (Non Renseigné)."""
    assert method in ['NR'], method + ' invalid method : choose NR '

    l_cat = [col for col in df.columns.tolist() if df[col].dtype == 'object']

    if l_var is None:
        l_var = l_cat
    else:
        l_var = [col for col in l_var if col in l_cat]

    df_local = df.copy()

    if method in ['NR']:
        fill_value = 'NR'

    for var in l_var:
        df_local[var] = df_local[var].fillna(fill_value)

    if verbose:
        print('  > method: ' + method)
        print('  > filled features:', df[l_var].isna().sum().loc[df[l_var].isna().sum() > 0].index.tolist())

    return df_local

def get_NA_features(df):
    """Retourne la liste des colonnes contenant des NA."""
    return df.isna().sum()[df.isna().sum() > 0].index.tolist()
