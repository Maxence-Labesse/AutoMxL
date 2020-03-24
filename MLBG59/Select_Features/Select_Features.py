""" Features selection

- select_features (func) : features selection following method

"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class FeatSelector(object):
    """features selection following  method

    - pca : use pca to reduce dataset dimensions
    - no_rescale_pca : use pca without rescaling data

    Parameters
    ----------
    method : string (Default pca)
        method use to select features
    """

    def __init__(self,
                 method='pca'
                 ):
        assert method in ['pca', 'no_rescale_pca'], 'invalid method : select pca / no_rescale_pca'

        self.method = method
        self.is_fitted = False
        self.l_select_var = []
        self.l_var_other = []
        self.selector = None
        self.scaler = None

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit(self, df, l_var=None, verbose=False):
        """fit selector

        Parameters
        ----------
        df : DataFrame
            input dataset
        l_var : list
            features to encode.
            If None, all features identified as numerical
        verbose : boolean (Default False)
            Get logging information
        """
        # get categorical and boolean features (see Features_Type module doc)
        l_num = [col for col in df.columns.tolist() if df[col].dtype != 'object']

        # list of features to encode
        if l_var is None:
            self.l_select_var = l_num
        else:
            self.l_select_var = [col for col in l_var if col in l_num]

        if len(self.l_select_var) > 1:
            # PCA method
            if self.method in ['pca', 'no_rescale_pca']:

                if self.method == 'pca':
                    scaler = StandardScaler()
                    df_local = scaler.fit_transform(df[self.l_select_var])
                    self.scaler = scaler
                else:
                    df_local = df['l_select_var'].copy()

            # init pca object
            pca = PCA()

            # fit and transform with pca
            pca.fit(df_local)
            self.selector = pca

            # Fitted !
            self.is_fitted = True

            # verbose
            if verbose:
                print(" **method : " + self.method)
                print("  >", len(self.l_select_var), "features to encode")

        else:
            print('not enough features !')

    """
    ----------------------------------------------------------------------------------------------
    """

    def transform(self, df, verbose=False):
        """ apply features selection on a dataset

        Parameters
        ----------
        df : DataFrame
            dataset to transform
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
        DataFrame : modified dataset
        """
        assert self.is_fitted, 'fit the encoding first using .fit method'
        print(df.columns.tolist())

        l_var_other = [col for col in df.columns.tolist() if col not in self.l_select_var]
        df_local = df[self.l_select_var].copy()

        # pca methods
        if self.method in ['pca', 'no_rescale_pca']:
            if self.scaler is not None:
                df_local = self.scaler.transform(df_local)

            pca = self.selector
            df_local = pd.DataFrame(pca.transform(df_local))
            df_local = df_local.rename(
                columns=dict(zip(df_local.columns.tolist(), ['Dim' + str(v) for v in df_local.columns.tolist()])))

        # find argmin to get 90% of variance
        n_dim = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.90)[0][0]

        # concat with other dataset features
        if len(l_var_other) > 0:
            df_reduced = pd.concat((df[l_var_other].reset_index(drop=True), df_local.iloc[:, :n_dim + 1]), axis=1)
        else:
            df_reduced = df_local.iloc[:, :n_dim + 1]

        # verbose
        if verbose:
            print("Numerical Dimensions reduction : " + str(len(self.l_select_var)) + " - > " + str(n_dim + 1))
            print("explained inertia : " + str(round(np.cumsum(pca.explained_variance_ratio_)[n_dim], 4)))

        return df_reduced

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit_transform(self, df, l_var, verbose):
        """ fit and apply features selection

        Parameters
        ----------
        df : DataFrame
            input dataset
        l_var : list
            features to encode.
            If None, all features identified as dates (see Features_Type module)
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
        DataFrame : modified dataset
        """
        df_local = df.copy()
        self.fit(df_local, l_var=l_var, verbose=verbose)
        df_reduced = self.transform(df_local, verbose=verbose)

        return df_reduced


"""
----------------------------------------------------------------------------------------------
"""


def select_features(df, target, method='pca', verbose=False):
    """features selection following  method

    - pca : use pca to reduce dataset dimensions
    - no_rescale_pca : use pca without rescaling data

    Parameters
    ----------
    df : DataFrame
        input dataset containing features
    target : string
        target name
    method : string (Default pca)
        method use to select features
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame
        modified dataset
    """
    # assert valid method
    assert method in ['pca', 'no_rescale_pca'], method + " invalid method : select pca, no_rescale_pca"

    # get numerical features (except target) and others
    l_num = [col for col in df._get_numeric_data().columns.tolist() if col != target]
    l_other = [col for col in df.columns.tolist() if col not in l_num]

    # prepare dataset to apply PCA
    df_num = df[l_num].copy()

    # PCA method
    if method in ['pca', 'no_rescale_pca']:

        if method == 'pca':
            scaler = StandardScaler()
            X = scaler.fit_transform(df_num)
        else:
            X = df_num.copy()

        # init pca object
        pca = PCA()

        # fit and transform with pca
        X_transform = pd.DataFrame(pca.fit_transform(X))
        X_transform = X_transform.rename(
            columns=dict(zip(X_transform.columns.tolist(), ['Dim' + str(v) for v in X_transform.columns.tolist()])))

        # find argmin to get 90% of variance
        n_dim = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.90)[0][0]

    # concat with other dataset features
    if len(l_other) > 0:

        df_pca = pd.concat((df[l_other].reset_index(drop=True), X_transform.iloc[:, :n_dim + 1]), axis=1)
    else:
        df_pca = X_transform.iloc[:, :n_dim + 1]

    # verbose
    if verbose:
        print("Numerical Dimensions reduction : " + str(len(l_num)) + " - > " + str(n_dim + 1))
        print("explained inertia : " + str(round(np.cumsum(pca.explained_variance_ratio_)[n_dim], 4)))

    return df_pca
