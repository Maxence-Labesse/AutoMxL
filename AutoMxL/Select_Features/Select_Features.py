from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class FeatSelector(object):

    def __init__(self,
                 method='pca'
                 ):
        assert method in ['pca', 'no_rescale_pca'], 'invalid method : select pca / no_rescale_pca'

        self.method = method
        self.is_fitted = False
        self.l_select_var = []
        self.selector = None
        self.scaler = None

    def fit(self, df, l_var=None, verbose=False):
        l_num = [col for col in df.columns.tolist() if df[col].dtype != 'object']

        if l_var is None:
            self.l_select_var = l_num
        else:
            self.l_select_var = [col for col in l_var if col in l_num]

        if len(self.l_select_var) > 1:
            if self.method in ['pca', 'no_rescale_pca']:

                if self.method == 'pca':
                    scaler = StandardScaler()
                    df_local = scaler.fit_transform(df[self.l_select_var])
                    self.scaler = scaler
                else:
                    df_local = df[self.l_select_var].copy()

            pca = PCA()

            pca.fit(df_local)
            self.selector = pca

            self.is_fitted = True

            if verbose:
                print(" **method : " + self.method)
                print("  >", len(self.l_select_var), "features to encode")

        else:
            print('not enough features !')

    def transform(self, df, verbose=False):
        assert self.is_fitted, 'fit the encoding first using .fit method'

        l_var_other = [col for col in df.columns.tolist() if col not in self.l_select_var]
        df_local = df[self.l_select_var].copy()

        if self.method in ['pca', 'no_rescale_pca']:
            if self.scaler is not None:
                df_local = self.scaler.transform(df_local)

            pca = self.selector
            df_local = pd.DataFrame(pca.transform(df_local))
            df_local = df_local.rename(
                columns=dict(zip(df_local.columns.tolist(), ['Dim' + str(v) for v in df_local.columns.tolist()])))

        n_dim = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]

        if len(l_var_other) > 0:
            df_reduced = pd.concat((df[l_var_other].reset_index(drop=True), df_local.iloc[:, :n_dim + 1]), axis=1)
        else:
            df_reduced = df_local.iloc[:, :n_dim + 1]

        if verbose:
            print("Numerical Dimensions reduction : " + str(len(self.l_select_var)) + " - > " + str(n_dim + 1))
            print("explained inertia : " + str(round(np.cumsum(pca.explained_variance_ratio_)[n_dim], 4)))
        return df_reduced

    def fit_transform(self, df, l_var, verbose=False):
        df_local = df.copy()
        self.fit(df_local, l_var=l_var, verbose=verbose)
        df_reduced = self.transform(df_local, verbose=verbose)

        return df_reduced

def select_features(df, target, method='pca', verbose=False):
    assert method in ['pca', 'no_rescale_pca'], method + " invalid method : select pca, no_rescale_pca"

    l_num = [col for col in df._get_numeric_data().columns.tolist() if col != target]
    l_other = [col for col in df.columns.tolist() if col not in l_num]

    df_num = df[l_num].copy()

    if method in ['pca', 'no_rescale_pca']:

        if method == 'pca':
            scaler = StandardScaler()
            X = scaler.fit_transform(df_num)
        else:
            X = df_num.copy()

        pca = PCA()

        X_transform = pd.DataFrame(pca.fit_transform(X))
        X_transform = X_transform.rename(
            columns=dict(zip(X_transform.columns.tolist(), ['Dim' + str(v) for v in X_transform.columns.tolist()])))

        n_dim = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]

    if len(l_other) > 0:

        df_pca = pd.concat((df[l_other].reset_index(drop=True), X_transform.iloc[:, :n_dim + 1]), axis=1)
    else:
        df_pca = X_transform.iloc[:, :n_dim + 1]

    if verbose:
        print("Numerical Dimensions reduction : " + str(len(l_num)) + " - > " + str(n_dim + 1))
        print("explained inertia : " + str(round(np.cumsum(pca.explained_variance_ratio_)[n_dim], 4)))

    return df_pca
