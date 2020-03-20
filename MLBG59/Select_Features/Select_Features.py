""" Features selection

- select_features (func) : features selection following method

"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def select_features(df, target, method='pca', verbose=False):
    """features selection following  method

    - pca : use pca to reduce dataset dimensions
    - no_rescale_pca : use pca without rescaling data

    Parameters
    ----------
    df : DataFrame
        input dataset containing features
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
        print(df_pca.shape)

    return df_pca
