from sklearn.ensemble import RandomForestClassifier
from AutoMxL.Modelisation.Utils import *
import pandas as pd

"""
Default bagging parameters
"""
default_bagging_param = {'n_sample': 5,
                         'pos_sample_size': 1.0,
                         'replace': False}

class Bagging(object):

    def __init__(self,
                 clf=RandomForestClassifier(n_estimators=100, max_leaf_nodes=100),
                 n_sample=5,
                 pos_sample_size=1.0,
                 replace=True):

        self.classifier = clf
        self.niter = n_sample
        self.pos_sample_size = pos_sample_size
        self.replace = replace
        self.list_model = list()
        self.is_fitted = False

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def get_params(self):
        return {'classifier': self.classifier,
                'niter': self.niter,
                'pos_sample_size': self.pos_sample_size,
                'replace': self.replace,
                'list_model': self.list_model}

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def fit(self, df_train, target):
        self.list_model = [None] * self.niter

        if isinstance(self.pos_sample_size, int):
            N = self.pos_sample_size
        else:
            N = int(self.pos_sample_size * df_train.loc[df_train[target] == 1].shape[0])

        for i in range(self.niter):
            df_train_bag = create_sample(df_train, target, N, replace=self.replace)

            X_train_bag = df_train_bag.copy()
            y_train_bag = X_train_bag[target]
            del X_train_bag[target]

            self.list_model[i] = self.classifier

            self.list_model[i].fit(X_train_bag, y_train_bag)

            self.is_fitted = True

        return self

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def predict(self, df):
        assert self.is_fitted, "Fit first !"
        mat_prob = np.zeros((self.niter, df.shape[0]))

        for j in range(self.niter):
            y_prob_rf = self.list_model[j].predict_proba(df)
            mat_prob[j] = y_prob_rf[:, 1]

        list_prob_pred = mat_prob.sum(axis=0) / self.niter
        list_pred = [round(elem, 0) for elem in list_prob_pred]

        return list_prob_pred, list_pred

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def bag_feature_importance(self, X):
        mat_feat_imp = np.zeros((self.niter, len(X.columns)))

        for i in range(self.niter):
            mat_feat_imp[i] = self.list_model[i].feature_importances_

        list_feat_imp_moy = mat_feat_imp.sum(axis=0) / self.niter

        features_dict = dict(zip(X.columns, list_feat_imp_moy))

        return features_dict

"""
-------------------------------------------------------------------------------------------------------------
"""

def create_sample(df, target, pos_target_nb, replace=False):
    df_pos = df.loc[(df[target] == 1)]
    df_neg = df.loc[(df[target] == 0)]

    n_size = min(3 * pos_target_nb, df_neg.shape[0])

    df_bag = pd.concat(
        (df_pos.sample(n=pos_target_nb, replace=replace), df_neg.sample(n=n_size, replace=replace)), axis=0)

    return df_bag
