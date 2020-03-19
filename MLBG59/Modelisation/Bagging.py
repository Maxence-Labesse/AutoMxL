""" Bagging algorithm class. Methods :

- Bagging (class) : generate new training more balanced and train model for each
- Bagging_sample (func) : generate bagging sample

"""
from sklearn.ensemble import RandomForestClassifier
from MLBG59.Modelisation.Utils import *
import pandas as pd

"""
Default bagging parameters
"""
default_bagging_param = {'n_sample': 5,
                         'pos_sample_size': 1.0,
                         'replace': False}


class Bagging(object):
    """Meta-algo designed to improve the stability and accuracy of ML classif/regression algos
    or to face an "imbalanced target distribution" issue.
    
    Bagging generates m new training sets more balanced. Then, a model is fitted on each
    sample and outputs are combined by averaging (for regression) or voting (for classification).

    Available classifiers : Random Forest and XGBOOST
    
    Parameters
    ----------
    clf : Model fitted on samples (Default  : RandomForestClassifier(n_estimators=100, max_leaf_nodes=100)
        Model fitted on the samples
    n_sample : int (Default : 5)
        number a samples
    pos_sample_size : int/float (Default : 1.0)
        Number/rate of target=1 observations in each sample (filled with 3 times more target=0 )

        - if int : number of target=1
        - if float : rate of total target=1

    replace : Boolean (Default : False)
        Enable sampling with replacement

    list_model : list (Default : None)
        Fitted models (created with fit method)
    """

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

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def get_params(self):
        """Get bagging object parameters

        Returns
        -------
        dict
            {param : value}
        """
        return {'classifier': self.classifier,
                'niter': self.niter,
                'pos_sample_size': self.pos_sample_size,
                'replace': self.replace,
                'list_model': self.list_model}

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def fit(self, df_train, target):
        """Create bagging samples from a DataFrame and fit the model (self.clf) on each sample
        
        Parameters
        ----------
        df_train : DataFrame
            Training dataset
        target : String
            Target name

        Returns
        -------
         self.list_model : list
            Fitted models
        """
        # list_model init
        self.list_model = [None] * self.niter

        # get number of target=1 in bagging samples
        if isinstance(self.pos_sample_size, int):
            N = self.pos_sample_size
        else:
            N = int(self.pos_sample_size * df_train.loc[df_train[target] == 1].shape[0])

        for i in range(self.niter):
            # Sample creation
            df_train_bag = Bagging_sample(df_train, target, N, replace=self.replace)

            # X_train / y_train
            X_train_bag = df_train_bag.copy()
            y_train_bag = X_train_bag[target]
            del X_train_bag[target]

            # Create and store model
            self.list_model[i] = self.classifier

            # fit model for each sample
            self.list_model[i].fit(X_train_bag, y_train_bag)

        return self

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def predict(self, df):
        """Apply models fitted on sample to a  dataset.
        Combine models by averaging the outputs (for regression) or voting (for classification)

        Parameters
        ----------
        df : DataFrame
            Dataset to apply the model

        Returns
        -------
        numpy.ndarray (float)
            Averaged classification probabilities
        numpy.ndarray (int)
            Predictions for each observation
        """
        # Init probs storage matrix
        mat_prob = np.zeros((self.niter, df.shape[0]))

        # for each fitted models
        for j in range(self.niter):
            # apply the model on test set
            y_prob_rf = self.list_model[j].predict_proba(df)
            # probabilities storage in matrix
            mat_prob[j] = y_prob_rf[:, 1]

        # probas averaging
        list_prob_pred = mat_prob.sum(axis=0) / self.niter
        # voting
        list_pred = [round(elem, 0) for elem in list_prob_pred]

        return list_prob_pred, list_pred

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def feature_importance(self, X):
        """Get features importance of the model by averaging importance of models fitted on the samples
        
        Parameters
        ----------
        X : DataFrame
            Input Dataset
            
        Returns
        -------
        dict
            {feature : importance}
            
        """
        # Init importance storage matrix
        mat_feat_imp = np.zeros((self.niter, len(X.columns)))

        #  for each fitted models
        for i in range(self.niter):
            # importances storage in matrix
            mat_feat_imp[i] = self.list_model[i].feature_importances_

        # Averaging importances
        list_feat_imp_moy = mat_feat_imp.sum(axis=0) / self.niter

        features_dict = dict(zip(X.columns, list_feat_imp_moy))

        return features_dict


"""
-------------------------------------------------------------------------------------------------------------
"""


def Bagging_sample(df, target, pos_target_nb, replace=False):
    """Generate a DataFrame sample with selected number of target=1
        
    Parameters
    ----------
    df : DataFrame
        Input dataset
    target : String
        Target name
    pos_target_nb : int
        Number of target=1 observations in the sample
    replace : Boolean (défaut : False)
        If True, create samples with replacement
            
    Returns
    -------
    DataFrame
        sample dataset
    """
    # split target = 1 / 0
    df_pos = df.loc[(df[target] == 1)]
    df_neg = df.loc[(df[target] == 0)]

    # sample creation
    df_bag = pd.concat(
        (df_pos.sample(n=pos_target_nb, replace=replace), df_neg.sample(n=3 * pos_target_nb, replace=replace)), axis=0)

    return df_bag
