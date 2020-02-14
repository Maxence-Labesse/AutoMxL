""" Bagging algorithm class

Bagging class methods :

- get_params
- train
- predict
- feat_importance_BAG_RF

Functions :

- Bagging sample

"""
from sklearn.ensemble import RandomForestClassifier
from MLBG59.Utils.Utils import *

"""
Default bagging parameters
"""
default_bagging_param = {'niter': 5,
                         'pos_sample_size': 1.0,
                         'replace': False}


class Bagging(object):
    """Meta-algo designed to improve the stability and accuracy of ML classif/regression algos
    or to face an "imbalanced target distribution" problem
    
    Bagging generates m new training sets more balanced. Then, m models are fitted using the m
    samples and combined by averaging the output (for regression) or voting (for classification).
    
    Parameters
    ----------
    classifier : Model fitted on samples (Default  : RandomForestClassifier(n_estimators=100, max_leaf_nodes=100)
        Model fitted on the samples
    niter : int (Default : 5)
        # samples
    pos_sample_size : int/float (Default : 1.0)
        Number of target=1 observations in each sample (filled with 3*target=0 obs )

        - if int : pos_sample_size
        - if float : pos_sample_size*df.loc[df[target] == 1]
    replace : Boolean (Default : False)
        If True, sampling with replacement

    list_model : list (Default : None)
        Fitted models list
    """
    def __init__(self,
                 classifier=RandomForestClassifier(n_estimators=100, max_leaf_nodes=100),
                 niter=5,
                 pos_sample_size=1.0,
                 replace=True):

        self.classifier = classifier
        self.niter = niter
        self.pos_sample_size = pos_sample_size
        self.replace = replace
        self.list_model = list()

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def get_params(self):
        """Get object parameters

        Parameters
        ----------
        self
        """
        return {'classifier': self.classifier,
                'niter': self.niter,
                'pos_sample_size': self.pos_sample_size,
                'replace': self.replace,
                'list_model': self.list_model}

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def train(self, df_train, target):
        """Create bagging samples from a DataFrame.
        Fit the model on each sample
        
        Parameters
        ----------
        self
        df_train : DataFrame
            Training dataset
        target : String
            Target

        Returns
        -------
         self.list_model : list
            Fitted models lisdt
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

    def predict(self, X):
        """Apply bagging models on a test set.
        combine by averaging the output (for regression) or voting (for classification)

        Parameters
        ----------
        self
        X : DataFrame
            Testing dataset

        Returns
        -------
        numpy.ndarray
            Averaged classification probabilities
        numpy.ndarray
            Predictions for each obs
        """
        # Init probs storage matrix
        mat_prob = np.zeros((self.niter, X.shape[0]))

        # for each fitted models
        for j in range(self.niter):
            # apply the model on test set
            y_prob_rf = self.list_model[j].predict_proba(X)
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

    def feat_importance_BAG_RF(self, X):
        """Get features importance of the model
        
        Parameters
        ----------
        self
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


def Bagging_sample(df, target, N, replace=False):
    """Sample creation with N target=1 and 3*N target=0 observations
        
    Parameters
    ----------
    df : DataFrame
        Input dataset
    target : String
        target
    N : int
        Number of target=1 observations
    replace : Boolean (défaut : False)
        If True, create samples with replacement
            
    Returns
    -------
    df_bag : DataFrame
        sample dataset
    """
    # split target = 1 / 0
    df_pos = df.loc[(df[target] == 1)]
    df_neg = df.loc[(df[target] == 0)]

    # sample creation
    df_bag = pd.concat((df_pos.sample(n=N, replace=replace), df_neg.sample(n=3 * N, replace=replace)), axis=0)

    return df_bag
