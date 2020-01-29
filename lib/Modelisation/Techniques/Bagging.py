from sklearn.ensemble import RandomForestClassifier
from lib.Utils.Utils import *

"""
Default bagging parameters
"""
default_bagging_param = {'niter': 5,
                         'pos_sample_size': 1.0,
                         'replace': False}


class Bagging(object):
    """ 
    Meta-algo designed to improve the stability and accuracy of ML classif/regression algos
    or to face an "imbalanced target distribution" problem
    
    Bagging generates m new training sets more balanced. Then, m models are fitted using the above m
    samples and combined by averaging the output (for regression) or voting (for classification).
    
    Parameters
    ---------
     > classifier : Model fitted on samples (Default  : andomForestClassifier(n_estimators=100, max_leaf_nodes=100)
          Model fitted on the samples
     > niter : int (Default : 5)
          # samples
     > pos_sample_size : int/float (Default : 1.0)
          number of target=1 observations in each sample (target=0 observations :  3x"number of target=1" )
              - if int : pos_sample_size
              - if float : pos_sample_size*len(X)
     > replace : Boolean (Defualt : False)
          If True, sampling with replacement
        
    Attributes
    ---------
     > liste_model : list (Default : None)
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
        """
        get object parameters
    
        """
        return {'classifier': self.classifier,
                'niter': self.niter,
                'pos_sample_size': self.pos_sample_size,
                'replace': self.replace,
                'list_model': self.list_model}

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def set_params(self, params):
        """
        Modify object parameters
        
        input
        -----
         > params : Dictonnary
             key : attribute name
             value : new attribute value
        """
        for k, v in params.items():
            if k not in self.get_params():
                print("\"" + k + "\" : Paramètre invalide")
            else:
                setattr(self, k, v)

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def train(self, df_train, target):
        """
        Create bagging samples from a DataFrame
        fit the model on each sample
        
        input
        -----
         > df_train : DataFrame
              train dataset
         > target : String
              target

        return
        ------
         > list_model : list
              fitted models lisdt
        """
        # list_model init
        self.list_model = [None] * self.niter

        # Number of target=1 observations for each sample
        if isinstance(self.pos_sample_size, int):
            N = self.pos_sample_size
        else:
            N = int(1.0 * df_train.loc[df_train[target] == 1].shape[0])

        # loop over the number of samples
        for i in range(self.niter):
            # Creation of the sample
            df_train_bag = Bagging_sample(df_train, target, N, replace=self.replace)

            # X_train / y_train
            X_train_bag = df_train_bag.copy()
            y_train_bag = X_train_bag[target]
            del X_train_bag[target]

            # Create and store model
            self.list_model[i] = self.classifier

            # fit model o nthe sample
            self.list_model[i].fit(X_train_bag, y_train_bag)

        return self

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def predict(self, X):
        """
        Apply baggin models on a test set
        combine by averaging the output (for regression) or voting (for classification)

        input
        -----
        X : DataFrame
            test set
        
        list_model : list
            fitted models list
            
        return
        ------
        list_proba_pred : numpy.ndarray
            averaged classification probabilities
            
        list_pred : numpy.ndarray
            contient les prédictions pour chaque ligne de X
            
        """
        # Init probs storage matrix
        mat_prob = np.zeros((self.niter, X.shape[0]))

        # for each fitted models
        for j in range(self.niter):
            # Apply the model on test set
            y_prob_rf = self.list_model[j].predict_proba(X)
            # probabilities storage in matrix
            mat_prob[j] = y_prob_rf[:, 1]

        # probas averaging
        list_prob_pred = mat_prob.sum(axis=0) / self.niter
        # voting
        list_pred = [round(elem, 0) for elem in list_prob_pred]

        return list_prob_pred, list_pred

    def feat_importance_BAG_RF(self, X):
        """
        Get features imporance of the final model
        
        
        input
        -----
        X : DataFrame
            dataset
        
        list_model : list
            fitted models list
            
        return
        ------
        feature_importances : DataFrame
            features and importances (O_O)
            
        """
        # Init importance storage matrix
        mat_feat_imp = np.zeros((self.niter, len(X.columns)))

        #  for each fitted models
        for i in range(self.niter):
            # importances storage in matrix
            mat_feat_imp[i] = self.list_model[i].feature_importances_

        # Averaging importances
        list_feat_imp_moy = mat_feat_imp.sum(axis=0) / self.niter

        # output dataframe
        feature_importances = pd.DataFrame(list_feat_imp_moy, index=X.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)

        features_dict = dict(zip(X.columns, list_feat_imp_moy))

        return features_dict


"""
-------------------------------------------------------------------------------------------------------------
"""


def Bagging_sample(df, target, N, replace=False):
    """
    Sample creation with N target=1 and 3*N target=0 observations
        
    input
    -----
    df : DataFrame
        source dataset
        
    target : String
        target
            
    N : int
        number of target=1 observations
        
    replace : Boolean (défaut : False)
        if True, create samples with replacement
            
    return
    ------
    df_bag : DataFrame
        created sample
             
    """
    # split target = 1 / 0
    df_pos = df.loc[(df[target] == 1)]
    df_neg = df.loc[(df[target] == 0)]

    # sample creation
    df_bag = pd.concat((df_pos.sample(n=N, replace=replace), df_neg.sample(n=3 * N, replace=replace)), axis=0)

    return df_bag
