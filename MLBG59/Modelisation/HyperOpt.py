""" Hyperopt class :
Model hyper-optimisation with random search

- Hyperopt (class) : Model hyper-optimisation with random search

"""
import xgboost
import random
import itertools as it
# import datetime
from MLBG59.Modelisation.Bagging import *
from MLBG59.Modelisation.Utils import *
from MLBG59.Utils.Display import color_print
from datetime import datetime
from MLBG59.param_config import default_bagging_param, default_RF_grid_param, default_XGB_grid_param


class HyperOpt(object):
    """Model hyper-optimisation with random search :

    - From a hyper-parameters grid, creates random HPs combinations
    - train a model for each combination
    - apply the model
    
    Parameters
    ----------
    classifier : string (Default : 'RF')
        classifier for modelisation
    grid_param : dict (Default : Default_RF_grid_param)
        HP grid
    n_param_comb : int (Default : 10)
        number of HP combinations
    bagging : Boolean (Default = False)
        use bagging method
    bagging_param : n-uple
        bagging parameters (Default : default_bagging_param (Bagging module))
    train_model_dict (created with fit method) : dict
        {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics'}
    bagging_object : Bagging
        bagging object
    comb_seed : int
        seed for randomized HP combinations
    """

    def __init__(self,
                 classifier='RF',
                 grid_param=None,
                 n_param_comb=10,
                 bagging=False,
                 bagging_param=default_bagging_param,
                 comb_seed=None):

        # parameters
        if grid_param is None:
            if classifier == 'RF':
                self.grid_param = default_RF_grid_param
            elif classifier == 'XGBOOST':
                self.grid_param = default_XGB_grid_param
        else:
            self.grid_param = grid_param
        self.classifier = classifier
        self.n_param_comb = n_param_comb
        self.bagging = bagging
        self.bagging_param = bagging_param
        self.comb_seed = comb_seed
        # attributes
        self.d_train_model = {}
        self.d_bagging = {}
        self.is_fitted = False

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def get_params(self):
        """Return Hyperopt object parameters

        Returns
        -------
        dict
            {param : value}
        """
        return {'classifier': self.classifier,
                'grid_param': self.grid_param,
                'n_param_comb': self.n_param_comb,
                'top_bagging': self.bagging,
                'bagging_param': self.bagging_param,
                'comb_seed': self.comb_seed}

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def fit(self, df_train, target, verbose=False):
        """Fit a model for each HP combination
        
        Parameters
        ----------
        df_train : DataFrame
            Training dataset
        target : string
            Target name
        verbose : boolean (Default False)
            Get logging information
            
        Returns
        -------
        self.train_model_dict (created with fit method) : dict
            {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics'}
        """
        # X / y
        y_train = df_train[target]
        X_train = df_train.drop(target, axis=1)

        # Sort HPs grid dict by param name (a->z)
        grid_names = sorted(self.grid_param)
        # random sampling : 'n_param_comb' HPS combinations
        # list(it.product(*(self.grid_param[Name] for Name in grid_names))) create all the possible combinations
        if self.comb_seed is not None:
            random.seed(self.comb_seed)

        sample_combinations = random.sample(list(it.product(*(self.grid_param[Name] for Name in grid_names))),
                                            k=self.n_param_comb)

        if verbose :
            print('\033[34m' + 'Random search:', self.n_param_comb, 'HP combs', '\033[0m')
            print('\033[34m' + 'Model : ', self.classifier, '\033[0m')

        # for each HP combination :
        for model_idx in range(len(sample_combinations)):
            t_ini_model = datetime.now()

            # Model params in dict
            HP_dict = dict(zip(grid_names, sample_combinations[model_idx]))

            # instantiate model
            if self.classifier == 'RF':  # Classifier Random Forest
                clf = RandomForestClassifier(**HP_dict)
            # elif self.classifier == 'XGBOOST':
            else:
                clf = xgboost.XGBClassifier(**HP_dict)

            # disabling bagging
            if not self.bagging:

                # model training
                clf_fit = clf.fit(X_train, y_train)
                # features importance
                features_dict = dict(zip(X_train.columns, clf.feature_importances_))
                # outputs
                y_proba = clf_fit.predict_proba(X_train)[:, 1]
                y_pred = clf_fit.predict(X_train)

            # enabling bagging
            else:
                # init bagging object with default params
                bag = Bagging(clf, **self.bagging_param)
                # model training
                bag.fit(df_train, target)
                clf_fit = bag.list_model
                # features importance
                features_dict = bag.bag_feature_importance(X_train)
                # classification probas
                y_proba, y_pred = bag.predict(df_train.drop(target, axis=1))

                self.d_bagging[model_idx] = bag

            # Model evaluation
            eval_dict = classifier_evaluate(y_train, y_pred, y_proba, verbose=0)

            # store
            train_model = {'HP': HP_dict,
                           'model': clf_fit,
                           'features_importance': features_dict,
                           'train_output': {'y_proba': y_proba, 'y_pred': y_pred},
                           'train_metrics': eval_dict}

            # store model results for each combination
            self.d_train_model[model_idx] = train_model

            # Fitted !
            self.is_fitted = True

            if verbose:
                t_fin_model = datetime.now()
                print(str(model_idx + 1) + '/' + str(len(sample_combinations)) +
                      ' >> {} Sec.'.format((t_fin_model - t_ini_model).total_seconds()))

        return self

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def predict(self, df, target, delta_auc, verbose=False):
        """Apply the models
        
        Parameters
        ----------
        df : DataFrame
            Dataset to apply the models
        target : string
            Target name
        delta_auc_th : float
            Threshold for valid models : abs(auc(train) - auc(test))
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
        dict
            {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics', 'metrics', 'output'}
        """
        assert self.is_fitted, 'fit first'

        d_apply_model = self.d_train_model

        # X / y
        y = df[target]
        X = df.drop(target, axis=1)

        # For each HPs combination
        for key, value in self.d_train_model.items():
            t_ini_model = datetime.now()

            modl = value['model']

            # Without bagging
            if not self.bagging:

                # classification probas
                y_proba = modl.predict_proba(X)[:, 1]

                # classification votes
                y_pred = modl.predict(X)

            # With bagging
            elif self.bagging:

                # classification probs and votes
                y_proba, y_pred = self.d_bagging[key].predict(X)

            # store outputs
            d_output = {'y_proba': y_proba,
                        'y_pred': y_pred}

            # compute model metrics
            eval_dict = classifier_evaluate(y, y_pred, y_proba, verbose=0)
            eval_dict['delta_auc'] = abs(self.d_train_model[key]['train_metrics']['Roc_auc'] - eval_dict["Roc_auc"])

            # store 
            d_apply_model[key]['outputs'] = d_output
            d_apply_model[key]['metrics'] = eval_dict

            # print metrics
            if verbose:
                print(value['HP'])
                if eval_dict['delta_auc'] <= delta_auc:
                    c_code = 32
                else:
                    c_code = 31

                color_print(
                    ' > AUC test: ' + str(round(eval_dict["Roc_auc"], 3)) + ' train: ' + str(
                        round(self.d_train_model[key]['train_metrics']['Roc_auc'], 3)) +
                    ' / F1: ' + str(round(eval_dict['F1'], 3)) +
                    ' / prec: ' + str(round(eval_dict['Precision'], 3)) +
                    ' / recall: ' + str(round(eval_dict['Recall'], 3)), color_code=c_code)

                t_fin_model = datetime.now()
                print('{} Sec.'.format((t_fin_model - t_ini_model).total_seconds()))

        return d_apply_model

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def get_best_model(self, d_model_info, metric='F1', delta_auc_th=0.03, verbose=False):
        """Identify valid models according to delta auc (test/train).
        Get the best model in respect of a selected metric among valid model

        Parameters
        ----------
        d_model_info : dict
            {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics', 'metrics', 'output'}
        metric : string (default = F1-score)
            Metric used to get the best model
        delta_auc_th : float
            Threshold for valid models : abs(auc(train) - auc(test))
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
        int
            Best model index
        list
            Valid model indexes
        """
        # select valid models (abs(auc_train - auc_test)<0.03)
        valid_model = {}
        for key, param in d_model_info.items():
            if param['metrics']['delta_auc'] <= delta_auc_th:
                valid_model[key] = param

        # Best model according to selected metric
        if len(valid_model.keys()) > 0:
            best_model_idx = max(valid_model, key=lambda x: valid_model[x].get('metrics').get(metric))
            if verbose:
                print(' >', len(valid_model.keys()), ' valid models |auc(train)-auc(test)|<=' + str(delta_auc_th))
                print(' > best model : ' + str(best_model_idx))
        else:
            best_model_idx = None
            print('0 valid model')

        return best_model_idx, list(valid_model.keys())

    """
    ---------------------------------------------------------------------------------------------------------------
    """

    def model_res_to_df(self, d_model_infos, sort_metric='F1'):
        """Store models summary in DataFrame

        Parameters
        ----------
        d_model_info : dict
            {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics', 'metrics', 'output'}
        sort_metric : string (default = 'F1')
            metric to sort models (descendant)

        Returns
        -------
        DataFrame
            model infos and metrics
        """
        # dataFrame columns names
        model_col = ['model_index']
        HP_col = list(self.d_train_model[0]['HP'].keys())
        bagging_col = ['bagging']
        metrics_col = ['Accuracy', 'Roc_auc', 'F1', 'Logloss', 'Precision', 'Recall', 'delta_auc']
        feat_imp_col = ['TOP_feat1', 'TOP_feat2', 'TOP_feat3', 'TOP_feat4', 'TOP_feat5']

        df_local = pd.DataFrame(columns=model_col + HP_col + bagging_col + metrics_col + feat_imp_col)

        # store informations in df
        for key, value in self.d_train_model.items():
            dict_tmp = {'model_index': key}
            dict_tmp.update(value['HP'].copy())
            dict_tmp.update({x: d_model_infos[key]['metrics'][x] for x in metrics_col})

            dict_tmp.update({'bagging': self.bagging})
            df_tmp = pd.DataFrame.from_dict(self.d_train_model[key]['features_importance'],
                                            orient='index').reset_index().rename(
                columns={'index': 'feat', 0: 'importance'}).sort_values(by='importance', ascending=False).head(5)
            serie_tmp = df_tmp['feat'] + ' ' + round(df_tmp['importance'], 5).astype(str)
            dict_tmp.update(dict(zip(feat_imp_col, serie_tmp.tolist())))

            df_local = df_local.append(dict_tmp, ignore_index=True)

        return df_local.loc[df_local['delta_auc'] <= 0.03].sort_values(by=sort_metric, ascending=False)
