import xgboost
import random
import itertools as it
# import datetime
from lib.Modelisation.Techniques.Bagging import *
from lib.Utils.Utils import *
from datetime import datetime

"""
Defaults HP grid for RF and XGBOOST
"""
default_RF_grid_param = {
    'n_estimators': np.random.uniform(low=20, high=200, size=20).astype(int),
    'max_features': ['auto', 'sqrt'],
    'max_depth': np.random.uniform(low=3, high=10, size=20).astype(int),
    'min_samples_split': [5, 10],
    # 'min_samples_leaf': [1, 2, 4]
}

default_XGB_grid_param = {
    'n_estimators': np.random.uniform(low=100, high=400, size=20).astype(int),
    # 'max_features': ['auto', 'sqrt'],
    'max_depth': np.random.uniform(low=3, high=10, size=20).astype(int),
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'learning_rate': [0.001, 0.003, 0.006, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.3, 0.6],
    'scale_pos_weight': [3, 4, 5, 6, 7, 8, 9],
    'n_jobs': [7]
    # 'max_delta_step': [5]
}

"""
-------------------------------------------------------------------------------------------------------------
"""


class Hyperopt(object):
    """ 
    For a selected algorithm and a hyper-parameters grid.
    Pick N random HPs combinations and train/predict the model for each of them.
    
    Parameters
    ---------
    classifier : string (Default : 'RF')
        classifier
    
    grid_param : dict (Default : default_RF_grid_param)
        HPs grid
        
    n_param_comb : int (Default : 10)
        number of HPS combinations 
       
    Attributes
    ---------
    dict_roc_train : dict
    
    dict_models : dict
    
    dict_features_imp : dict
    
    dict_bagging : dict
    
    df_models_summ : DataFrame
    
    """

    def __init__(self,
                 classifier='RF',
                 grid_param=default_RF_grid_param,
                 n_param_comb=10,
                 top_bagging=False,
                 bagging_param=default_bagging_param,
                 comb_seed=None):

        # parameters
        self.classifier = classifier
        self.grid_param = grid_param
        self.n_param_comb = n_param_comb
        self.top_bagging = top_bagging
        self.bagging_param = bagging_param
        self.comb_seed = comb_seed
        # attributes
        self.train_model_dict = None
        self.bagging_object = None

    def get_params(self, deep=True):
        """"
        return object parameters
    
        return
        -----
        dict containing parameters
        
        """
        return {'classifier': self.classifier,
                'grid_param': self.grid_param,
                'n_param_comb': self.n_param_comb,
                'top_bagging': self.top_bagging,
                'bagging_param': self.bagging_param,
                'comb_seed': self.comb_seed}

    def set_params(self, **params):
        """
        modify object parameters
        
        input
        -----
        params : dict
            dict containing the new parameters {param : value}
            
        """
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Paramètre invalide")
            else:
                setattr(self, k, v)

    def train(self, df_train, target, print_level=1):
        """
        Train the models
        
        input
        -----
        df_train : DataFrame
            training set
        
        target : string
            target name
            
        print_level : int (0->2) (Default : 1)
            level of detail displayed (0 any / 1 medium / 2 full)
            
        return
        ------
        ...
             
        """
        t_ini = datetime.now()

        # Sort HPs grid dict by param name (a->z)
        grid_names = sorted(self.grid_param)
        # random sampling : 'n_param_comb' HPS combinations
        # list(it.product(*(self.grid_param[Name] for Name in grid_names))) create all the possible combinations
        if self.comb_seed != None:
            random.seed(self.comb_seed)

        sample_combinations = random.sample(list(it.product(*(self.grid_param[Name] for Name in grid_names))),
                                            k=self.n_param_comb)

        # Display
        if print_level > 0:
            print('\033[34m' + 'Entrainement avec Grid Search : Nombre de combinaisons d\'HPs :', self.n_param_comb,
                  '\033[0m')
            print('\033[34m' + 'Modèle : ', self.classifier, '\033[0m')

        # init train_model_dict
        self.train_model_dict = {}
        self.bagging_object = {}

        # for each HP combinations :
        for l in range(len(sample_combinations)):

            t_ini_model = datetime.now()

            # Model params
            HP_dict = dict(zip(grid_names, sample_combinations[l]))

            # Model creation
            if self.classifier == 'RF':  # Classifier Random Forest
                clf = RandomForestClassifier(**HP_dict)
            elif self.classifier == 'XGBOOST':
                clf = xgboost.XGBClassifier(**HP_dict)

            # X / y
            y_train = df_train[target]
            X_train = df_train.drop(target, axis=1)

            # Without bagging
            if self.top_bagging == False:

                # model training
                clf_fit = clf.fit(X_train, y_train)

                # features importance
                features_dict = dict(zip(X_train.columns, clf.feature_importances_))

                # classification probas
                y_proba = clf_fit.predict_proba(X_train)[:, 1]
                y_pred = clf_fit.predict(X_train)

            # With bagging   
            elif self.top_bagging == True:

                # init bagging object with default params
                bag = Bagging(clf, **self.bagging_param)

                # model training
                bag.train(df_train, target)
                clf_fit = bag.list_model

                # features importance
                features_dict = bag.feat_importance_BAG_RF(X_train)

                # classification probas
                y_proba, y_pred = bag.predict(df_train.drop(target, axis=1))

                self.bagging_object[l] = bag

            # Model evaluation
            eval_dict = classifier_evaluate(y_train, y_pred, y_proba, print_level=0)

            # store
            train_model = {'HP': HP_dict,
                           'probas': y_proba,
                           'model': clf_fit,
                           'features_importance': features_dict,
                           'evaluation': eval_dict}

            self.train_model_dict[l] = train_model

            t_fin_model = datetime.now()

            if print_level > 0:
                print(str(l + 1) + '/' + str(len(sample_combinations)))
                print(str(tuple(sorted(HP_dict.items()))))
                print('{} Sec.'.format((t_fin_model - t_ini_model).total_seconds()))

        t_fin = datetime.now()
        print('\nTime to train all the models : {} Sec.\n'.format((t_fin - t_ini).total_seconds()))

        return self

    def predict(self, df, target, print_level=1):
        """
        Apply the models
        
        input
        -----
        df : DataFrame
            set to apply the models
            
        target : string
            target name
            
        print_level : int (0->2) (Default : 1)
            level of detail displayed (0 any / 1 medium / 2 full)
        
        return
        ------
        ...
        
        """
        t_ini = datetime.now()

        # Display
        if print_level > 0:
            print('\033[34m' + 'Application des modèles entrainés : \033[0m')

        # init
        res_model_dict = {}

        # X / y
        y = df[target]
        X = df.drop(target, axis=1)

        # For each HPs combination
        for key, value in self.train_model_dict.items():

            model = value['model']

            t_ini_model = datetime.now()

            # Without bagging
            if self.top_bagging == False:

                # classification probas
                y_proba = model.predict_proba(X)[:, 1]

                # classification votes
                y_pred = model.predict(X)

            # With bagging
            elif self.top_bagging == True:

                # classification probs and votes
                y_proba, y_pred = self.dict_bagging[key].predict(X)

                # store
            dict_model = {'y_proba': y_proba,
                          'y_pred': y_pred}

            # compute model metrics
            eval_dict = classifier_evaluate(y, y_pred, y_proba, print_level=0)
            fpr_train, tpr_train = self.train_model_dict[key]['evaluation']['fpr tpr']
            eval_dict['delta_auc'] = abs(auc(fpr_train, tpr_train) - eval_dict["Roc_auc"])

            # store 
            res_model_dict[key] = {'predictions': dict_model,
                                   'evaluation': eval_dict}

            # Display
            if print_level > 0:
                print(value['HP'])
                if print_level > 1:
                    # AUC
                    fpr, tpr = eval_dict["fpr tpr"]
                    roc_auc_train = auc(fpr_train, tpr_train)
                    plt.plot(fpr, tpr, label="AUC test: " + str(round(eval_dict["Roc_auc"], 3)))
                    plt.plot(fpr_train, tpr_train, label="AUC train : " + str(round(roc_auc_train, 3)), color='red')
                    plt.legend(loc=4, fontsize=12)
                    plt.show()

            t_fin_model = datetime.now()
            if print_level > 0:
                print('{} Sec.'.format((t_fin_model - t_ini_model).total_seconds()))

        t_fin = datetime.now()
        print('\nTime to apply all the models : {} Sec.'.format((t_fin - t_ini).total_seconds()))

        return res_model_dict

    def get_best_model(self, test_model_dict, metric='F1', delta_auc=0.03, print_level=1):
        """

        """
        # select valid models (abs(auc_train - auc_test)<0.03)
        valid_model = {}
        for key, param in test_model_dict.items():
            if param['evaluation']['delta_auc'] <= 0.03:
                valid_model[key] = param

        # Best model according to selected metric
        if len(valid_model.keys()) > 0:
            best_model_idx = max(valid_model, key=lambda x: valid_model[x].get('evaluation').get(metric))
        else:
            best_model_idx = None

        return best_model_idx, list(valid_model.keys())

    def model_res_to_df(self, test_model_dict, metric='F1'):
        """
    
        """
        model_col = ['modele']
        HP_col = list(self.train_model_dict[0]['HP'].keys())
        bagging_col = ['bagging']
        metrics_col = ['Accuracy', 'Roc_auc', 'F1', 'Logloss', 'Precision', 'Recall', 'delta_auc']
        feat_imp_col = ['TOP_feat1', 'TOP_feat2', 'TOP_feat3', 'TOP_feat4', 'TOP_feat5']

        df_test = pd.DataFrame(columns=model_col + HP_col + bagging_col + metrics_col + feat_imp_col)

        for key, value in self.train_model_dict.items():
            dict_tmp = {'modele': key}
            dict_tmp.update(value['HP'].copy())
            dict_tmp.update({x: test_model_dict[key]['evaluation'][x] for x in metrics_col})
            dict_tmp.update({'bagging': self.top_bagging})
            df_tmp = pd.DataFrame.from_dict(self.train_model_dict[0]['features_importance'],
                                            orient='index').reset_index().rename(
                columns={'index': 'feat', 0: 'importance'}).sort_values(by='importance', ascending=False).head(5)
            serie_tmp = df_tmp['feat'] + ' ' + round(df_tmp['importance'], 5).astype(str)
            dict_tmp.update(dict(zip(feat_imp_col, serie_tmp.tolist())))

            df_test = df_test.append(dict_tmp, ignore_index=True)

        return df_test.loc[df_test['delta_auc'] <= 0.03].sort_values(by=metric, ascending=False)

    """
    
    
    col=['target','Param', 'TOP_Bagging', 'Top_1_feat', 'Top_2_feat', 'Top_3_feat', 'Top_4_feat', 'Top_5_feat', 'Top_6_feat']
    self.df_models_summ = pd.DataFrame(columns=col)
    
    # store model infos in df_models_summ
            self.df_models_summ = pd.DataFrame(np.array([[
                target,
                str(tuple(sorted(dict_smpl.items()))),
                self.top_bagging,
                str(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[0].name)+' ('+str(round(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[0,0] , 5))+')',
                str(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[1].name)+' ('+str(round(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[1,0] , 5))+')',
                str(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[2].name)+' ('+str(round(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[2,0] , 5))+')',
                str(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[3].name)+' ('+str(round(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[3,0] , 5))+')',
                str(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[4].name)+' ('+str(round(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[4,0] , 5))+')',
                str(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[5].name)+' ('+str(round(self.dict_features_imp[tuple(sorted(dict_smpl.items()))].iloc[5,0] , 5))+')'
            ]]), columns=col).append(self.df_models_summ, ignore_index=True)
            
    col = ['Param','Roc_auc', 'Roc_auc_train', 'Accuracy','Precision','Recall','F1'] + list(self.grid_param.keys())
        df_res=pd.DataFrame(columns=col)         
            
      # store metrics in df_res
            df_res = pd.DataFrame([[
                str(params),
                round(eval_dict['Roc_auc'],3),  
                round(roc_auc_train,3),
                round(eval_dict['Accuracy'],5),
                round(eval_dict['Precision'],5),
                round(eval_dict['Recall'],5),
                round(eval_dict['F1'],5)
            ]+list_hp], columns=col).append(df_res, ignore_index=True)
            
        df_res = pd.merge(df_res,self.df_models_summ,how='inner',on='Param')         
            
    """
