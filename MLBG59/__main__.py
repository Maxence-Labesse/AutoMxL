from MLBG59.Utils.Display import print_title1
from MLBG59.Utils.Decorators import timer
from MLBG59.Explore.Get_Info import recap
from MLBG59.Explore.Get_Outliers import get_cat_outliers, get_num_outliers
from MLBG59.Preprocessing.Date_Data import all_to_date, date_to_anc
from MLBG59.Preprocessing.Missing_Values import fill_numerical, fill_categorical
from MLBG59.Preprocessing.Process_Outliers import replace_category, replace_extreme_values
from MLBG59.Preprocessing.Categorical_Data import dummy_all_var
from MLBG59.Modelisation.HyperOpt import *


class AutoML(pd.DataFrame):
    """Covers the complete pipeline of a classification project from the raw dataset to a deployable model.

    AutoML is Built as a class inherited from pandas DataFrame for which each step correponds to a class method that
    can be called with only default parameters or chosen ones.

    - Data exploration (dataset information and outliers analysis)
    - Preprocessing (clean and prepare data)
    - Modelisation (random search)

    Available classifiers : Random Forest and XGBOOST

    
    Parameters 
    ----------
    _obj : DataFrame
        Source Dataset
    d_features : dict (created by audit method)

        {x : list of variables names}

        - x = numerical : numerical features
        - x = categorical : categorical features
        - x = date : date features
        - x = NA : features which contains NA values
        - x = low_variance : list of the low variance features
    d_num_outliers : dict (created with get_outliers method)
        {feature : [lower_limit,upper_limit]}
    d_cat_outliers : dict (created with get_outliers method)
        {feature : outliers categories list}
    """

    def __init__(self, *args, target=None):
        super(AutoML, self).__init__(*args)
        # parameters
        self.target = target
        # attributes
        self.d_features = None
        self.d_num_outliers = None
        self.d_cat_outliers = None

    def __repr__(self):
        return 'MLBG59 instance'

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def recap(self, verbose=False):
        """get and store global informations about the dataset :

        - Variables type (num, cat, date)
        - NA values
        - low variance variables
        
        Parameters
        ----------
        target : string (Default : None)
              target name
        verbose : boolean (Default False)
            Get logging information
            
        Returns
        -------
        dict : self.d_features {x : list of variables names}

        - x = numerical : numerical features
        - x = categorical : categorical features
        - x = date : date features
        - x = NA : features which contains NA values
        - x = low_variance : list of the features with low variance
        """
        if verbose:
            print_title1('\nExplore')

        # call std_audit_dataset function
        self.d_features = recap(
            self, verbose=verbose)

        # created attributes display
        if verbose:
            color_print("\nCreated attributes :  d_features (dict) ")
            print("Keys :")
            print("  -> numerical")
            print("  -> categorical")
            print("  -> date")
            print("  -> NA")
            print("  -> low_variance")

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def get_outliers(self, num_xstd=4, cat_freq=0.05, verbose=False):
        """Identify cat and num features which contains outlier

        * num : x outlier <=> abs(x - mean) > num_xstd * var
        * cat : Modalities with frequency <x% (Default 5%)
        
        Parameters
        ----------
        num_xstd : int (Default : 3)
            Variance gap coef
        cat_freq : float (Default : 0.05)
            Minimum modality frequency
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
        self.d_num_outliers
            {variable : list of categories considered as outliers}
        self.d_num_outliers
            {variable : [lower_limit, upper_limit]}
        """
        if verbose:
            print_title1('\nGet_outliers')
        # cat outliers
        self.d_cat_outliers = get_cat_outliers(self, var_list=self.d_features['categorical'], threshold=cat_freq,
                                               verbose=verbose)
        # num outliers
        self.d_num_outliers = get_num_outliers(self, var_list=self.d_features['numerical'], xstd=num_xstd,
                                               verbose=verbose)

        # created attributes display
        if verbose:
            color_print("\nCreated attributes : ")
            print("  -> d_num_outliers")
            print("  -> d_cat_outliers")

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def preprocess(self, date_ref=None, process_outliers=False, verbose=False):
        """Prepare the data before feeding it to the model :

            - remove low variance features
            - transform date features to timedelta
            - fill missing values
            - process categorical data
            - replace outliers (optional)
        
        Parameters
        ----------
        date_ref : string '%d/%m/%y' (Default : None)
            ref ate to compute timedelta.
            If None, today date
        process_outliers : boolean (Default : False)
              Enable outliers replacement
        verbose : boolean (Default False)
            Get logging information
        
        Returns
        -------
        self
        """
        if verbose:
            print_title1('\nPreprocess')

        target = self.target

        ######################################
        # Remove features with null variance
        ######################################
        if verbose:
            color_print("Remove features with null variance")
            print('  features : ', list(self.d_features['low_variance']))

        df_local = self.copy()

        if self.d_features['low_variance'] is not None:
            for col in self.d_features['low_variance']:
                if col in df_local.columns.tolist() and col != self.target:
                    df_local = df_local.drop(col, axis=1)

        # delete removed cols from num_column
        self.d_features['numerical'] = [x for x in self.d_features['numerical'] if
                                        x not in self.d_features['low_variance']]

        ####################################################
        # Transform date -> time between date and date_ref
        ####################################################
        if verbose:
            strg = "Transform date -> timelapse"
            color_print(strg)

        # Parse date to datetime
        df_local = all_to_date(df_local, var_list=self.d_features['date'], verbose=0)

        # compute time between date and date_ref
        df_local, new_var_list = date_to_anc(df_local, var_list=None, date_ref=date_ref, verbose=verbose)
        self.d_features['numerical'] = self.d_features['numerical'] + new_var_list

        ##################
        # fill NA values
        ##################
        # num features
        if verbose:
            color_print('Fill NA')
            color_print('  Num:')
        df_local = fill_numerical(df_local, var_list=self.d_features['numerical'], method='median', top_var_NA=True,
                                  verbose=verbose)

        # cat features
        if verbose:
            color_print('  Cat:')
        df_local = fill_categorical(df_local, var_list=self.d_features['categorical'], method='NR', verbose=verbose)

        ####################
        # replace outliers
        ####################
        if process_outliers:
            # num features
            if verbose:
                color_print('Outliers processing')
                color_print('  Num:')
            if self.target in self.d_num_outliers:
                self.d_num_outliers.remove(self.target)
            for var in self.d_num_outliers.keys():
                df_local = replace_extreme_values(df_local, var, self.d_num_outliers[var][0],
                                                  self.d_num_outliers[var][1], verbose=verbose)

            # cat features
            if verbose:
                color_print('  Cat:')
            for var in self.d_cat_outliers.keys():
                df_local = replace_category(df_local, var, self.d_cat_outliers[var],
                                            verbose=verbose)

        ####################
        # one hot encoding
        ####################
        if verbose:
            color_print('Categorical features processing')

        df_local = dummy_all_var(df_local, var_list=self.d_features['categorical'], prefix_list=None, keep=False,
                                 verbose=verbose)

        self.__dict__.update(df_local.__dict__)
        self.target = target

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def train_predict(self, clf='XGBOOST', metric='F1', n_comb=10, comb_seed=None, verbose=True):
        """Model hyper-optimisation with random search.

        - Creates random hyper-parameters combinations from HP grid
        - train and test a model for each combination
        - get the best model in respect of a selected metric among valid model

        Available classifiers : Random Forest, XGBOOST (and bagging)
        
        Parameters
        ----------
        clf : string (Default : 'XGBOOST')
            classifier used for modelisation
        metric : string (Default : 'F1')
            objective metric
        n_comb : int (Default : 10)
            HP combination number
        comb_seed = int (Default : None)
            random combination seed
        verbose : boolean (Default False)
            Get logging information
            
        Returns
        -------
        dict
            {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics', 'metrics', 'output'}
        int
            best model index
        DataFrame
            Models info and metrics stored in DataFrame
        """
        if verbose:
            print('')
            print_title1('Train predict')

        # Train/Test split
        df_train, df_test = train_test(self, 0.3)

        # Create Hyperopt object
        hyperopt = Hyperopt(classifier=clf, grid_param=default_XGB_grid_param, n_param_comb=n_comb,
                            top_bagging=False, comb_seed=comb_seed)

        # Entrainement des modèles
        if verbose:
            color_print('training models')
        hyperopt.fit(df_train, self.target, verbose=verbose)

        if verbose:
            color_print('\napplying models')
        # Application des modèles sur X_test :
        dict_res_model = hyperopt.predict(df_test, self.target, delta_auc=0.03, verbose=verbose)

        # selection best model
        if verbose:
            color_print('\nbest model selection')
        best_model_idx, _ = hyperopt.get_best_model(dict_res_model, metric=metric, delta_auc_th=0.03, verbose=False)

        if verbose:
            print_title1('best model : ' + str(best_model_idx))
            print(metric + ' : ' + str(round(dict_res_model[best_model_idx]['metrics'][metric], 4)))
            print('AUC : ' + str(round(dict_res_model[best_model_idx]['metrics']['Roc_auc'], 4)))

        df_model_res = hyperopt.model_res_to_df(dict_res_model, sort_metric=metric)

        return dict_res_model, best_model_idx, df_model_res
