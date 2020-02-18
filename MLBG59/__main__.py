from MLBG59.Utils.Display import print_title1
from MLBG59.Utils.Decorators import timer
from MLBG59.Explore.Get_Infos import recap
from MLBG59.Explore.Get_Outliers import get_cat_outliers, get_num_outliers
from MLBG59.Preprocessing.Date_Data import all_to_date, date_to_anc
from MLBG59.Preprocessing.Missing_Values import fill_numerical, fill_categorical
from MLBG59.Preprocessing.Process_Outliers import replace_category, replace_extreme_values
from MLBG59.Preprocessing.Categorical_Data import dummy_all_var
from MLBG59.Modelisation.HyperOpt import *


class AutoML(pd.DataFrame):
    """Allow the user to quickly move forward the different steps of a score computing task (binary classification)

    * data analysis
    * preprocessing
    * features selection
    * modelisation
    * models selection
    
    Parameters 
    ----------
    _obj : DataFrame
        Source Dataset
    d_features {x : list of variables names} : dict (created by audit method)
        - x = numerical : numerical features
        - x = categorical : categorical features
        - x = date : date features
        - x = NA : features which contains NA values
        - x = low_variance : list of the features with low variance
    d_num_outliers : dict (created by get_outliers method)
        numerical features which contains outliers
    d_cat_outliers : dict (created by get_outliers method)
        categorical features which contains outliers (modalities frequency < 5%)
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
        dict self.d_features {x : list of variables names}

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
    def get_outliers(self, verbose=False):
        """Identify cat and num features which contains outlier
        * num : deviation from the mean > x*std dev (x=3 by default)
        * cat : <x% frequency modalities (x=5 by default)
        
        Parameters
        ----------
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
        dict self.d_num_outliers
        dict self.d_num_outliers
        """
        if verbose:
            print_title1('\nGet_outliers')
        # cat outliers
        self.d_cat_outliers = get_cat_outliers(self, var_list=self.d_features['categorical'], threshold=0.05, verbose=verbose)
        # num outliers
        self.d_num_outliers = get_num_outliers(self, var_list=self.d_features['numerical'], xstd=4, verbose=verbose)

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
        """Preprocessing of the dataset :

        - Remove features with null variance
        - transform date to time between date and date_ref
        - fill NA values
        - replace outliers
        - one hot encoding
        
        Parameters
        ----------
        date_ref : string '%d/%m/%y' (Default : None)
            Date to compute timedelta
            If None, today date
        process_outliers : boolean (Default : False)
              Enable outliers replacement
        verbose : boolean (Default False)
            Get logging information
        
        Returns
        -------
        preprocessed dataset
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
        self.d_features['numerical'] = [x for x in self.d_features['numerical'] if x not in self.d_features['low_variance']]

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

        df_local = dummy_all_var(df_local, var_list=self.d_features['categorical'], prefix_list=None, keep=False, verbose=verbose)

        self.__dict__.update(df_local.__dict__)
        self.target = target

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def train_predict(self, n_comb=10, comb_seed=None, verbose=True):
        """
        Train and apply models
        
        Parameters
        ----------
        n_comb : int (Default : 10)
            HP combination number
        comb_seed = int (Default : None)
            random combination seed
        verbose : int (0/1) (Default : 1)
            get more operations information
            
        Returns
        -------
        dict : hyperopt.train_model_dict
        dict : dict_res_model
        HyperOpt object : hyperopt
        """
        if verbose:
            print('')
            print_title1('Train predict')

        # Train/Test split
        df_train, df_test = train_test(self, 0.3)

        # Create Hyperopt object
        hyperopt = Hyperopt(classifier='XGBOOST', grid_param=default_XGB_grid_param, n_param_comb=n_comb,
                            top_bagging=False, comb_seed=comb_seed)

        # Entrainement des modèles
        color_print('training models')
        hyperopt.fit(df_train, self.target, verbose=verbose)

        color_print('\napplying models')
        # Application des modèles sur X_test :
        dict_res_model = hyperopt.predict(df_test, self.target, delta_auc=0.03, verbose=verbose)

        # selection best model
        color_print('\nbest model selection')
        best_model_idx, _ = hyperopt.get_best_model(dict_res_model, metric='F1', delta_auc_th=0.03, verbose=verbose)

        if verbose:
            print_title1('best model : ' + str(best_model_idx))
            print(hyperopt.model_res_to_df(dict_res_model, sort_metric='F1'))

        df_test = hyperopt.model_res_to_df(dict_res_model, sort_metric='F1')

        return hyperopt, dict_res_model, best_model_idx, df_test
