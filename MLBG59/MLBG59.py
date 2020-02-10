"""
https://trello.com/b/mBU1kFpZ/auto-ml
"""
from MLBG59.Utils.Display import print_title1
from MLBG59.Utils.Decorators import timer
from MLBG59.Audit.Audit_Dataset import audit_dataset
from MLBG59.Audit.Get_Outliers import get_cat_outliers, get_num_outliers
from MLBG59.Preprocessing.Date_Data import all_to_date, date_to_anc
from MLBG59.Preprocessing.Missing_Values import fill_all_num, fill_all_cat
from MLBG59.Preprocessing.Outliers import process_cat_outliers, process_num_outliers
from MLBG59.Preprocessing.Categorical_Data import dummy_all_var
from MLBG59.Modelisation.HyperOpt import *


class AutoML(pd.DataFrame):
    """
    Allow the user to quickly move forward the different steps of a score computing task (binary classification)
    * data analysis
    * preprocessing
    * features selection
    * modelisation
    * models selection
    
    Parameters 
    ----------
    _obj : DataFrame
        Source Dataset

    Atributes
    ---------
    x_columns : list
        list of the x features names
        x = num : numerical features
        x = cat : categorical features
        x = date : date features
        x = NA : features which contains NA values
        x = low_var : features with low variance
        x = num_outliers : numerical features which contains outliers
            (deviation from the mean > x*std dev (x=3 by default))
        x = cat_outliers : categorical features which contains outliers (modalities frequency < 5%)
    """

    def __init__(self, *args, target=None):
        super(AutoML, self).__init__(*args)
        # parameters
        self.target = target
        # attributes
        self.num_columns = None
        self.date_columns = None
        self.cat_columns = None
        self.NA_columns = None
        self.low_var_columns = None
        self.num_outliers_columns = None
        self.cat_outliers_columns = None

    def __repr__(self):
        return 'MLBG59 instance'

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def audit(self, verbose=1):
        """
        * Short audit of the dataset
        * Identify features of each type (num, cat, date), features containing NA and features whose variance is null
        
        input
        -----
         > target : string (Default : None)
              target name
         > verbose : int (0/1) (Default : 1)
             get more operations information
            
        return
        ------
         > self.num_columns
         > self.cat_columns
         > self.date_columns
         > self.NA_columns
         > self.low_var_columns
        """
        if verbose > 0:
            print('')
            print_title1('Audit')

        # call std_audit_dataset function
        self.num_columns, self.date_columns, self.cat_columns, self.NA_columns, self.low_var_columns = audit_dataset(
            self, verbose=verbose)

        # created attributes display
        if verbose > 0:
            color_print("\nCreated attributes :  ")
            print("  -> num_columns ")
            print("  -> date_columns ")
            print("  -> cat_columns ")
            print("  -> NA_columns ")
            print("  -> low_var_columns ")

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def get_outliers(self, verbose=1):
        """
        Identify cat and num features which contains outlier
        * num : deviation from the mean > x*std dev (x=3 by default)
        * cat : <x% frequency modalities (x=5 by default)
        
        input
        -----
         > verbose : int (0/1) (Default : 1)
             get more operations information

        return
        ------
         > self.num_outliers_columns
         > self.cat_outliers_columns
        """
        if verbose > 0:
            print('')
            print_title1('Get_outliers')
        # cat outliers
        self.cat_outliers_columns = [
            *get_cat_outliers(self, var_list=self.cat_columns, threshold=0.05, verbose=verbose)]
        # num outliers
        self.num_outliers_columns = [*get_num_outliers(self, var_list=self.num_columns, xstd=4, verbose=verbose)]

        # created attributes display
        if verbose > 0:
            color_print("\nCreated attributes : ")
            print("  -> num_outliers_columns ")
            print("  -> cat_outliers_columns ")

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def preprocess(self, date_ref=None, process_outliers=False, verbose=1):
        """
        Preprocessing of the dataset :
        * Remove features with null variance
        * transform date to time between date and date_ref
        * fill NA values
        * replace outliers
        * one hot encoding
        
        input
        -----
         > date_ref : str (Default : None)
              date_ref to compute time between date en date_ref
         > process_outliers = boolean (Default : False)
              if True, process outliers
         > verbose : int (0/1) (Default : 1)
             get more operations information
        
        return
        ------
        preprocessed dataset
        """
        if verbose > 0:
            print('')
            print_title1('Preprocess')

        target = self.target

        ######################################
        # Remove features with null variance
        ######################################
        if verbose > 0:
            color_print("Remove features with null variance")
            print('  features : ', list(self.low_var_columns))

        df_local = self.copy()

        if self.low_var_columns is not None:
            for col in self.low_var_columns:
                if col in df_local.columns.tolist() and col != self.target:
                    df_local = df_local.drop(col, axis=1)

        # delete removed cols from num_column
        self.num_columns = [x for x in self.num_columns if x not in self.low_var_columns]

        ####################################################
        # Transform date -> time between date and date_ref
        ####################################################
        if verbose > 0:
            strg = "Transform date -> timelapse"
            color_print(strg)

        # Parse date to datetime
        df_local = all_to_date(df_local, var_list=self.date_columns, verbose=0)

        # compute time between date and date_ref
        df_local, new_var_list = date_to_anc(df_local, var_list=None, date_ref=date_ref, verbose=verbose)
        self.num_columns = self.num_columns + new_var_list

        ##################
        # fill NA values
        ##################
        # num features
        if verbose > 0:
            color_print('Fill NA')
            color_print('  Num:')
        df_local = fill_all_num(df_local, var_list=self.num_columns, method='median', top_var_NA=True,
                                verbose=verbose)

        # cat features
        if verbose > 0:
            color_print('  Cat:')
        df_local = fill_all_cat(df_local, var_list=self.cat_columns, method='NR', verbose=verbose)

        ####################
        # replace outliers
        ####################
        if process_outliers:
            # cat features
            if verbose > 0:
                color_print('Outliers processing')
                color_print('  Num:')
            if self.target in self.num_outliers_columns:
                self.num_outliers_columns = self.num_outliers_columns.remove(self.target)
            df_local = process_num_outliers(df_local, self.num_outliers_columns, xstd=4, verbose=verbose)

            # num features
            if verbose > 0:
                color_print('  Cat:')
            df_local = process_cat_outliers(df_local, self.cat_outliers_columns, threshold=0.05, method="percent",
                                        verbose=verbose)

        ####################
        # one hot encoding
        ####################
        if verbose > 0:
            color_print('Categorical features processing')

        df_local = dummy_all_var(df_local, var_list=self.cat_columns, prefix_list=None, keep=False, verbose=verbose)

        self.__dict__.update(df_local.__dict__)
        self.target = target

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def train_predict(self, n_comb=5, comb_seed=None, verbose=1):
        """
        Train and apply models
        
        input
        -----
         > n_comb : int (Default : 10)
             HP combination number
         > comb_seed = int
             random combination seed
         > verbose : int (0/1) (Default : 1)
             get more operations information
            
        return
        ------
         > hyperopt.train_model_dict
         > dict_res_model
         > hyperopt
        """
        if verbose > 0:
            print('')
            print_title1('Train predict')

        # Train/Test split
        df_train, df_test = train_test(self, 0.3)

        # Create Hyperopt object
        hyperopt = Hyperopt(classifier='XGBOOST', grid_param=default_XGB_grid_param, n_param_comb=n_comb,
                            top_bagging=False, comb_seed=comb_seed)

        # Entrainement des modèles
        color_print('training models')
        hyperopt.train(df_train, self.target, verbose=verbose)

        color_print('\napplying models')
        # Application des modèles sur X_test :
        dict_res_model = hyperopt.predict(df_test, self.target, delta_auc=0.03, verbose=verbose)

        # selection best model
        color_print('\nbest model selection')
        best_model_idx, _ = hyperopt.get_best_model(dict_res_model, metric='F1', delta_auc_th=0.03, verbose=verbose)

        if verbose > 0:
            print_title1('best model : ' + str(best_model_idx))
            print(hyperopt.model_res_to_df(dict_res_model, sort_metric='F1'))

        return hyperopt, dict_res_model, best_model_idx
