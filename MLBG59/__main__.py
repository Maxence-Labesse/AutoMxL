from MLBG59.Utils.Display import print_title1
from MLBG59.Utils.Decorators import timer
from MLBG59.Explore.Explore import explore
from MLBG59.Preprocessing.Date import DateEncoder
from MLBG59.Preprocessing.Missing_Values import NAEncoder
from MLBG59.Preprocessing.Outliers import OutliersEncoder
from MLBG59.Preprocessing.Categorical import CategoricalEncoder
from MLBG59.Modelisation.HyperOpt import *
from MLBG59.Select_Features.Select_Features import select_features


class AutoML(pd.DataFrame):
    """Covers the complete pipeline of a classification project from a raw dataset to a deployable model.

    AutoML is built as a class inherited from pandas DataFrame. Each step corresponds to a class method that
    can be called with default or filled parameters.

    - Data exploration (dataset information and outliers analysis)
    - Preprocessing (clean and prepare data)
    - Modelisation (random search)

    Available classifiers : Random Forest and XGBOOST.

    Note : A method can be applied if the previous one has been applied too.

    Parameters
    ----------
    _obj : DataFrame
        Source Dataset
    target : string
        target name
    step : string
        last method applied on object
    d_features : dict (created by audit method)

        {x : list of variables names}

        - date : date features
        - identifier : identifier features
        - verbatim : verbatim features
        - boolean : boolean features
        - categorical : categorical features
        - numerical : numerical features
        - categorical : categorical features
        - date : date features
        - NA : features which contains NA values
        - low_variance : list of the features with low variance

    d_num_outliers : dict (created with get_outliers method)
        {feature : [lower_limit,upper_limit]}
    d_cat_outliers : dict (created with get_outliers method)
        {feature : outliers categories list}
    """

    def __init__(self, *args, target=None, **kwargs):
        super(AutoML, self).__init__(*args, **kwargs)
        # parameters
        self.target = target
        # attributes
        self.step = 'None'
        self.d_features = None
        self.d_preprocess = None
        self.is_fitted = False

    def __repr__(self):
        return 'MLBG59 instance'

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def explore(self, verbose=False):
        """get and store global information about the dataset :

        - Variables type
        - NA values
        - low variance variables


        Parameters
        ----------
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
        dict : self.d_features {x : list of variables names}

        - date : date features
        - identifier : identifier features
        - verbatim : verbatim features
        - boolean : boolean features
        - categorical : categorical features
        - numerical : numerical features
        - categorical : categorical features
        - date : date features
        - NA : features which contains NA values
        - low_variance : list of the features with low variance

        """
        if verbose:
            print_title1('Explore')

        df_local = self.copy()
        if self.target is not None:
            df_local = df_local.drop(self.target, axis=1)

        # call std_audit_dataset function
        self.d_features = explore(
            df_local, verbose=verbose)

        self.step = 'recap'

        # created attributes display
        if verbose:
            color_print("\nCreated attributes :  d_features (dict) ")
            print("Keys :")
            print("  -> date")
            print("  -> identifier")
            print("  -> verbatim")
            print("  -> boolean")
            print("  -> categorical")
            print("  -> numerical")
            print("  -> date")
            print("  -> NA")
            print("  -> low_variance")

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def preprocess(self, date_ref=None, process_outliers=False,
                   cat_method='deep_encoder', verbose=False):
        """Prepare the data before feeding it to the model :

            - remove low variance features
            - remove identifiers and verbatims features
            - transform date features to timedelta
            - fill missing values
            - process categorical and boolean data (one-hot-encoding or Pytorch NN encoder)
            - replace outliers (optional)

        Parameters
        ----------
        fit_transform : pass
            pass
        date_ref : string '%d/%m/%y' (Default : None)
            ref date to compute timedelta.
            If None, today date
        process_outliers : boolean (Default : False)
            Enable outliers replacement (if get_outliers method applied)
        cat_method : string (Default : 'one_hot')
            Categorical features encoding method

            - one_hot
            - deep_encoder

        verbose : boolean (Default False)
            Get logging information
        """
        # check pipe step
        assert self.step in ['recap'], 'apply recap method first'

        if verbose:
            print_title1('Preprocess')

        #######
        # Fit #
        #######
        # Features Removing 'zero variance / verbatims / identifiers)
        if verbose: color_print("Features removing (zero variance / verbatims / identifiers)")

        l_remove = self.d_features['low_variance'] + self.d_features['verbatim'] + self.d_features['identifier']

        if verbose:
            print("  >", len(l_remove), "features to remove")
            if len(l_remove) > 0: print(l_remove)

        # Transform date -> time between date and date_ref
        if verbose: color_print("Transform date")

        date_encoder = DateEncoder(method='timedelta', date_ref=date_ref)
        date_encoder.fit(self, l_var=self.d_features['date'], verbose=verbose)

        # Missing Values
        if verbose:
            color_print('Missing values')

        NA_encoder = NAEncoder()
        NA_encoder.fit(self, l_var=self.d_features['NA'], verbose=verbose)

        # replace outliers
        if process_outliers:
            if verbose:
                color_print('Outliers')
            out_encoder = OutliersEncoder()
            out_encoder.fit(self, l_var=None, verbose=verbose)
        else:
            out_encoder = None

        # categorical processing
        if verbose:
            color_print('Categorical and boolean features processing')

        cat_col = self.d_features['categorical'] + self.d_features['boolean']
        cat_encoder = CategoricalEncoder(method=cat_method)
        cat_encoder.fit(self, l_var=cat_col, target=self.target, verbose=verbose)

        # store preprocessing params
        self.d_preprocess = {'remove': l_remove, 'date': date_encoder, 'NA': NA_encoder, 'categorical': cat_encoder}
        if out_encoder is not None:
            self.d_preprocess['outlier'] = out_encoder

        # created attributes display
        if verbose:
            color_print("\nCreated attributes :  d_preprocess (dict) ")
            print("Keys :")
            print("  -> remove")
            print("  -> date")
            print("  -> NA")
            print("  -> categorical")
            print("  -> outlier (optional")

        # is_fitted
        self.is_fitted = True

        #############
        # Transform #
        #############
        df_local = self.copy()
        # store target
        target = self.target

        # apply preprocessing
        df_local = self.appply_preprocess(df_local, verbose=verbose)

        # update self
        self.__dict__.update(df_local.__dict__)
        self.target = target
        self.step = 'preprocess'

        # verbose
        if verbose:
            color_print("\nNew DataFrame size ")
        print("  > row number : ", self.shape[0], "\n  > col number : ", self.shape[1])

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def appply_preprocess(self, df, verbose=False):
        """Apply preprocessing

        Parameters
        ----------
        df : DataFrame
            dataset to apply preprocessing on
        verbose : boolean (Default False)
            Get logging information
        Returns
        -------
        DataFrame : Preprocessed dataset
        """
        if verbose:
            print_title1('Preprocess-transform')

        # check pipe step and is_fitted
        assert self.is_fitted, "fit first (please)"

        #
        df_local = df.copy()

        # Remove features with zero variance / verbatims and identifiers
        if verbose:
            color_print("Remove features (zero variance, verbatims and identifiers")

        if len(self.d_preprocess['remove']) > 0:
            df_local = df_local.drop(self.d_preprocess['remove'], axis=1)
            print(self.d_preprocess['Remove']
            Removed
            features
            ')
            else:
            print(" > No features to remove")

            # Transform date -> time between date and date_ref
            if verbose:
                color_print("Transform date")
            df_local = self.d_preprocess['date'].transform(df_local, verbose=verbose)

            # Missing Values
            if verbose:
                color_print('Missing values')
            df_local = self.d_preprocess['NA'].transform(df_local, verbose=verbose)

            # replace outliers
            if self.d_preprocess['outlier'] is not None:
                df_local = self.d_preprocess['outlier'].transform(df_local, verbose=verbose)

            # categorical processing
            if verbose:
                color_print('Categorical and boolean features processing')
            df_local = self.d_preprocess['categorical'].transform(df_local, verbose=verbose)

        return df_local

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def select_features(self, method='pca', verbose=False):
        """Select features to speed up modelisation.
        (May incresea model performance aswell)

        Available methods : pca
        Parameters
        ----------
        method : string (Default : pca)
            method used to select features
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
            DataFrame : reduced datasset
        """
        assert self.step in ['preprocess'], 'apply preprocess method'
        target = self.target
        if verbose:
            print('')
            print_title1('Features Selection')
        df_local = self.copy()
        df_local = select_features(df=df_local, target=self.target, method=method, verbose=verbose)
        self.__dict__.update(df_local.__dict__)
        self.target = target
        self.step = 'features_selection'

    """
        --------------------------------------------------------------------------------------------------------------------
    """

    @timer
    def train_predict(self, clf='XGBOOST', metric='F1', n_comb=10, comb_seed=None, verbose=True):
        """Model hyper-optimisation with random search.

        - Create random hyper-parameters combinations from HP grid
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
        comb_seed : int (Default : None)
            random combination seed
        verbose : boolean (Default False)
            Get logging information

        Returns
        -------
        dict
            {model_index : {'HP', 'probas', 'model', 'features_importance', 'train_metrics', 'metrics', 'output'}
        list
            valid model indexes
        int
            best model index
        DataFrame
            Models information and metrics stored in DataFrame
        """
        assert self.step in ['preprocess', 'features_selection'], 'apply preprocess method'

        if verbose:
            print('')
            print_title1('Train predict')

        # Train/Test split
        df_train, df_test = train_test(self, 0.3)

        # Create Hyperopt object
        hyperopt = Hyperopt(classifier=clf, grid_param=None, n_param_comb=n_comb,
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
        best_model_idx, l_valid_models = hyperopt.get_best_model(dict_res_model, metric=metric, delta_auc_th=0.03,
                                                                 verbose=False)

        df_model_res = hyperopt.model_res_to_df(dict_res_model, sort_metric=metric)

        if best_model_idx is not None:
            print_title1('best model : ' + str(best_model_idx))
            print(metric + ' : ' + str(round(dict_res_model[best_model_idx]['metrics'][metric], 4)))
            print('AUC : ' + str(round(dict_res_model[best_model_idx]['metrics']['Roc_auc'], 4)))

        return dict_res_model, l_valid_models, best_model_idx, df_model_res
