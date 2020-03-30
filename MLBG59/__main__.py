from MLBG59.Utils.Display import print_title1
from MLBG59.Utils.Decorators import timer
from MLBG59.Explore.Explore import explore
from MLBG59.Preprocessing.Date import DateEncoder
from MLBG59.Preprocessing.Missing_Values import NAEncoder
from MLBG59.Preprocessing.Outliers import OutliersEncoder
from MLBG59.Preprocessing.Categorical import CategoricalEncoder
from MLBG59.Modelisation.HyperOpt import *
from MLBG59.Select_Features.Select_Features import FeatSelector
from time import time


class AML(pd.DataFrame):
    """Covers the complete pipeline of a classification project from a raw dataset to a deployable model.

    AML is built as a class inherited from pandas DataFrame. Each Machine Learning step corresponds to method that
    can be called with default or filled parameters.

    - explore: explore dataset and identify features types
    - preprocess: clean and prepare data (optional : outliers processing).
    - select_features: features selection (optional)
    - model_train_predict : split AML in train/test sets to fits/apply models with random search.
      Returns the list of the valid models (without overfitting) and the best one.

    deployment methods:

    - preprocess_apply : apply fitted preprocessing transformation to a new dataset
    - select_features_apply : idem
    - model_apply : apply fitted models to a new dataset


    Notes :

    - A method requires that the former one has been applied (actuel step is given by "step" attribute)
    - Target has to be binary and encoded as int (1/0) (see MLGB59.Start.Encode_Target module if you need help)
    - don't call your target "target" please :>

    Parameters
    ----------
    _obj : DataFrame
        Source Dataset
    target : string
        target name
  """

    def __init__(self, *args, target=None, **kwargs):
        super(AML, self).__init__(*args, **kwargs)
        assert target != 'target', 'target name cannot be "target"'
        # parameters
        self.target = target
        # attributes
        self.step = 'None'
        self.d_features = None
        self.d_preprocess = None
        self.features_selector = None
        self.d_hyperopt = None
        self.is_fitted_preprocessing = False
        self.is_fitted_selector = False
        self.is_fitted_model = False

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    def __repr__(self):
        return 'MLBG59 instance'

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    def duplicate(self):
        res = AML(self)
        res.__dict__.update(self.__dict__)
        return res

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    def explore(self, verbose=False):
        """data exploration and features type identification

        Note :  if you disagree with automated identification, you can directly modify d_features attribute

        Create self.d_features : dict {x : list of variables names}
            - date: date features
            - identifier: identifier features
            - verbatim: verbatim features
            - boolean: boolean features
            - categorical: categorical features
            - numerical: numerical features
            - categorical: categorical features
            - date: date features
            - NA: features which contains NA values
            - low_variance: list of the features with low variance and unique values

        Parameters
        ----------
        verbose : boolean (Default False)
            Get logging information
        """
        if verbose:
            start_time = time()
            print_title1('Explore')

        df_local = self.copy()
        if self.target is not None:
            df_local = df_local.drop(self.target, axis=1)

        # call std_audit_dataset function
        self.d_features = explore(
            df_local, verbose=verbose)

        self.step = 'explore'

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
            print('\n\t\t>>>', 'explore execution time:', round(time() - start_time, 4), 'secs. <<<')

    """
    --------------------------------------------------------------------------------------------------------------------
    """

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
        date_ref : string '%d/%m/%y' (Default : None)
            ref date to compute date features timedelta.
            If None, today date
        process_outliers : boolean (Default : False)
            Enable outliers replacement
        cat_method : string (Default : 'deep_encoder')
            Categorical features encoding method
        verbose : boolean (Default False)
            Get logging information


        Returns
        -------

        create self.d_preprocess : dict {step : transformation}
            - remove: list of the features to remove
            - date: fitted DateEncoder object
            - NA: fitted NAEncoder object
            - categorical: fitted CategoricalEncoder object
            - outlier: fitted OutlierEncoder object
        """
        # check pipe step
        assert self.step in ['explore'], 'apply explore method first'
        assert not self.is_fitted_preprocessing, 'preprocessing encoders already fitted'

        ###############################
        # Fit and apply preprocessing #
        ###############################
        if verbose:
            start_time = time()
            print_title1('Fit and apply preprocessing')

        target = self.target
        df_local = self.copy()

        # Features Removing 'zero variance / verbatims / identifiers)
        if verbose:
            color_print("Features removing (zero variance / verbatims / identifiers)")

        l_remove = self.d_features['low_variance'] + self.d_features['verbatim'] + self.d_features['identifier']
        if len(l_remove) > 0:
            df_local = df_local.drop(l_remove, axis=1)

        if verbose:
            print("  >", len(l_remove), "features to remove")
            if len(l_remove) > 0:
                print(" ", l_remove)

        # Transform date -> time between date and date_ref
        if verbose:
            color_print("Transform date")

        date_encoder = DateEncoder(method='timedelta', date_ref=date_ref)
        date_encoder.fit(self, l_var=self.d_features['date'], verbose=False)
        df_local = date_encoder.transform(df_local, verbose=verbose)

        # Missing Values
        if verbose:
            color_print('Missing values')

        NA_encoder = NAEncoder()
        NA_encoder.fit(df_local, l_var=None, verbose=False)
        df_local = NA_encoder.transform(df_local, verbose=verbose)

        # replace outliers
        if process_outliers:
            if verbose:
                color_print('Outliers')
            out_encoder = OutliersEncoder()
            out_encoder.fit(df_local, l_var=None, verbose=False)
            df_local = out_encoder.transform(df_local, verbose=verbose)
        else:
            out_encoder = None

        # categorical processing
        if verbose:
            color_print('Encode Categorical and boolean')

        cat_col = self.d_features['categorical'] + self.d_features['boolean']
        # apply one-hot encoding if target not filled in class parameters
        if self.target is None:
            cat_method = 'one_hot'
            color_print('No target -> one_hot encoding !', 31)

        # get embedding
        cat_encoder = CategoricalEncoder(method=cat_method)
        cat_encoder.fit(self, l_var=cat_col, target=self.target, verbose=verbose)
        df_local = cat_encoder.transform(df_local, verbose=verbose)

        # store preprocessing params
        self.d_preprocess = {'remove': l_remove, 'date': date_encoder, 'NA': NA_encoder, 'categorical': cat_encoder}
        if out_encoder is not None:
            self.d_preprocess['outlier'] = out_encoder

        if verbose:
            color_print("\nCreated attributes :  d_preprocess (dict) ")
            print("Keys :")
            print("  -> remove")
            print("  -> date")
            print("  -> NA")
            print("  -> categorical")
            print("  -> outlier (optional)")

        # is_fitted
        self.is_fitted_preprocessing = True

        # update self
        self.__dict__.update(df_local.__dict__)
        self.target = target
        self.step = 'preprocess'

        if verbose:
            color_print("New DataFrame size ")
            print("  > row number : ", self.shape[0], "\n  > col number : ", self.shape[1])
            print('\n\t\t>>>', 'proprocess execution time:', round(time() - start_time, 4), 'secs. <<<')

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    def preprocess_apply(self, df, verbose=False):
        """Apply preprocessing.
        Requires preprocess method to have been applied (so that all encoder are fitted)

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
            start_time = time()
            print_title1('Apply Preprocessing')

        # check pipe step and is_fitted
        assert self.is_fitted_preprocessing, "fit first (please)"

        #
        df_local = df.copy()

        # Remove features with zero variance / verbatims and identifiers
        if verbose:
            color_print("Remove features (zero variance, verbatims and identifiers")

        if len(self.d_preprocess['remove']) > 0:
            df_local = df_local.drop(self.d_preprocess['remove'], axis=1)
            if verbose:
                print("  >", len(self.d_preprocess['remove']), 'removed features')
        else:
            if verbose:
                print("  > No features to remove")

        # Transform date -> time between date and date_ref
        if verbose:
            color_print("Transform date")
        df_local = self.d_preprocess['date'].transform(df_local, verbose=verbose)

        # Missing Values
        if verbose:
            color_print('Missing values')
        df_local = self.d_preprocess['NA'].transform(df_local, verbose=verbose)

        # replace outliers
        if 'outlier' in list(self.d_preprocess.keys()):
            if verbose:
                color_print('Outliers')
            df_local = self.d_preprocess['outlier'].transform(df_local, verbose=verbose)

        # categorical processing
        if verbose:
            color_print('Encode categorical and boolean')
            print('\n\t\t>>>', 'preprocess_apply execution time:', round(time() - start_time, 4), 'secs. <<<')
        df_local = self.d_preprocess['categorical'].transform(df_local, verbose=verbose)

        return df_local

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    def select_features(self, method='pca', verbose=False):
        """ fit and apply features selection (optional)

        Parameters
        ----------
        method : string (Default pca)
            method use to select features
        verbose : boolean (Default False)
            Get logging information

        """
        assert self.step in ['preprocess'], 'apply preprocess method'

        target = self.target

        if verbose:
            start_time = time()
            print_title1('Features Selection')

        df_local = self.copy()

        l_select_var = [col for col in df_local.columns.tolist() if col != self.target]

        # df_local = select_features(df=df_local, target=self.target, method=method, verbose=verbose)

        features_selector = FeatSelector(method=method)
        features_selector.fit(df_local, l_var=l_select_var, verbose=verbose)
        df_local = features_selector.transform(df_local, verbose=verbose)

        self.__dict__.update(df_local.__dict__)
        self.target = target
        self.features_selector = features_selector
        self.is_fitted_selector = True
        self.step = 'features_selection'

        if verbose :
            print('\n\t\t>>>', 'select_features execution time:', round(time() - start_time, 4), 'secs. <<<')

    """
        --------------------------------------------------------------------------------------------------------------------
    """

    def select_features_apply(self, df, verbose=False):
        """Apply features selection.

         Requires Select_Features method to have been applied

         Parameters
         ----------
         df : DataFrame
             dataset to apply selection on
         verbose : boolean (Default False)
             Get logging information

         Returns
         -------
         DataFrame : reduced dataset
         """
        # check pipe step and is_fitted
        assert self.is_fitted_selector, "fit first (please)"

        if verbose:
            start_time = time()
            print_title1('Apply select_features')

        df_local = df.copy()

        df_local = self.features_selector.transform(df_local, verbose=verbose)

        if verbose:
            print('\n\t\t>>>', 'select_features_apply execution time:', round(time() - start_time, 4), 'secs. <<<')

        return df_local

    """
        --------------------------------------------------------------------------------------------------------------------
    """

    def model_train_test(self, clf='XGBOOST', grid_param=None, metric='F1', top_bagging=False, n_comb=10,
                         comb_seed=None,
                         verbose=True):
        """train and test models with random search

        - creates models with random hyper-parameters combinations from HP grid
        - splits (random 80/20) train/test sets to fit/apply models
        - identifies valid models |(auc(train)-auc(test)|<0.03
        - gets the best model in respect of a selected metric among valid model


        Notes :

        - Available classifiers : Random Forest, XGBOOST
        - can enable bagggin algo with top_bagging parameter

        Parameters
        ----------
        clf : string (Default : 'XGBOOST')
            classifier used for modelisation
        metric : string (Default : 'F1')
            objective metric
        top_bagging : boolean (Default : False)
            enable Bagging
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
            models information and metrics stored in DataFrame
        """
        assert self.step in ['preprocess', 'features_selection'], 'apply preprocess method'

        if verbose:
            start_time = time()
            print_title1('Train predict')

        # Train/Test split
        df_train, df_test = train_test(self, 0.2)

        # Create Hyperopt object
        hyperopt = HyperOpt(classifier=clf, grid_param=grid_param, n_param_comb=n_comb,
                            bagging=top_bagging, comb_seed=comb_seed)

        # fit model on train set
        if verbose:
            color_print('training models')

        hyperopt.fit(df_train, self.target, verbose=verbose)

        # Apply model on test set
        if verbose:
            color_print('\napplying models')

        d_fitted_models = hyperopt.predict(df_test, self.target, delta_auc=0.03, verbose=verbose)

        # model selection
        if verbose:
            color_print('\nbest model selection')
        best_model_idx, l_valid_models = hyperopt.get_best_model(d_fitted_models, metric=metric, delta_auc_th=0.03,
                                                                 verbose=False)

        df_model_res = hyperopt.model_res_to_df(d_fitted_models, sort_metric=metric)

        if best_model_idx is not None:
            print_title1('best model : ' + str(best_model_idx))
            print(metric + ' : ' + str(round(d_fitted_models[best_model_idx]['metrics'][metric], 4)))
            print('AUC : ' + str(round(d_fitted_models[best_model_idx]['metrics']['Roc_auc'], 4)))
            if round(d_fitted_models[best_model_idx]['metrics'][metric], 4) == 1.0:
                color_print("C'était pas qu'un physique finalement hein ?", 32)
            print('\n\t\t>>>', 'model_train_test execution time:', round(time() - start_time, 4), 'secs. <<<')

        self.d_hyperopt = hyperopt
        self.is_fitted_model = True

        return d_fitted_models, l_valid_models, best_model_idx, df_model_res

    """
    ------------------------------------------------------------------------------------------------------------------------
    """

    def model_train(self, clf='XGBOOST', grid_param=None, top_bagging=False, n_comb=10, comb_seed=None, verbose=True):
        """train models with random search

        - creates models with random hyper-parameters combinations from HP grid
        - fits models on self
        - identifies valid models |(auc(train)-auc(test)|<0.03
        - gets the best model in respect of a selected metric among valid model


        Notes :

        - Available classifiers : Random Forest, XGBOOST (and bagging)
        - can enable bagggin algo with top_bagging parameter

        Parameters
        ----------
        clf : string (Default : 'XGBOOST')
            classifier used for modelisation
        top_bagging : boolean (Default : False)
            enable Bagging
        n_comb : int (Default : 10)
            HP combination number
        comb_seed : int (Default : None)
            random combination seed
        verbose : boolean (Default False)
            Get logging information

        """
        assert self.step in ['preprocess', 'features_selection'], 'apply preprocess method'

        df_train = self.copy()

        if verbose:
            start_time = time()
            print_title1('Train Models')

        # instantiate Hyperopt object
        hyperopt = HyperOpt(classifier=clf, grid_param=grid_param, n_param_comb=n_comb,
                            bagging=top_bagging, comb_seed=comb_seed)

        # fit model on train set
        if verbose:
            color_print('training models')

        # fit hyperopt on self
        hyperopt.fit(df_train, self.target, verbose=verbose)

        self.d_hyperopt = hyperopt
        self.is_fitted_model = True

        if verbose:
            print('\n\t\t>>>', 'model_train execution time:', round(time() - start_time, 4), 'secs. <<<')

    """
    ------------------------------------------------------------------------------------------------------------------------
    """

    def model_predict(self, df, metric='F1', verbose=True):
        """apply fitted models on a dataset

        Parameters
        ----------
        metric : string (Default : 'F1')
            objective metric
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
        assert self.is_fitted_model, "model is not fitted yet, apply model_train_predict or model_train methods"

        if verbose:
            start_time = time()
            color_print('\napplying models')

        # apply models on dataset
        d_fitted_models = self.d_hyperopt.predict(df, self.target, delta_auc=0.03, verbose=verbose)

        # model selection
        if verbose:
            color_print('\nbest model selection')
        best_model_idx, l_valid_models = self.d_hyperopt.get_best_model(d_fitted_models, metric=metric,
                                                                        delta_auc_th=0.03,
                                                                        verbose=False)
        # store model results
        df_model_res = self.d_hyperopt.model_res_to_df(d_fitted_models, sort_metric=metric)

        if best_model_idx is not None:
            print_title1('best model : ' + str(best_model_idx))
            print(metric + ' : ' + str(round(d_fitted_models[best_model_idx]['metrics'][metric], 4)))
            print('AUC : ' + str(round(d_fitted_models[best_model_idx]['metrics']['Roc_auc'], 4)))
            if round(d_fitted_models[best_model_idx]['metrics'][metric], 4) == 1.0:
                color_print("C'était pas qu'un physique finalement hein ?", 32)
            print('\n\t\t>>>', 'model_predict execution time:', round(time() - start_time, 4), 'secs. <<<')

        return d_fitted_models, l_valid_models, best_model_idx, df_model_res
