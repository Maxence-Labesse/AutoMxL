from AutoMxL.Utils.Display import print_title1
from AutoMxL.Utils.Decorators import timer
from AutoMxL.Explore.Explore import explore
from AutoMxL.Preprocessing.Date import DateEncoder
from AutoMxL.Preprocessing.Missing_Values import NAEncoder
from AutoMxL.Preprocessing.Outliers import OutliersEncoder
from AutoMxL.Preprocessing.Categorical import CategoricalEncoder
from AutoMxL.Modelisation.HyperOpt import *
from AutoMxL.Select_Features.Select_Features import FeatSelector
from time import time

class AML(pd.DataFrame):

    def __init__(self, *args, target=None, **kwargs):
        super(AML, self).__init__(*args, **kwargs)
        assert target != 'target', 'target name cannot be "target"'
        self.target = target
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
        return 'AutoMxL instance'

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
        if verbose:
            start_time = time()
            print_title1('Explore')

        df_local = self.copy()
        if self.target is not None:
            df_local = df_local.drop(self.target, axis=1)

        self.d_features = explore(
            df_local, verbose=verbose)

        self.step = 'explore'

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
        assert self.step in ['explore'], 'apply explore method first'
        assert not self.is_fitted_preprocessing, 'preprocessing encoders already fitted'

        if verbose:
            start_time = time()
            print_title1('Fit and apply preprocessing')

        target = self.target
        df_local = self.copy()

        if verbose:
            color_print("Features removing (zero variance / verbatims / identifiers)")

        l_remove = self.d_features['low_variance'] + self.d_features['verbatim'] + self.d_features['identifier']
        if len(l_remove) > 0:
            df_local = df_local.drop(l_remove, axis=1)

        if verbose:
            print("  >", len(l_remove), "features to remove")
            if len(l_remove) > 0:
                print(" ", l_remove)

        if verbose:
            color_print("Transform date")

        date_encoder = DateEncoder(method='timedelta', date_ref=date_ref)
        date_encoder.fit(self, l_var=self.d_features['date'], verbose=False)
        df_local = date_encoder.transform(df_local, verbose=verbose)

        if verbose:
            color_print('Missing values')

        NA_encoder = NAEncoder()
        NA_encoder.fit(df_local, l_var=None, verbose=False)
        df_local = NA_encoder.transform(df_local, verbose=verbose)

        if process_outliers:
            if verbose:
                color_print('Outliers')
            out_encoder = OutliersEncoder()
            out_encoder.fit(df_local, l_var=None, verbose=False)
            df_local = out_encoder.transform(df_local, verbose=verbose)
        else:
            out_encoder = None

        if verbose:
            color_print('Encode Categorical and boolean')

        cat_col = self.d_features['categorical'] + self.d_features['boolean']
        if self.target is None:
            cat_method = 'one_hot'
            color_print('No target -> one_hot encoding !', 31)

        cat_encoder = CategoricalEncoder(method=cat_method)
        cat_encoder.fit(self, l_var=cat_col, target=self.target, verbose=verbose)
        df_local = cat_encoder.transform(df_local, verbose=verbose)

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

        self.is_fitted_preprocessing = True

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
        if verbose:
            start_time = time()
            print_title1('Apply Preprocessing')

        assert self.is_fitted_preprocessing, "fit first (please)"

        df_local = df.copy()

        if verbose:
            color_print("Remove features (zero variance, verbatims and identifiers")

        if len(self.d_preprocess['remove']) > 0:
            df_local = df_local.drop(self.d_preprocess['remove'], axis=1)
            if verbose:
                print("  >", len(self.d_preprocess['remove']), 'removed features')
        else:
            if verbose:
                print("  > No features to remove")

        if verbose:
            color_print("Transform date")
        df_local = self.d_preprocess['date'].transform(df_local, verbose=verbose)

        if verbose:
            color_print('Missing values')
        df_local = self.d_preprocess['NA'].transform(df_local, verbose=verbose)

        if 'outlier' in list(self.d_preprocess.keys()):
            if verbose:
                color_print('Outliers')
            df_local = self.d_preprocess['outlier'].transform(df_local, verbose=verbose)

        if verbose:
            color_print('Encode categorical and boolean')
            print('\n\t\t>>>', 'preprocess_apply execution time:', round(time() - start_time, 4), 'secs. <<<')
        df_local = self.d_preprocess['categorical'].transform(df_local, verbose=verbose)

        return df_local

    """
    --------------------------------------------------------------------------------------------------------------------
    """

    def select_features(self, method='pca', verbose=False):
        assert self.step in ['preprocess'], 'apply preprocess method'

        target = self.target

        if verbose:
            start_time = time()
            print_title1('Features Selection')

        df_local = self.copy()

        l_select_var = [col for col in df_local.columns.tolist() if col != self.target]

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

    def model_train_test(self, clf='XGBOOST', grid_param=None, metric='F1', delta_auc=0.03, top_bagging=False, n_comb=10,
                         comb_seed=None,
                         verbose=False):
        assert self.step in ['preprocess', 'features_selection'], 'apply preprocess method'

        if verbose:
            start_time = time()
            print_title1('Train predict')

        df_train, df_test = train_test(self, 0.2)

        hyperopt = HyperOpt(classifier=clf, grid_param=grid_param, n_param_comb=n_comb,
                            bagging=top_bagging, comb_seed=comb_seed)

        if verbose:
            color_print('training models')

        hyperopt.fit(df_train, self.target, verbose=verbose)

        if verbose:
            color_print('\napplying models')

        d_fitted_models = hyperopt.predict(df_test, self.target, delta_auc=delta_auc, verbose=verbose)

        if verbose:
            color_print('\nbest model selection')
        best_model_idx, l_valid_models = hyperopt.get_best_model(d_fitted_models, metric=metric, delta_auc_th=delta_auc,
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

    def model_train(self, clf='XGBOOST', grid_param=None, top_bagging=False, n_comb=10, comb_seed=None, verbose=False):
        assert self.step in ['preprocess', 'features_selection'], 'apply preprocess method'

        df_train = self.copy()
        target = self.target

        if verbose:
            start_time = time()
            print_title1('Train Models')

        hyperopt = HyperOpt(classifier=clf, grid_param=grid_param, n_param_comb=n_comb,
                            bagging=top_bagging, comb_seed=comb_seed)

        if verbose:
            color_print('training models')

        hyperopt.fit(df_train, self.target, verbose=verbose)

        self.d_hyperopt = hyperopt
        self.is_fitted_model = True
        self.target = target
        self.step = 'train_model'

        if verbose:
            print('\n\t\t>>>', 'model_train execution time:', round(time() - start_time, 4), 'secs. <<<')

    """
    ------------------------------------------------------------------------------------------------------------------------
    """

    def model_predict(self, df, metric='F1', delta_auc=0.03, verbose=False):
        assert self.is_fitted_model, "model is not fitted yet, apply model_train_predict or model_train methods"

        if verbose:
            start_time = time()
            color_print('\napplying models')

        d_fitted_models = self.d_hyperopt.predict(df, self.target, delta_auc=delta_auc, verbose=verbose)

        if verbose:
            color_print('\nbest model selection')
        best_model_idx, l_valid_models = self.d_hyperopt.get_best_model(d_fitted_models, metric=metric,
                                                                        delta_auc_th=delta_auc,
                                                                        verbose=False)
        df_model_res = self.d_hyperopt.model_res_to_df(d_fitted_models, sort_metric=metric)

        if best_model_idx is not None:
            print_title1('best model : ' + str(best_model_idx))
            print(metric + ' : ' + str(round(d_fitted_models[best_model_idx]['metrics'][metric], 4)))
            print('AUC : ' + str(round(d_fitted_models[best_model_idx]['metrics']['Roc_auc'], 4)))
            if round(d_fitted_models[best_model_idx]['metrics'][metric], 4) == 1.0:
                color_print("C'était pas qu'un physique finalement hein ?", 32)
            print('\n\t\t>>>', 'model_predict execution time:', round(time() - start_time, 4), 'secs. <<<')

        return d_fitted_models, l_valid_models, best_model_idx, df_model_res
