from AutoMxL.__main__ import AML
import unittest
import pandas as pd
from AutoMxL.Preprocessing.Categorical import CategoricalEncoder
from AutoMxL.Preprocessing.Date import DateEncoder
from AutoMxL.Preprocessing.Missing_Values import NAEncoder
from AutoMxL.Select_Features.Select_Features import FeatSelector
import numpy as np

# import numpy as np

# Defaults HP grid for RF
default_RF_grid_param = {
    'n_estimators': np.random.uniform(low=20, high=100, size=20).astype(int),
    'max_features': ['auto', 'log2'],
    'max_depth': np.random.uniform(low=2, high=10, size=20).astype(int),
    'min_samples_split': [5, 10, 15]}

df_test = pd.read_csv('tests/df_test.csv')
# init
auto_df_init = AML(df_test, target='y_yes')
# explore
auto_df_explore = auto_df_init.duplicate()
auto_df_explore.explore()
# preprocess
auto_df_preprocess = auto_df_explore.duplicate()
auto_df_preprocess.preprocess(date_ref=None, process_outliers=False, cat_method='one_hot')
# select_features
auto_df_select = auto_df_preprocess.duplicate()
auto_df_select.select_features(method='pca')
# model_train_test
auto_df_model = auto_df_select.duplicate()
d_res_model, l_valid, best_model_idx, df_res = auto_df_model.model_train_test(grid_param=default_RF_grid_param,
                                                                              n_comb=5, comb_seed=2, verbose=True)

# apply
df_prep = auto_df_select.preprocess_apply(df_test)
df_sel = auto_df_select.select_features_apply(df_prep)


class TestInit(unittest.TestCase):
    """ init method"""

    # Test autoML object instantiation from DataFrame
    def test_df_is_not_none(self):
        self.assertIsNotNone(auto_df_init)

    # Test target Attribute
    def test_target_is_not_none(self):
        self.assertIsNotNone(auto_df_init.target)


"""
-------------------------------------------------------------------------------------------------------------------------
"""

# Instantiate autoML object from df_test and target
auto_df = AML(df_test.copy(), target='y_yes')
# Explore
auto_df.explore(verbose=False)


class TestExplore(unittest.TestCase):
    """ explore method """

    def test_d_features(self):
        # numerical features
        self.assertEqual(auto_df_explore.d_features['numerical'], ['age', 'euribor3m'])
        # boolean features
        self.assertEqual(auto_df_explore.d_features['boolean'], [])
        # categorical features
        self.assertEqual(auto_df_explore.d_features['categorical'], ['job', 'education'])
        # date features
        self.assertEqual(auto_df_explore.d_features['date'], ['date_1', 'date_2'])
        # features containing NA values
        self.assertEqual(auto_df_explore.d_features['NA'], ['job', 'age', 'date_1'])
        # null variance features
        self.assertEqual(auto_df_explore.d_features['low_variance'], ['null_var'])

        # unchange dataset
        self.assertEqual(df_test.columns.tolist(), auto_df_explore.columns.tolist())
        self.assertTrue(auto_df_explore.isnull().sum().max() > 0)


"""
-------------------------------------------------------------------------------------------------------------------------
"""


class TestPreprocess(unittest.TestCase):
    """ preprocess method """

    def test_encoders(self):
        self.assertEqual(auto_df_preprocess.d_preprocess['remove'], ['null_var'])
        self.assertTrue(isinstance(auto_df_preprocess.d_preprocess['date'], DateEncoder))
        self.assertTrue(isinstance(auto_df_preprocess.d_preprocess['NA'], NAEncoder))
        self.assertTrue(isinstance(auto_df_preprocess.d_preprocess['categorical'], CategoricalEncoder))
        self.assertNotIn('outlier', auto_df_preprocess.d_preprocess.keys())

    def test_preprocessing(self):
        self.assertTrue(auto_df_preprocess.isnull().sum().max() == 0)
        self.assertEqual(auto_df_preprocess.columns.tolist(), auto_df_preprocess._get_numeric_data().columns.tolist())
        self.assertTrue(auto_df_preprocess.is_fitted_preprocessing)


"""
-------------------------------------------------------------------------------------------------------------------------
"""


class TestPreprocessApply(unittest.TestCase):
    """ preproces method """

    def test_preprocessing_apply(self):
        self.assertTrue(auto_df_preprocess.isnull().sum().max() == 0)
        self.assertEqual(auto_df_preprocess.columns.tolist(), auto_df_preprocess._get_numeric_data().columns.tolist())
        self.assertEqual((df_prep == auto_df_preprocess).sum().min(), 41188)


"""
-------------------------------------------------------------------------------------------------------------------------
"""


class TestSelectFeatures(unittest.TestCase):
    """ select_features method """

    def test_select_features(self):
        #
        self.assertLess(auto_df_select.shape[1], auto_df_preprocess.shape[1])
        self.assertFalse(auto_df_preprocess.is_fitted_selector)
        self.assertTrue(auto_df_select.is_fitted_selector)
        self.assertTrue(isinstance(auto_df_select.features_selector, FeatSelector))


"""
-------------------------------------------------------------------------------------------------------------------------
"""


class TestSelectFeaturesApply(unittest.TestCase):
    """ select_features method """

    def test_select_features_apply(self):
        #
        self.assertLess(df_sel.shape[1], df_prep.shape[1])
        self.assertEqual((df_sel == auto_df_select).sum().min(), 41188)


"""
-------------------------------------------------------------------------------------------------------------------------
"""


class TestModelTrainTest(unittest.TestCase):
    """ method """

    def test_comb_samples(self):
        # tests HP comb values
        self.assertEqual(auto_df_model.hyperopt.d_train_model[0]['HP']['min_samples_split'], 15)

    def test_best_model(self):
        if best_model_idx is not None:
            # test best model is valid (delta_auc)
            self.assertTrue(d_res_model[best_model_idx]['metrics']['delta_auc'] < 0.03)
            # best model have the max F1 score among valid models
            for i in range(5):
                if d_res_model[i]['metrics']['delta_auc'] < 0.03:
                    self.assertTrue(
                        d_res_model[best_model_idx]['metrics']['F1'] >= d_res_model[i]['metrics']['F1'])
