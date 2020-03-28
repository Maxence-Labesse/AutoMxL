from MLBG59.Modelisation.Bagging import *
from MLBG59.Modelisation.HyperOpt import HyperOpt
import unittest
import pandas as pd
import sklearn

df_iris_binary = pd.read_csv('tests/iris_binary.csv')
df_train, df_test = train_test(df_iris_binary, 0.2)
df_test = df_test.drop('Setosa', axis=1)


class TestBagging(unittest.TestCase):
    """
    Bagging class
    """
    bagging = Bagging(clf=RandomForestClassifier(n_estimators=100, max_leaf_nodes=100),
                      n_sample=3,
                      pos_sample_size=1.0,
                      replace=True)

    #
    def test_init(self):
        """ init method """
        self.assertIsNotNone(self.bagging)
        self.assertEqual(type(self.bagging.classifier), sklearn.ensemble.forest.RandomForestClassifier)
        self.assertEqual(self.bagging.niter, 3)
        self.assertEqual(self.bagging.pos_sample_size, 1.0)
        self.assertEqual(self.bagging.replace, True)

    #
    def test_fit(self):
        """ fit method"""
        self.bagging.fit(df_train, 'Setosa')
        for i in range(self.bagging.niter):
            self.assertEqual(type(self.bagging.list_model[i]), sklearn.ensemble.forest.RandomForestClassifier)
        self.assertTrue(self.bagging.is_fitted)

    def test_predict(self):
        """ test method """
        self.bagging.fit(df_train, 'Setosa')
        res_bagging = self.bagging.predict(df_test)
        # le test nul :>
        self.assertEqual(len(res_bagging[0]), df_test.shape[0])
        self.assertEqual(len(res_bagging[1]), df_test.shape[0])


"""
----------------------------------------------------------------------------------------------------------------
"""

default_XGB_grid_param = {
    'n_estimators': np.random.uniform(low=100, high=300, size=20).astype(int),
    'max_features': ['auto', 'log2'],
    'max_depth': np.random.uniform(low=3, high=10, size=20).astype(int),
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'learning_rate': [0.0001, 0.0003, 0.0006, 0.0009, 0.001, 0.003, 0.006, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.3,
                      0.6],
    'scale_pos_weight': [2, 3, 4, 5, 6, 7]}


class TestHyperOpt(unittest.TestCase):
    """
    HyperOpt class
    """
    hyperopt = HyperOpt(classifier='XGBOOST', grid_param=default_XGB_grid_param, n_param_comb=2, bagging=False,
                        comb_seed=1)

    hyperopt.fit(df_iris_binary, target='Setosa')

    def test_init(self):
        """ init method """
        self.assertEqual(self.hyperopt.classifier, 'XGBOOST')
        self.assertEqual(self.hyperopt.grid_param, default_XGB_grid_param)
        self.assertEqual(self.hyperopt.n_param_comb, 2)
        self.assertEqual(self.hyperopt.bagging, False)
        self.assertEqual(self.hyperopt.comb_seed, 1)

    def test_fit(self):
        """ fit method """
        self.hyperopt.fit(df_iris_binary, target='Setosa')
        #
        self.assertEqual(len(self.hyperopt.d_train_model.keys()), 2)

    def test_predict(self):
        """ predict method"""
        d_apply_model = self.hyperopt.predict(df_iris_binary, target='Setosa', delta_auc=0.03)
        self.assertEqual(len(d_apply_model.keys()), 2)

    def test_get_best_model(self):
        """ get_best_model method """
        d_apply_model = self.hyperopt.predict(df_iris_binary, target='Setosa', delta_auc=0.03)
        best_idx, l_valid = self.hyperopt.get_best_model(d_apply_model, metric='F1', delta_auc_th=0.03)
        self.assertIn(best_idx, range(2))
        self.assertLessEqual(len(l_valid), 2)

    def test_model_res_to_df(self):
        """ model_res_to_df """
        d_apply_model = self.hyperopt.predict(df_iris_binary, target='Setosa', delta_auc=0.03)
        df_res = self.hyperopt.model_res_to_df(d_apply_model, sort_metric='F1')
        self.assertEqual(df_res.shape[0], 2)
