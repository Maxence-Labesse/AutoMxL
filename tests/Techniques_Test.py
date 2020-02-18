from MLBG59.Modelisation.Bagging import *
from MLBG59.Utils.Utils import *

import unittest
import pandas as pd
import sklearn

df_iris_binary = pd.read_csv("test/iris_binary.csv")
df_train, df_test = train_test(df_iris_binary, 0.2)
df_test = df_test.drop('Setosa', axis=1)


class Test_Bagging(unittest.TestCase):

    def setUp(self):
        self.bagging = Bagging(classifier=RandomForestClassifier(n_estimators=100, max_leaf_nodes=100),
                               n_sample=3,
                               pos_sample_size=1.0,
                               replace=True)

    # 
    def test_init(self):
        self.assertIsNotNone(self.bagging)
        self.assertEqual(type(self.bagging.classifier), sklearn.ensemble.forest.RandomForestClassifier)
        self.assertEqual(self.bagging.niter, 3)
        self.assertEqual(self.bagging.pos_sample_size, 1.0)
        self.assertEqual(self.bagging.replace, True)

    # 
    def test_get_params(self):
        param_dict = self.bagging.get_params()
        self.assertEqual(type(param_dict['classifier']), sklearn.ensemble.forest.RandomForestClassifier)
        self.assertEqual(param_dict['niter'], 3)
        self.assertEqual(param_dict['pos_sample_size'], 1.0)
        self.assertEqual(param_dict['replace'], True)
        self.assertEqual(param_dict['list_model'], list())

    #
    def test_set_params(self):
        self.bagging.set_params({'niter': 5, 'replace': False, 'test': 'invalid'})
        param_dict = self.bagging.get_params()
        self.assertEqual(param_dict['niter'], 5)
        self.assertEqual(param_dict['pos_sample_size'], 1.0)
        self.assertEqual(param_dict['replace'], False)
        self.assertEqual(param_dict['list_model'], list())

    #
    def test_train(self):
        self.bagging.fit(df_train, 'Setosa')
        for i in range(3):
            self.assertEqual(type(self.bagging.list_model[i]), sklearn.ensemble.forest.RandomForestClassifier)

    #
    def test_predict(self):
        self.bagging.fit(df_train, 'Setosa')
        res_bagging = self.bagging.predict(df_test)
        self.assertEqual(len(res_bagging[0]), df_test.shape[0])
        self.assertEqual(len(res_bagging[1]), df_test.shape[0])

# class Test_bagging_sample(unittest.Testcase)
