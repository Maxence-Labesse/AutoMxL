from MLBG59.Explore.Explore import *
from MLBG59.Utils.Display import *
import unittest
import pandas as pd

df_test = pd.read_csv('tests/df_test.csv')
df_test_bis = pd.read_csv('tests/df_test_bis.csv')


class Test_get_infos(unittest.TestCase):

    def test_explore(self):
        d_features = explore(df_test, verbose=False)
        # numerical features identification
        self.assertEqual(d_features['numerical'], ['age', 'euribor3m'])
        # boolean features identification
        self.assertEqual(d_features['boolean'], ['y_yes'])
        # categorical features identification
        self.assertEqual(d_features['categorical'], ['job', 'education'])
        # date features identification
        self.assertEqual(d_features['date'], ['date_1', 'date_2'])
        # features containing NA values identification
        self.assertEqual(d_features['NA'], ['job', 'age', 'date_1'])
        # null variance features identification
        self.assertEqual(d_features['low_variance'], ['null_var'])

    def test_get_feautres_type(self):
        l_feat = df_test_bis.columns.tolist()
        l_feat.remove('Date_nai')
        l_feat.remove('Is_man_cat')
        d_features = get_features_type(df_test_bis, l_var=l_feat)
        # date features identification
        self.assertEqual(d_features['date'], ['American_date_nai'])
        # identifier features identification
        self.assertEqual(d_features['identifier'], ['Unnamed: 0', 'Id_cat', 'Id_num'])
        # verbatim features identification
        self.assertEqual(d_features['verbatim'], ['Name', 'Verb'])
        # boolean features identification
        self.assertEqual(d_features['boolean'], ['Sexe', 'Is_man_num'])
        # categorical features identification
        self.assertEqual(d_features['categorical'], ['Age', 'Height', 'Eyes', 'Hair'])
        # numerical features identification
        self.assertEqual(d_features['numerical'], [])

    def test_low_variance(self):
        self.assertEqual(low_variance_features(df_test, verbose=False).index.tolist(), ['null_var'])
