from MLBG59.Explore.Explore import *
import unittest
import pandas as pd

# test configt
df_test = pd.read_csv('tests/df_test.csv')
df_test_bis = pd.read_csv('tests/df_test_bis.csv')


class Test_Explore(unittest.TestCase):
    """
    Explore module
    """

    def test_explore(self):
        """ explore function"""
        d_features = explore(df_test, verbose=False)

        # numerical features
        self.assertEqual(d_features['numerical'], ['age', 'euribor3m'])
        # boolean features
        self.assertEqual(d_features['boolean'], ['y_yes'])
        # categorical features
        self.assertEqual(d_features['categorical'], ['job', 'education'])
        # date features
        self.assertEqual(d_features['date'], ['date_1', 'date_2'])
        # features containing NA values
        self.assertEqual(d_features['NA'], ['job', 'age', 'date_1'])
        # null variance features
        self.assertEqual(d_features['low_variance'], ['null_var'])

    def test_get_features_type(self):
        """ get_features_type function """
        l_feat = df_test_bis.columns.tolist()
        l_feat.remove('Date_nai')
        l_feat.remove('Is_man_cat')
        d_features = get_features_type(df_test_bis, l_var=l_feat)

        # date features
        self.assertEqual(d_features['date'], ['American_date_nai'])
        # identifier features
        self.assertEqual(d_features['identifier'], ['Unnamed: 0', 'Id_cat', 'Id_num'])
        # verbatim features
        self.assertEqual(d_features['verbatim'], ['Name', 'Verb'])
        # boolean features
        self.assertEqual(d_features['boolean'], ['Sexe', 'Is_man_num'])
        # categorical features
        self.assertEqual(d_features['categorical'], ['Age', 'Height', 'Eyes', 'Hair'])
        # numerical features
        self.assertEqual(d_features['numerical'], [])

    def test_low_variance(self):
        self.assertEqual(low_variance_features(df_test, verbose=False).index.tolist(), ['null_var'])


"""
---------------------------------------------------------------------------------------------------
"""


class Test_Features_Type(unittest.TestCase):
    """
    Features_Type module
    """

    def test_is_date(self):
        """ is_date function"""
        self.assertTrue(is_date(df_test_bis, 'Date_nai'))
        self.assertTrue(is_date(df_test_bis, 'American_date_nai'))
        self.assertFalse(is_date(df_test_bis, 'Sexe'))

    def test_is_id(self):
        """ is_identifier function"""
        self.assertFalse(is_identifier(df_test_bis, 'Name'))
        self.assertTrue(is_identifier(df_test_bis, 'Id_cat'))
        self.assertTrue(is_identifier(df_test_bis, 'Id_num'))

    def test_is_verb(self):
        """ is_verbatim function"""
        self.assertTrue(is_verbatim(df_test_bis, 'Verb'))
        self.assertFalse(is_verbatim(df_test_bis, 'Hair'))

    def test_is_bool(self):
        """ is_boolean function"""
        self.assertTrue(is_boolean(df_test_bis, 'Is_man_cat'))
        self.assertTrue(is_boolean(df_test_bis, 'Is_man_num'))

    def test_is_cat(self):
        """ is_categorical function"""
        self.assertTrue(is_categorical(df_test_bis, 'Eyes'))
        self.assertTrue(is_categorical(df_test_bis, 'Hair'))
        self.assertFalse(is_categorical(df_test_bis, 'Is_man_num'))

    def test_features_per_type(self):
        """ features_per_type function"""
        # date
        self.assertEqual(features_from_type(df_test_bis, typ='date'), ['Date_nai', 'American_date_nai'])
        # identifier
        self.assertEqual(features_from_type(df_test_bis, l_var=['Id_cat', 'Id_num', 'American_date_nai', 'Verb'],
                                            typ='identifier'), ['Id_cat', 'Id_num'])
        # verbatim
        self.assertEqual(features_from_type(df_test_bis, typ='verbatim'), ['Name', 'Verb'])
        # boolean
        self.assertEqual(features_from_type(df_test_bis, typ='boolean'), ['Sexe', 'Is_man_cat', 'Is_man_num'])
        # cateogircal
        self.assertEqual(
            features_from_type(df_test_bis, l_var=['Eyes', 'Hair', 'American_date_nai'], typ='categorical'),
            ['Eyes', 'Hair'])
