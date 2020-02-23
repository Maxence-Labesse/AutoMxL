import os
import sys
import unittest
import numpy as np

cwd = os.getcwd()
sys.path.insert(0, os.path.dirname(cwd))
from MLBG59.Explore.Features_Type import *

data = {'Name': ['Tom', 'Nick', 'Krish', np.nan, 'John'],
        'Id_cat': ['S01', 'S02', 'S03', np.nan, 'S05'],
        'Id_num': [np.nan, 22, 33, 44, 10],
        'Verb': ['En double appel', 'Photo de class', np.nan, 'After', 'Printemps blanc'],
        'Age': [20, np.nan, np.nan, 18, 32],
        'Height': [np.nan, 170, 180, 190, 185],
        'Eyes': ['blue', 'red', 'red', 'blue', 'green'],
        'Sexe': ['M', 'M', 'M', np.nan, 'F'],
        'Is_man_cat': ['Oui', 'Oui', 'Oui', 'Non', 'Non'],
        'Is_man_num': [1, 1, 1, 0, 0],
        'Hair': ['brown', 'brown', 'dark', 'blond', 'blond'],
        'Date_nai': ['27/10/2010', np.nan, '04/03/2019', '05/08/1988', '04/08/1988'],
        'American_date_nai': [20101027, np.nan, 20190304, 19880805, 19880804]}


class Test_Features_Type(unittest.TestCase):

    def setUp(self):
        self.df_test = pd.DataFrame(data)

    # test is_date
    def test_is_date(self):
        self.assertTrue(is_date(self.df_test, 'Date_nai'))
        self.assertTrue(is_date(self.df_test, 'American_date_nai'))
        self.assertFalse(is_date(self.df_test, 'Sexe'))

    # test is_identifier
    def test_is_id(self):
        self.assertFalse(is_identifier(self.df_test, 'Name'))
        self.assertTrue(is_identifier(self.df_test, 'Id_cat'))
        self.assertTrue(is_identifier(self.df_test, 'Id_num'))

    # test is_verbatim
    def test_is_verb(self):
        self.assertTrue(is_verbatim(self.df_test, 'Verb'))
        self.assertFalse(is_verbatim(self.df_test, 'Hair'))

    # test is_boolean
    def test_is_bool(self):
        self.assertTrue(is_boolean(self.df_test, 'Is_man_cat'))
        self.assertTrue(is_boolean(self.df_test, 'Is_man_num'))

    # test is_categorical
    def test_is_cat(self):
        self.assertTrue(is_categorical(self.df_test, 'Eyes'))
        self.assertTrue(is_categorical(self.df_test, 'Hair'))
        self.assertFalse(is_categorical(self.df_test, 'Is_man_num'))

    # test features_per_type
    def test_features_per_type(self):
        self.assertEqual(features_from_type(self.df_test, typ='date'), ['Date_nai', 'American_date_nai'])
        self.assertEqual(features_from_type(self.df_test, var_list=['Id_cat', 'Id_num', 'American_date_nai', 'Verb'],
                                            typ='identifier'), ['Id_cat', 'Id_num'])
        self.assertEqual(features_from_type(self.df_test, typ='verbatim'), ['Name', 'Verb'])
        self.assertEqual(features_from_type(self.df_test, typ='boolean'), ['Sexe', 'Is_man_cat', 'Is_man_num'])
        self.assertEqual(
            features_from_type(self.df_test, var_list=['Eyes', 'Hair', 'American_date_nai'], typ='categorical'),
            ['Eyes', 'Hair'])
