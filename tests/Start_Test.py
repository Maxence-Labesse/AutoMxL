import os
import sys

cwd = os.getcwd()
sys.path.insert(0, os.path.dirname(cwd))
import unittest
import pandas as pd
from MLBG59.Start.Load import get_delimiter, import_data
from MLBG59.Start.Encode_Target import category_to_target

file = 'df_test.csv'
var = 'job'
cat = 'admin.'


class Test_Load(unittest.TestCase):

    def test_get_delimiter(self):
        self.assertEqual(get_delimiter(file), ',')

    def test_import_data(self):
        self.assertIsNotNone(type(import_data(file=file, verbose=False)), pd.DataFrame)

    def test_cat_to_target(self):
        df_test = pd.read_csv('df_test.csv')
        df_test_mod, new_var = category_to_target(df_test, var, cat)

        #
        self.assertEqual(new_var, var + '_' + cat)
        # new var is created in new dataset
        self.assertIn(new_var, df_test_mod.columns.tolist())
        # old var is removed from new dataset
        self.assertNotIn(var, df_test_mod.columns.tolist())
        # old var is still in old dataset
        self.assertIn(var, df_test.columns.tolist())
        # volumetry test
        self.assertEqual(df_test[var].value_counts()[cat], df_test_mod[new_var].sum())
