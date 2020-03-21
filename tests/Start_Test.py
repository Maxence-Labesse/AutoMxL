import unittest
import pandas as pd
from MLBG59.Start.Load import get_delimiter, import_data
from MLBG59.Start.Encode_Target import category_to_target, range_to_target

# test config
file = 'tests/df_test.csv'
var = 'job'
cat = 'admin.'
df_test = pd.read_csv('tests/df_test.csv')
df_test_bis = pd.read_csv('tests/df_test_bis.csv')
raw_target = 'Height'


class Test_Load(unittest.TestCase):
    """
    Encode module
    """

    def test_get_delimiter(self):
        """test get_delimiter function"""
        # identify delimiter
        self.assertEqual(get_delimiter(file), ',')

    def test_import_data(self):
        """test import_data function"""
        # DataFrame created
        self.assertEqual(type(import_data(file=file, verbose=False)), pd.DataFrame)


"""
------------------------------------------------------------------------------------------------
"""


class Test_Encode_Target(unittest.TestCase):
    """
    Encode_Target module
    """

    def test_cat_to_target(self):
        """cat_to_target function"""
        df_test_cat_target, new_var = category_to_target(df_test, var, cat)
        # test new target name
        self.assertEqual(new_var, var + '_' + cat)
        # new target in new dataset
        self.assertIn(new_var, df_test_cat_target.columns.tolist())
        # old target removed from new dataset
        self.assertNotIn(var, df_test_cat_target.columns.tolist())
        # volumetry test
        self.assertEqual(df_test[var].value_counts()[cat], df_test_cat_target[new_var].sum())

    def test_range_to_target(self):
        """range_to_target function"""
        df_range_target, new_var = range_to_target(df_test_bis, var=raw_target, min=180, max=185, verbose=False)
        # new target in new dataset
        self.assertIn(new_var, df_range_target.columns.tolist())
        # old target removed from new dataset
        self.assertNotIn(raw_target, df_range_target.columns.tolist())
        # lower and upper filled
        self.assertEqual(df_range_target[new_var].tolist(), [0, 0, 1, 0, 1, 1])
        # only lower filled
        df_range_target, new_var = range_to_target(df_test_bis, var=raw_target, min=180, verbose=False)
        self.assertEqual(df_range_target[new_var].tolist(), [0, 0, 1, 1, 1, 1])
        # only upper filled
        df_range_target, new_var = range_to_target(df_test_bis, var=raw_target, max=185, verbose=False)
        self.assertEqual(df_range_target[new_var].tolist(), [0, 1, 1, 0, 1, 1])
