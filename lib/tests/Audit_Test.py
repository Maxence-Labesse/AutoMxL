from lib.Audit.Audit_Dataset import *
from lib.Audit.Get_Outliers import *
import unittest
import pandas as pd

df_test = pd.read_csv('lib/tests/df_test.csv')


class Test_Audit(unittest.TestCase):

    def test_audit_dataset(self):
        num_columns, date_columns, cat_columns, NA_columns, low_var_columns = audit_dataset(df_test, verbose=0)
        # numerical features identification
        self.assertEqual(num_columns, ['age', 'euribor3m', 'null_var', 'y_yes'])
        # categorical features identification
        self.assertEqual(cat_columns, ['job', 'education'])
        # date features identification
        self.assertEqual(date_columns, ['date_1', 'date_2'])
        # features containing NA values identification
        self.assertEqual(NA_columns, ['job', 'age', 'date_1'])
        # null variance features identification
        self.assertEqual(low_var_columns, ['null_var'])

    def test_is_date(self):
        # date identified as date
        self.assertTrue(is_date(df_test, 'date_1'))
        # not date not identified as date
        self.assertFalse(is_date(df_test, 'job'))

    def test_get_all_dates(self):
        #
        self.assertEqual(get_all_dates(df_test), ['date_1', 'date_2'])

    def test_low_variance(self):
        self.assertEqual(low_variance_features(df_test, verbose=0).index.tolist(), ['null_var'])


"""
-------------------------------------------------------------------------------------------------------------------------
"""


class Test_Get_Outliers(unittest.TestCase):

    def test_get_cat_outliers(self):
        #
        out_cat_dict = get_cat_outliers(df_test, var_list=['job', 'education'], threshold=0.05, verbose=0)
        self.assertEqual(list(out_cat_dict.keys()), ['job'])
        self.assertEqual(out_cat_dict['job'],
                         ['retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown'])

    def test_get_num_outliers(self):
        #
        out_num_dict = get_num_outliers(df_test, var_list=None, xstd=3, verbose=0)
        self.assertEqual(list(out_num_dict.keys()), ['age'])
        self.assertEqual(round(out_num_dict['age'][0], 2), 8.73)
