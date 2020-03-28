import unittest
import pandas as pd
from MLBG59.Select_Features.Select_Features import FeatSelector

df = pd.read_csv('tests/df_test.csv')
df['age'] = df['age'].fillna(df['age'].median())


class TestFeatSelector(unittest.TestCase):
    """
    FeatSelector class
    """
    sel1 = FeatSelector(method='pca')
    sel2 = FeatSelector()

    def test_init(self):
        """ init method """
        self.assertEqual(self.sel1.method, 'pca')

    def test_fit(self):
        """ fit method """
        self.sel1.fit(df, l_var=['age', 'euribor3m'])
        self.assertEqual(self.sel1.l_select_var, ['age', 'euribor3m'])
        self.assertIsNotNone(self.sel1.selector)
        self.assertIsNotNone(self.sel1.scaler)
        self.assertTrue(self.sel1.is_fitted)
        self.assertEqual(df.columns.tolist(),
                         ['age', 'job', 'education', 'euribor3m', 'date_1', 'date_2', "null_var", 'y_yes'])

    def test_transform(self):
        """ transform method"""
        self.sel1.fit(df, l_var=['age', 'euribor3m'])
        df_sel1 = self.sel1.transform(df)
        self.assertEqual(df_sel1.columns.tolist(), ['job', 'education', 'date_1', 'date_2', 'null_var', 'y_yes', 'Dim0',
                                                    'Dim1'])

    def test_fit_transform(self):
        df_sel2 = self.sel2.fit_transform(df, l_var=None)
        self.assertEqual(self.sel2.l_select_var, ['age', 'euribor3m', 'null_var', 'y_yes'])
        self.assertIsNotNone(self.sel2.selector)
        self.assertIsNotNone(self.sel2.scaler)
        self.assertTrue(self.sel2.is_fitted)
        self.assertEqual(df.columns.tolist(),
                         ['age', 'job', 'education', 'euribor3m', 'date_1', 'date_2', "null_var", 'y_yes'])
        self.assertEqual(df_sel2.columns.tolist(), ['job', 'education', 'date_1', 'date_2', 'Dim0', 'Dim1', 'Dim2'])
