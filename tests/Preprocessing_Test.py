from MLBG59.Preprocessing.Categorical import *
from MLBG59.Preprocessing.Date import *
from MLBG59.Preprocessing.Outliers import *
from MLBG59.Preprocessing.Missing_Values import *
import unittest
import pandas as pd
import numpy as np

# test config
df = pd.read_csv('tests/df_test_bis.csv')


class Test_Missing_values(unittest.TestCase):
    """
    Test Missing_Values module
    """

    def test_fill_numerical(self):
        """ fill_numerical function"""
        df_fill_all_num = fill_numerical(self.df, ['Age', 'Height'], method='zero', track_num_NA=True, verbose=False)
        self.assertIn('top_NA_Height', df_fill_all_num.columns.tolist())
        self.assertIn('top_NA_Age', df_fill_all_num.columns.tolist())
        self.assertEqual(df_fill_all_num.iloc[0]['Height'], 0)
        self.assertEqual(df_fill_all_num.iloc[1]['Age'], 0)

    def test_fill_categorical(self):
        """ fill_categorical function"""
        df_fill_all_cat = fill_categorical(self.df, l_var=['Name', 'Sexe'], method='NR', verbose=False)
        self.assertEqual(df_fill_all_cat.iloc[3]['Name'], 'NR')
        self.assertEqual(df_fill_all_cat.iloc[3]['Sexe'], 'NR')

    def test_NAEncoder(self):
        """ NAEncoder class"""
        NA_encoder1 = NAEncoder(replace_num_with='median', replace_cat_with='NR', track_num_NA=True)
        NA_encoder1.fit(df, l_var=['Name', 'Age'])
        # df_NA1 = NAEncoder.transform(df, )
        NA_encoder2 = NAEncoder(replace_num_with='zero', replace_cat_with='NR', track_num_NA=False)

        # raw features contain NA
        s_NA = df.isna().sum()[df.isna().sum() > 0]
        self.assertEqual(s_NA.index, ['Name', 'Id_cat', 'Id_num', 'Verb', 'Age', 'Height', 'Sexe', 'Date_nai',
                                      'American_date_nai'])
        # modified features

        # modified values

#
# """
# ------------------------------------------------------------------------------------------------
# """
#
#
# class Test_Date(unittest.TestCase):
#     """
#     Test Date Module
#     """
#
#     def setUp(self):
#         self.df = df_test_bis.copy()
#         self.df_to_date = all_to_date(self.df, ['Date_nai', 'American_date_nai'], verbose=False)
#         self.df_to_anc, self.new_var_list = date_to_anc(self.df_to_date, l_var=['American_date_nai', 'Date_nai'],
#                                                         date_ref='27/10/2010', verbose=False)
#
#     def test_all_to_date(self):
#         self.assertEqual(np.dtype(self.df_to_date['American_date_nai']), 'datetime64[ns]')
#         self.assertEqual(np.dtype(self.df_to_date['Date_nai']), 'datetime64[ns]')
#         self.assertEqual(np.dtype(self.df_to_date['American_date_nai']), 'datetime64[ns]')
#
#     def test_date_to_anc(self):
#         self.assertIn('anc_American_date_nai', self.df_to_anc.columns)
#         self.assertIn('anc_Date_nai', self.df_to_anc.columns)
#         self.assertNotIn('Date_nai', self.df_to_anc.columns)
#         self.assertNotIn('American_Date_nai', self.df_to_anc.columns)
#         self.assertEqual(self.df_to_anc['anc_Date_nai'][0], 0.0)
#         self.assertIn('anc_American_date_nai', self.new_var_list)
#
#
# """
# ------------------------------------------------------------------------------------------------
# """
#
#
# class Test_Categorical(unittest.TestCase):
#     """
#     Test Categorical module
#     """
#
#     def setUp(self):
#         self.df = df_test_bis.copy()
#         self.df_dummy = dummy_all_var(self.df, var_list=['Eyes', 'Sexe'], prefix_list=None, keep=False, verbose=False)
#         self.df_dummy_pref = dummy_all_var(self.df, var_list=['Eyes', 'Sexe'], prefix_list=['Ey', 'Sx'], keep=True,
#                                            verbose=False)
#
#     def test_dummy_all_var(self):
#         self.assertIn('Eyes_blue', self.df_dummy.columns)
#         self.assertIn('Eyes_red', self.df_dummy.columns)
#         self.assertNotIn('Eyes', self.df_dummy.columns)
#         self.assertNotIn('Sexe', self.df_dummy.columns)
#         self.assertEqual(self.df_dummy['Eyes_blue'].tolist(), [1, 0, 0, 1, 0, 1])
#         self.assertEqual(self.df_dummy['Sexe_M'].tolist(), [1, 1, 1, 0, 0, 1])
#         self.assertIn('Ey_blue', self.df_dummy_pref.columns)
#         self.assertIn('Sx_M', self.df_dummy_pref.columns)
#         self.assertIn('Eyes', self.df_dummy_pref.columns)
#         self.assertIn('Sexe', self.df_dummy_pref.columns)
#
#
# """
# ------------------------------------------------------------------------------------------------
# """
#
#
# class Test_Outliers(unittest.TestCase):
#
#     def setUp(self):
#         self.df = df_test_bis.copy()
#         self.df_process_cat = replace_category(self.df, 'Hair', ['blond'], verbose=False)
#         self.df_process_cat = replace_category(self.df_process_cat, 'Name', ['Tom', 'Nick'], verbose=False)
#
#     def test_process_cat_outliers(self):
#         self.assertEqual(self.df_process_cat['Name'].tolist(),
#                          ['outliers', 'outliers', 'Krish', np.nan, 'John', 'Jack'])
#         self.assertEqual(self.df_process_cat['Hair'].tolist(),
#                          ['brown', 'brown', 'dark', 'outliers', 'outliers', 'outliers'])
#
#     def test_process_num_outliers(self):
#         df_outlier_proc = replace_extreme_values(self.df, 'Height', 175, 185)
#         self.assertEqual(df_outlier_proc['Height'].tolist()[1:], [175.0, 180.0, 185.0, 185.0, 185.0])
