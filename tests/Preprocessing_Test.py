from MLBG59.Preprocessing.Categorical_Data import *
from MLBG59.Preprocessing.Date_Data import *
from MLBG59.Preprocessing.Outliers import *
from MLBG59.Preprocessing.Missing_Values import *

import unittest
import pandas as pd
import numpy as np

data = {'Name': ['Tom', 'nick', 'krish', np.nan],
        'Age': [20, np.nan, np.nan, 18],
        'Height': [np.nan, 170, 180, 190],
        'Eyes': ['blue', 'red', 'red', 'blue'],
        'Sexe': ['M', 'M', 'M', np.nan],
        'Hair': ['brown', 'brown', 'brown', 'blond'],
        'Date_nai': ['27/10/2010', np.nan, '04/03/2019', '05/08/1988'],
        'American_date_nai': [20101027, np.nan, 20190304, 19880805]}


class Test_Missing_values(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(data)

    def test_fill_num(self):
        df_fill_med = fill_num(self.df, 'Age', method='median', top_var_NA=True)
        df_fill_zero = fill_num(self.df, 'Height', method='zero', top_var_NA=True)
        df_fill_mean = fill_num(self.df, 'Height', method='mean', top_var_NA=True)
        self.assertEqual(df_fill_med.iloc[1]['Age'], 19)
        self.assertEqual(df_fill_med['top_NA_Age'].sum(), 2)
        self.assertEqual(df_fill_med.loc[1]['top_NA_Age'], 1)
        self.assertEqual(df_fill_med.iloc[1]['Age'], 19)
        self.assertEqual(df_fill_zero.iloc[0]['Height'], 0)
        self.assertEqual(df_fill_mean.iloc[0]['Height'], 180)

    def test_fill_all_num(self):
        df_fill_all_num = fill_all_num(self.df, ['Age', 'Height'], method='zero', top_var_NA=True, verbose=0)
        self.assertIn('top_NA_Height', df_fill_all_num.columns.tolist())
        self.assertIn('top_NA_Age', df_fill_all_num.columns.tolist())
        self.assertEqual(df_fill_all_num.iloc[0]['Height'], 0)
        self.assertEqual(df_fill_all_num.iloc[1]['Age'], 0)

    def test_fill_cat(self):
        df_fill_nr = fill_cat(self.df, 'Name', method='NR', top_var_NA=True)
        self.assertEqual(df_fill_nr.iloc[3]['Name'], 'NR')
        self.assertEqual(df_fill_nr['top_NA_Name'].sum(), 1)
        self.assertEqual(df_fill_nr.loc[3]['top_NA_Name'], 1)

    def test_fill_all_cat(self):
        df_fill_all_cat = fill_all_cat(self.df, var_list=['Name', 'Sexe'], method='NR', top_var_NA=True, verbose=0)
        self.assertIn('top_NA_Name', df_fill_all_cat.columns.tolist())
        self.assertIn('top_NA_Sexe', df_fill_all_cat.columns.tolist())
        self.assertEqual(df_fill_all_cat.iloc[3]['Name'], 'NR')
        self.assertEqual(df_fill_all_cat.iloc[3]['Sexe'], 'NR')


class Test_Date_preprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(data)
        self.df_to_date = all_to_date(self.df, ['Date_nai', 'American_date_nai'], verbose=0)
        self.df_to_anc = date_to_anc(self.df_to_date, var_list=['American_date_nai', 'Date_nai'], date_ref='27/10/2010',
                                     verbose=0)

    def test_all_to_date(self):
        #self.assertEqual(np.dtype(self.df_to_date['American_date_nai']), 'datetime64[ns]')
        self.assertEqual(np.dtype(self.df_to_date['Date_nai']), 'datetime64[ns]')

    def date_to_anc(self):
        self.assertIn('anc_American_date_nai', self.df_to_anc.columns)
        self.assertIn('anc_Date_nai', self.df_to_anc.columns)
        self.assertNotIn('Date_nai', self.df_to_anc.columns)
        self.assertNotIn('American_Date_nai', self.df_to_anc.columns)
        self.assertEqual(self.df_to_anc['anc_date_nai'][0], 0.0)


class Test_Categorical(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(data)
        self.df_dummy = dummy_all_var(self.df, var_list=['Eyes', 'Sexe'], prefix_list=None, keep=False, verbose=0)
        self.df_dummy_pref = dummy_all_var(self.df, var_list=['Eyes', 'Sexe'], prefix_list=['Ey', 'Sx'], keep=True,
                                           verbose=0)

    def test_dummy_all_var(self):
        self.assertIn('Eyes_blue', self.df_dummy.columns)
        self.assertIn('Eyes_red', self.df_dummy.columns)
        self.assertNotIn('Eyes', self.df_dummy.columns)
        self.assertNotIn('Sexe', self.df_dummy.columns)
        self.assertEqual(self.df_dummy['Eyes_blue'].tolist(), [1, 0, 0, 1])
        self.assertEqual(self.df_dummy['Sexe_M'].tolist(), [1, 1, 1, 0])
        self.assertIn('Ey_blue', self.df_dummy_pref.columns)
        self.assertIn('Sx_M', self.df_dummy_pref.columns)
        self.assertIn('Eyes', self.df_dummy_pref.columns)
        self.assertIn('Sexe', self.df_dummy_pref.columns)


class Test_Outliers(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(data)
        self.df_perc_bind, _ = remove_bind(self.df, 'Hair', method='percent', threshold=0.30)
        self.df_nb_bind, _ = remove_bind(self.df, 'Hair', method='number', threshold=1)
        self.df_nbvar_bind, _ = remove_bind(self.df, 'Hair', method='nbvar', threshold=2)
        self.df_process_cat = process_cat_outliers(self.df, ['Name', 'Hair'], method="percent", threshold=0.30,
                                                   verbose=0)

    def test_remove_bind(self):
        self.assertEqual(self.df_perc_bind['Hair'].tolist(), ['brown', 'brown', 'brown', 'other'])
        self.assertEqual(self.df_nb_bind['Hair'].tolist(), ['brown', 'brown', 'brown', 'other'])
        self.assertEqual(self.df_nbvar_bind['Hair'].tolist(), ['brown', 'brown', 'brown', 'other'])

    def test_process_cat_outliers(self):
        self.assertEqual(self.df_process_cat['Name'].tolist(), ['other', 'other', 'other', 'other'])
        self.assertEqual(self.df_process_cat['Hair'].tolist(), ['brown', 'brown', 'brown', 'other'])

    def test_process_num_outliers(self):
        df_outlier = pd.DataFrame({'Out_1': np.ones(1000).tolist() + [1000], 'Out_2': np.ones(1000).tolist() + [-1000]})
        df_outlier_proc = process_num_outliers(df_outlier, None, xstd=3, verbose=0)
        self.assertEqual(df_outlier_proc['Out_1'][1000], 96.67678469055576)
        self.assertEqual(df_outlier_proc['Out_2'][1000], -94.86832980505137)
