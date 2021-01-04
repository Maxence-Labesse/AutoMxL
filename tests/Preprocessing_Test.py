from AutoMxL.Preprocessing.Categorical import *
from AutoMxL.Preprocessing.Date import *
from AutoMxL.Preprocessing.Outliers import *
from AutoMxL.Preprocessing.Missing_Values import *
import unittest
import pandas as pd
import math

# test config
df = pd.read_csv('tests/df_test_bis.csv')


class TestMissingValues(unittest.TestCase):
    """
    Test Missing_Values module
    """

    def test_fill_numerical(self):
        """ fill_numerical function"""
        df_fill_all_num = fill_numerical(df, ['Age', 'Height'], method='zero', track_num_NA=True, verbose=False)
        self.assertIn('top_NA_Height', df_fill_all_num.columns.tolist())
        self.assertIn('top_NA_Age', df_fill_all_num.columns.tolist())
        self.assertEqual(df_fill_all_num.iloc[0]['Height'], 0)
        self.assertEqual(df_fill_all_num.iloc[1]['Age'], 0)

    def test_fill_categorical(self):
        """ fill_categorical function"""
        df_fill_all_cat = fill_categorical(df, l_var=['Name', 'Sexe'], method='NR', verbose=False)
        self.assertEqual(df_fill_all_cat.iloc[3]['Name'], 'NR')
        self.assertEqual(df_fill_all_cat.iloc[3]['Sexe'], 'NR')

    def test_NAEncoder(self):
        """ NAEncoder class"""
        NA_encoder1 = NAEncoder(replace_num_with='median', replace_cat_with='NR', track_num_NA=True)
        NA_encoder1.fit(df, l_var=['Name', 'Age'])
        df_NA1 = NA_encoder1.transform(df)
        #
        NA_encoder2 = NAEncoder(replace_num_with='zero', replace_cat_with='NR', track_num_NA=False)
        df_NA2 = NA_encoder2.fit_transform(df)

        # created features
        self.assertIn("top_NA_Age", df_NA1.columns.tolist())
        self.assertNotIn('top_NA_Height', df_NA1.columns.tolist())
        self.assertNotIn("top_NA_Age", df_NA2.columns.tolist())
        # raw features contain NA
        self.assertEqual(get_NA_features(df),
                         ['Name', 'Id_cat', 'Id_num', 'Verb', 'Age', 'Height', 'Sexe', 'Date_nai', 'American_date_nai'])
        # filled features
        self.assertEqual(get_NA_features(df_NA1),
                         ['Id_cat', 'Id_num', 'Verb', 'Height', 'Sexe', 'Date_nai', 'American_date_nai'])
        self.assertEqual(get_NA_features(df_NA2), [])
        # modified values
        self.assertTrue(math.isnan(df['Name'][3]))
        self.assertTrue(math.isnan(df['Age'][1]))
        self.assertEqual(df_NA1['Name'][3], 'NR')
        self.assertEqual(df_NA1['Age'][1], 25.5)
        self.assertEqual(df_NA2['Name'][3], 'NR')
        self.assertEqual(df_NA2['Age'][1], 0)


"""
------------------------------------------------------------------------------------------------
"""

df_to_date = all_to_date(df, ['Date_nai', 'American_date_nai'], verbose=False)
df_to_anc, new_var_list = date_to_anc(df_to_date, l_var=['American_date_nai', 'Date_nai'], date_ref='27/10/2010')


class TestDate(unittest.TestCase):
    """
    Test Date Module
    """

    def test_all_to_date(self):
        """ all_to_date function """
        self.assertEqual(np.dtype(df_to_date['American_date_nai']), 'datetime64[ns]')
        self.assertEqual(np.dtype(df_to_date['Date_nai']), 'datetime64[ns]')
        self.assertEqual(np.dtype(df_to_date['American_date_nai']), 'datetime64[ns]')

    def test_date_to_anc(self):
        """ date_to_anc function"""
        self.assertIn('anc_American_date_nai', df_to_anc.columns)
        self.assertIn('anc_Date_nai', df_to_anc.columns)
        self.assertNotIn('Date_nai', df_to_anc.columns)
        self.assertNotIn('American_Date_nai', df_to_anc.columns)
        self.assertEqual(df_to_anc['anc_Date_nai'][0], 0.0)
        self.assertIn('anc_American_date_nai', new_var_list)

    def test_DateEncoder(self):
        """ DateEncoder class"""
        Date_encoder1 = DateEncoder(method='timedelta', date_ref='27/10/2010')
        Date_encoder1.fit(df, l_var=['American_date_nai', 'Age'])
        df_date1 = Date_encoder1.transform(df)
        #
        date_encoder2 = DateEncoder(method='timedelta', date_ref='27/10/2011')
        df_date2 = date_encoder2.fit_transform(df)

        # created/removed features
        self.assertIn('anc_American_date_nai', df_date1.columns.tolist())
        self.assertIn('Date_nai', df_date1.columns.tolist())
        self.assertNotIn('anc_Date_nai', df_date1.columns.tolist())
        self.assertIn('Age', df_date1.columns.tolist())
        self.assertNotIn('American_date_nai', df_date2.columns.tolist())
        self.assertNotIn('Date_nai', df_date2.columns.tolist())
        # features formats
        self.assertEqual(df_date1['anc_American_date_nai'].dtype, 'float64')
        self.assertEqual(df_date2['anc_Date_nai'].dtype, 'float64')
        # features values
        self.assertEqual(df_date1['anc_American_date_nai'][0], 0.0)
        self.assertEqual(df_date2['anc_Date_nai'][0], 1.0)
        self.assertEqual(df_date2['anc_American_date_nai'][0], 1.0)


"""
------------------------------------------------------------------------------------------------
"""


class TestCategorical(unittest.TestCase):
    """
    Test Categorical module
    """

    def test_dummy_all_var(self):
        """ dummy_all_var func """
        df_dummy = dummy_all_var(df, var_list=['Eyes', 'Sexe'], prefix_list=None, keep=False, verbose=False)
        df_dummy_pref = dummy_all_var(df, var_list=['Eyes', 'Sexe'], prefix_list=['Ey', 'Sx'], keep=True,
                                      verbose=False)
        # created/removed features
        self.assertIn('Eyes_blue', df_dummy.columns)
        self.assertIn('Eyes_red', df_dummy.columns)
        self.assertNotIn('Eyes', df_dummy.columns)
        self.assertNotIn('Sexe', df_dummy.columns)
        self.assertIn('Ey_blue', df_dummy_pref.columns)
        self.assertIn('Sx_M', df_dummy_pref.columns)
        self.assertIn('Eyes', df_dummy_pref.columns)
        self.assertIn('Sexe', df_dummy_pref.columns)
        # features values
        self.assertEqual(df_dummy['Eyes_blue'].tolist(), [1, 0, 0, 1, 0, 1])
        self.assertEqual(df_dummy['Sexe_M'].tolist(), [1, 1, 1, 0, 0, 1])

    def test_CategoricalEncoder(self):
        """ CategoricalEncoder """
        df_pred = pd.read_csv('tests/df_test.Csv')
        df_pred['job'] = df_pred['job'].fillna('NR')

        cat_encoder1 = CategoricalEncoder(method='deep_encoder')
        df_cat1 = cat_encoder1.fit_transform(df_pred, target='y_yes', l_var=['job', 'education'], verbose=False)
        print('\n\n')
        #
        cat_encoder2 = CategoricalEncoder(method='one_hot')
        cat_encoder2.fit(df, l_var=['Name', 'Eyes'], verbose=False)
        print('\n\n')
        df_cat2 = cat_encoder2.transform(df)

        # features created/removed
        self.assertIn('job_0', df_cat1.columns.tolist())
        self.assertIn('education_0', df_cat1.columns.tolist())
        self.assertNotIn('job', df_cat1.columns.tolist())
        self.assertIn('Eyes_blue', df_cat2.columns.tolist())
        self.assertNotIn('Eyes', df_cat2.columns.tolist())

        # features embedding
        self.assertEqual(list(cat_encoder1.d_embeddings.keys()), ['job', 'education'])
        # features values
        self.assertEqual(df_cat2['Eyes_green'].tolist(), [0, 0, 0, 0, 1, 0])


"""
------------------------------------------------------------------------------------------------
"""


class TestOutliers(unittest.TestCase):

    def test_replace_category(self):
        """ replace_category function """
        df_process_cat = replace_category(df, 'Hair', ['blond'], verbose=False)
        df_process_cat = replace_category(df_process_cat, 'Name', ['Tom', 'Nick'], verbose=False)

        # features values
        self.assertEqual(df_process_cat['Name'].tolist(),
                         ['outliers', 'outliers', 'Krish', np.nan, 'John', 'Jack'])
        self.assertEqual(df_process_cat['Hair'].tolist(),
                         ['brown', 'brown', 'dark', 'outliers', 'outliers', 'outliers'])

    def test_extrem_values(self):
        """ extreme_values function """
        df_outlier_proc = replace_extreme_values(df, 'Height', 175, 185)
        self.assertEqual(df_outlier_proc['Height'].tolist()[1:], [175.0, 180.0, 185.0, 185.0, 185.0])

    def test_OutlierEncode(self):
        """ OutlierEncodeR class """
        out_encoder1 = OutliersEncoder(cat_threshold=0.25, num_xstd=1)
        out_encoder1.fit(df, l_var=['Height', 'Sexe', 'Hair', 'Age'], verbose=False)
        df_out1 = out_encoder1.transform(df, verbose=False)
        out_encoder2 = OutliersEncoder(cat_threshold=0.2, num_xstd=1)
        df_out2 = out_encoder2.fit_transform(df, verbose=False)

        # cat outliers
        self.assertEqual(list(df_out1['Hair']), ['brown', 'brown', 'dark', 'blond', 'blond', 'blond'])
        self.assertEqual(list(df_out1['Sexe']), ['M', 'M', 'M', np.nan, 'F', 'M'])
        self.assertEqual(list(df_out2['Name']), ['outliers'] * 6)
        # num outliers
        self.assertEqual(list(df_out1['Height'].round(4))[1:], [175.2177, 180.0, 188.7823, 185.0, 185.0])
        self.assertEqual(list(df_out2['Unnamed: 0'].round(4)), [0.7922, 1.0, 2.0, 3.0, 4.0, 4.2078])
        print('out2')
