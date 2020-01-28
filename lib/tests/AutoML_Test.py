from lib.autoML import *
import unittest
import pandas as pd
import numpy as np


df_test = pd.read_csv('~/automl/Iteration3/df_test.csv')



class TestInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Init"""
        cls.df = autoML(df_test.copy(), target='y_yes')
    
    # Test autoML object instantiation from DataFrame
    def test_df_is_not_none(self):
        self.assertIsNotNone(self.df)

    # Test target Attribute
    def test_target_is_not_none(self):
        self.assertIsNotNone(self.df.target)

"""
-------------------------------------------------------------------------------------------------------------------------
""" 

class TestAudit(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Instantiate autoML object from df_test and target
        cls.df = autoML(df_test.copy(), target='y_yes')
        # Audit 
        cls.df.audit(print_level=0)
        
    def test_num_columns(self):
        # numerical features identification
        self.assertEqual(self.df.num_columns,['age','euribor3m','null_var','y_yes'])
    
    def test_cat_columns(self):
        # categorical features identification
        self.assertEqual(self.df.cat_columns,['job','education'])
        
    def test_date_columns(self):
        # date features identification
        self.assertEqual(self.df.date_columns,['date_1','date_2'])

    def test_NA_columns(self):
        # features containing NA values identification
        self.assertEqual(self.df.NA_columns,['job','age','date_1'])
        
    def test_low_var_columns(self):
        # null variance features identification
        self.assertEqual(self.df.low_var_columns,['null_var'])

        
"""
-------------------------------------------------------------------------------------------------------------------------
"""

class TestGet_outliers(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # instantiate autoML object
        cls.df = autoML(df_test.copy(), target='y_yes')
        # audit
        cls.df.audit(print_level=0)
        # get outliers
        cls.df.get_outliers(verbose=0)
    
    #
    def test_cat_outliers_columns(self):
        # categorical features containing outliers identification
        self.assertEqual(self.df.cat_outliers_columns,['job'])
    
    #
    def test_num_outliers_columns(self):
        # numerical features containing outliers identification
        self.assertEqual(self.df.num_outliers_columns,['age'])  

"""
-------------------------------------------------------------------------------------------------------------------------
"""

class TestPreprocess(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # instantiate autoML object
        cls.df = autoML(df_test.copy(), target='y_yes')
        # audit
        cls.df.audit(print_level=0)
        # get outliers
        cls.df.get_outliers(verbose=0)
        # clean and preprocess
        cls.df.preprocess(date_ref=df_test['date_1'][0], process_outliers = True, verbose=0)
    
    def test_remove_low_var(self):
        # low variance features removing
        self.assertNotIn('null_var',self.df.columns)
        
    def test_transform_date(self):
        # anc column created in df
        self.assertIn('anc_date_1',self.df.columns)
        # anc column value
        self.assertEqual(self.df['anc_date_1'][0],0)
        
    def test_fill_na_values(self):
        # 0 na values un df columns
        self.assertTrue((self.df.isna().sum().tolist() == np.zeros(self.df.shape[1])).all())

    """
    def test_replace_outliers(self):
    """
    
    def test_one_hot_encoding(self):
        # feature_modalitie created in df
        self.assertIn('job_management',self.df.columns)
        self.assertIn('education_high.school',self.df.columns)
        # values of one hot encoding
        self.assertEqual(self.df['job_management'].sum(),df_test['job'].value_counts().loc['management'])
        
"""
-------------------------------------------------------------------------------------------------------------------------
"""
        
class TestTrain_Predict(unittest.TestCase) :
    
    @classmethod
    def setUpClass(cls):
        # instantiate autoML object
        cls.df = autoML(df_test.copy(), target='y_yes')
        # audit
        cls.df.audit(print_level=0)
        # get outliers
        cls.df.get_outliers(verbose=0)
        # preprocess
        cls.df.preprocess(date_ref=df_test['date_1'][0], process_outliers = True, verbose=0)
        # train and predict
        cls.hyperopt, cls.dict_res_model, cls.best_model_idx = cls.df.train_predict(n_comb=5, comb_seed=2, verbose=0)
        
    def test_comb_samples(self):
        # tests HP comb values
        self.assertEqual(self.hyperopt.train_model_dict[0]['HP']['learning_rate'],0.001)
        
    def test_best_model(self):
        if self.best_model_idx!=None :
            # test best model is valid (delta_auc)
            self.assertTrue(self.dict_res_model[self.best_model_idx]['evaluation']['delta_auc']<0.03)
            # best model have the max F1 score among valid models
            for i in range(5) :
                if self.dict_res_model[i]['evaluation']['delta_auc']<0.03 :
                    self.assertTrue(self.dict_res_model[self.best_model_idx]['evaluation']['F1']>=self.dict_res_model[i]['evaluation']['F1'])
    
    
                
                