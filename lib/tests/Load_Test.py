import os
os.chdir('/home/mxla@comptes.racine.local/automl/Iteration3')

from lib.Load.Load import *
import unittest
import pandas as pd

path=''
file='df_test.csv'

class Test_Load(unittest.TestCase):
    
    #
    def test_detectdelimiter(self):
        self.assertEqual(detectDelimiter(file),',')
        
    def test_load(self):
        self.assertIsNotNone(type(load_data(path='',file=file)),pd.DataFrame)