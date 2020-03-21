import pandas as pd
import unittest
from MLBG59.Start.Encode_Target import range_to_target

# import données
df = pd.read_csv('tests/df_test_bis.csv')
raw_target = 'Height'


# Classe héritée de unittest.TestCase
class Test_Range_To_Target(unittest.TestCase):
    """
    On test la fonction de Lucile
    """

    def test_features(self):
        """
        On test le bon remplacement de la variable cible
        """
        df_range_target, new_var = range_to_target(df, var=raw_target, min=180, max=185, verbose=False)

        # nouvelle target dans le nouveau dataset
        self.assertIn(new_var, df_range_target.columns.tolist())
        # ancienne target n'est plus dans le nouveau dataset
        self.assertNotIn(raw_target, df_range_target.columns.tolist())

    def test_values(self):
        """
        Test des valeurs de la variable nouvelle variable cible
        """
        # si lower et upper renseignés
        df_range_target, new_var = range_to_target(df, var=raw_target, min=180, max=185, verbose=False)
        self.assertEqual(df_range_target[new_var].tolist(), [0, 0, 1, 0, 1, 1])
        # si lower renseigné uniquement
        df_range_target, new_var = range_to_target(df, var=raw_target, min=180, verbose=False)
        self.assertEqual(df_range_target[new_var].tolist(), [0, 0, 1, 1, 1, 1])
        # si upper renseigné uniquement
        df_range_target, new_var = range_to_target(df, var=raw_target, max=185, verbose=False)
        self.assertEqual(df_range_target[new_var].tolist(), [0, 1, 1, 0, 1, 1])