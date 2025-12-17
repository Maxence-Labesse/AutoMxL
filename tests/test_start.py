import pytest
import pandas as pd
import tempfile
import os

from AutoMxL.Start.Load import import_data
from AutoMxL.Start.Encode_Target import category_to_target, range_to_target


class TestImportData:

    def test_import_data_loads_csv_as_dataframe(self):
        """Charge un CSV et retourne un DataFrame avec les bonnes dimensions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('col1,col2,col3\n')
            f.write('a,1,x\n')
            f.write('b,2,y\n')
            f.write('c,3,z\n')
            temp_path = f.name

        try:
            df = import_data(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (3, 3)
            assert list(df.columns) == ['col1', 'col2', 'col3']
        finally:
            os.unlink(temp_path)

    def test_import_data_unsupported_format_returns_none(self):
        """Format non supporté retourne None."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.unsupported', delete=False) as f:
            f.write('some content')
            temp_path = f.name

        try:
            df = import_data(temp_path)
            assert df is None
        finally:
            os.unlink(temp_path)


class TestCategoryToTarget:

    def test_category_to_target_creates_binary_target(self):
        """Transforme une catégorie en target 0/1 et supprime la source."""
        df = pd.DataFrame({
            'status': ['yes', 'no', 'yes', 'no', 'yes'],
            'value': [1, 2, 3, 4, 5]
        })

        df_result, target_name = category_to_target(df, var='status', cat='yes')

        assert target_name == 'status_yes'
        assert 'status' not in df_result.columns
        assert target_name in df_result.columns
        assert list(df_result[target_name]) == [1, 0, 1, 0, 1]


class TestRangeToTarget:

    def test_range_to_target_creates_binary_target(self):
        """Transforme une plage numérique en target 0/1 et supprime la source."""
        df = pd.DataFrame({
            'age': [5, 15, 25, 35, 45],
            'name': ['a', 'b', 'c', 'd', 'e']
        })

        df_result, target_name = range_to_target(df, var='age', min=10, max=30)

        assert 'age' not in df_result.columns
        assert target_name in df_result.columns
        assert list(df_result[target_name]) == [0, 1, 1, 0, 0]

    def test_range_to_target_with_partial_bounds(self):
        """Fonctionne avec seulement min (sans max)."""
        df = pd.DataFrame({
            'score': [5, 10, 15, 20, 25],
            'id': [1, 2, 3, 4, 5]
        })

        df_result, target_name = range_to_target(df, var='score', min=15)

        assert 'score' not in df_result.columns
        assert list(df_result[target_name]) == [0, 0, 1, 1, 1]
