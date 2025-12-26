"""Tests pour la zone Explore."""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import importlib.util
from types import ModuleType

# Prevent AutoMxL __init__.py from loading (it imports torch)
sys.modules['AutoMxL'] = ModuleType('AutoMxL')
sys.modules['AutoMxL.Utils'] = ModuleType('AutoMxL.Utils')
sys.modules['AutoMxL.Explore'] = ModuleType('AutoMxL.Explore')

# Load modules directly without going through AutoMxL package __init__.py
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Get the base directory
base_dir = os.path.join(os.path.dirname(__file__), '..')

# Load Decorators first
decorators = load_module_from_path(
    'AutoMxL.Utils.Decorators',
    os.path.join(base_dir, 'AutoMxL', 'Utils', 'Decorators.py')
)

# Load Display
display = load_module_from_path(
    'AutoMxL.Utils.Display',
    os.path.join(base_dir, 'AutoMxL', 'Utils', 'Display.py')
)

# Load Features_Type
features_type = load_module_from_path(
    'AutoMxL.Explore.Features_Type',
    os.path.join(base_dir, 'AutoMxL', 'Explore', 'Features_Type.py')
)
is_date = features_type.is_date
is_boolean = features_type.is_boolean
is_categorical = features_type.is_categorical
is_identifier = features_type.is_identifier
is_verbatim = features_type.is_verbatim
features_from_type = features_type.features_from_type

# Load Explore module
explore_mod = load_module_from_path(
    'AutoMxL.Explore.Explore',
    os.path.join(base_dir, 'AutoMxL', 'Explore', 'Explore.py')
)
get_features_type = explore_mod.get_features_type
low_variance_features = explore_mod.low_variance_features


@pytest.fixture
def sample_df():
    """DataFrame avec différents types de colonnes."""
    return pd.DataFrame({
        'id': ['A001', 'A002', 'A003', 'A004', 'A005'],
        'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01']),
        'boolean': [True, False, True, False, True],
        'category': ['cat1', 'cat2', 'cat1', 'cat2', 'cat1'],
        'numeric': [1.5, 2.3, 3.1, 4.2, 5.0],
        'constant': [1, 1, 1, 1, 1]
    })


def test_get_features_type_returns_all_keys(sample_df):
    """get_features_type() retourne un dictionnaire avec toutes les clés de types."""
    result = get_features_type(sample_df)

    expected_keys = {'date', 'identifier', 'verbatim', 'boolean', 'categorical', 'numerical'}
    assert set(result.keys()) == expected_keys


def test_is_date_detects_datetime_column(sample_df):
    """is_date() détecte correctement une colonne datetime."""
    assert is_date(sample_df, 'date') is True
    assert is_date(sample_df, 'numeric') is False


def test_is_boolean_detects_binary_column():
    """is_boolean() détecte une colonne avec exactement 2 valeurs."""
    df = pd.DataFrame({
        'binary': [0, 1, 0, 1, 0, 1],
        'ternary': [0, 1, 2, 0, 1, 2]
    })
    assert is_boolean(df, 'binary') is True
    assert is_boolean(df, 'ternary') is False


def test_is_categorical_detects_low_cardinality():
    """is_categorical() détecte les colonnes à faible cardinalité."""
    df = pd.DataFrame({
        'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
        'unique': [f'val_{i}' for i in range(60)]
    })
    assert is_categorical(df, 'category') is True
    assert is_categorical(df, 'unique') is False


def test_low_variance_features_detects_constant():
    """low_variance_features() détecte les colonnes constantes."""
    df = pd.DataFrame({
        'constant': [1, 1, 1, 1, 1],
        'variable': [1, 2, 3, 4, 5]
    })
    result = low_variance_features(df, threshold=0, rescale=True)

    assert 'constant' in result.index
    assert 'variable' not in result.index
