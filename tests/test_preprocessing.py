"""Tests pour la zone Preprocessing."""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import importlib.util
from types import ModuleType
from datetime import datetime

# Prevent AutoMxL __init__.py from loading (it imports torch)
sys.modules['AutoMxL'] = ModuleType('AutoMxL')
sys.modules['AutoMxL.Utils'] = ModuleType('AutoMxL.Utils')
sys.modules['AutoMxL.Explore'] = ModuleType('AutoMxL.Explore')
sys.modules['AutoMxL.Preprocessing'] = ModuleType('AutoMxL.Preprocessing')


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base_dir = os.path.join(os.path.dirname(__file__), '..')

# Load dependencies
decorators = load_module_from_path(
    'AutoMxL.Utils.Decorators',
    os.path.join(base_dir, 'AutoMxL', 'Utils', 'Decorators.py')
)
display = load_module_from_path(
    'AutoMxL.Utils.Display',
    os.path.join(base_dir, 'AutoMxL', 'Utils', 'Display.py')
)
features_type = load_module_from_path(
    'AutoMxL.Explore.Features_Type',
    os.path.join(base_dir, 'AutoMxL', 'Explore', 'Features_Type.py')
)

# Load Preprocessing modules (skip Deep_Encoder and Categorical - they need torch)
date_module = load_module_from_path(
    'AutoMxL.Preprocessing.Date',
    os.path.join(base_dir, 'AutoMxL', 'Preprocessing', 'Date.py')
)
missing_values_module = load_module_from_path(
    'AutoMxL.Preprocessing.Missing_Values',
    os.path.join(base_dir, 'AutoMxL', 'Preprocessing', 'Missing_Values.py')
)
outliers_module = load_module_from_path(
    'AutoMxL.Preprocessing.Outliers',
    os.path.join(base_dir, 'AutoMxL', 'Preprocessing', 'Outliers.py')
)

DateEncoder = date_module.DateEncoder
date_to_anc = date_module.date_to_anc
NAEncoder = missing_values_module.NAEncoder
fill_numerical = missing_values_module.fill_numerical
fill_categorical = missing_values_module.fill_categorical
OutliersEncoder = outliers_module.OutliersEncoder
replace_extreme_values = outliers_module.replace_extreme_values
get_cat_outliers = outliers_module.get_cat_outliers


# =============================================================================
# Tests DateEncoder
# =============================================================================

def test_date_encoder_converts_to_anciennete():
    """DateEncoder convertit les dates en années et renomme avec préfixe anc_."""
    df = pd.DataFrame({
        'date_creation': pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']),
        'other': [1, 2, 3]
    })
    date_ref = datetime(2025, 1, 1)

    result, new_names = date_to_anc(df, l_var=['date_creation'], date_ref=date_ref)

    assert 'anc_date_creation' in result.columns
    assert 'date_creation' not in result.columns
    # 2020-01-01 → ~5 ans d'ancienneté
    assert result['anc_date_creation'].iloc[0] == pytest.approx(5.0, abs=0.1)


# =============================================================================
# Tests NAEncoder
# =============================================================================

def test_na_encoder_fills_numerical_with_mean():
    """NAEncoder remplit les NA numériques avec la moyenne."""
    df = pd.DataFrame({
        'num': [1.0, 2.0, 3.0, np.nan, 5.0]
    })

    result = fill_numerical(df, l_var=['num'], method='mean', track_num_NA=False)

    assert result['num'].isna().sum() == 0
    # mean of [1, 2, 3, 5] = 2.75
    assert result['num'].iloc[3] == pytest.approx(2.75, abs=0.01)


def test_na_encoder_fills_categorical_with_NR():
    """NAEncoder remplit les NA catégoriels avec 'NR'."""
    df = pd.DataFrame({
        'cat': ['A', 'B', None, 'A']
    })

    result = fill_categorical(df, l_var=['cat'], method='NR')

    assert result['cat'].isna().sum() == 0
    assert result['cat'].iloc[2] == 'NR'


def test_na_encoder_tracks_na_columns():
    """NAEncoder crée les colonnes top_NA_* quand track_num_NA=True."""
    df = pd.DataFrame({
        'num': [1.0, np.nan, 3.0]
    })

    result = fill_numerical(df, l_var=['num'], method='mean', track_num_NA=True)

    assert 'top_NA_num' in result.columns
    assert result['top_NA_num'].iloc[0] == 0  # pas de NA
    assert result['top_NA_num'].iloc[1] == 1  # était NA


# =============================================================================
# Tests OutliersEncoder
# =============================================================================

def test_outliers_encoder_caps_extreme_values():
    """OutliersEncoder cap les valeurs numériques aux bornes."""
    df = pd.DataFrame({
        'num': [1.0, 2.0, 3.0, 100.0]  # 100 est extrême
    })

    result = replace_extreme_values(df, 'num', lower_th=0.0, upper_th=10.0)

    assert result['num'].iloc[3] == 10.0  # cappé à 10


def test_outliers_encoder_aggregates_rare_categories():
    """OutliersEncoder identifie les catégories rares."""
    df = pd.DataFrame({
        'cat': ['A'] * 50 + ['B'] * 50 + ['rare1', 'rare2']  # rare1, rare2 < 2%
    })

    outliers = get_cat_outliers(df, l_var=['cat'], threshold=0.05)

    assert 'cat' in outliers
    assert 'rare1' in outliers['cat']
    assert 'rare2' in outliers['cat']
    assert 'A' not in outliers['cat']
