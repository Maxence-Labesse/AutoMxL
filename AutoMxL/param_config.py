"""
Configuration des paramètres par défaut pour AutoMxL.

Ce module centralise les hyperparamètres par défaut utilisés par :
- Deep Encoder (Preprocessing) : batch_size, n_epoch, learning_rate
- Bagging (Modelisation) : n_sample, pos_sample_size, replace
- Random Search (Modelisation) : grilles d'hyperparamètres RF et XGBoost
"""
import numpy as np

batch_size = 124
n_epoch = 20
learning_rate = 0.001
crit = 'MSE'
optim = 'Adam'

default_bagging_param = {'n_sample': 5,
                         'pos_sample_size': 1.0,
                         'replace': False}

default_RF_grid_param = {
    'n_estimators': np.random.uniform(low=20, high=500, size=40).astype(int),
    'max_features': ['auto', 'log2'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [5, 10, 15, 20]}

default_XGB_grid_param = {
    'n_estimators': np.random.uniform(low=100, high=300, size=40).astype(int),
    'max_features': ['auto', 'log2'],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'learning_rate': [0.0001, 0.0003, 0.0006, 0.0009, 0.001, 0.003, 0.006, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.3,
                      0.6],
    'scale_pos_weight': [2, 3, 4, 5, 6, 7, 8, 9]}
