from MLBG59 import *
from MLBG59.Utils.Display import print_dict

# Import data
#df_raw = import_data('data/bank-additional-full.csv', verbose=False)
#df, target = category_to_target(df_raw, "y", "yes")

df_raw = import_data('data/covtype.csv', verbose=False)
df, target = category_to_target(df_raw, "target", 2)
print(df.shape)
"""
df.rename(columns={'y_yes': 'target'})
target= 'target'
print(df.columns.tolist())
"""

# instantiate AutoML object
auto_df = AML(df.copy(), target=target)

# explore data
auto_df.explore(verbose=False)
# print_dict(auto_df.d_features)

"""
# data preparation
auto_df.preprocess(process_outliers=True, cat_method='deep_encoder', verbose=True)
print('\n\n')
print_dict(auto_df.d_preprocess)

# features selection
auto_df.select_features(method='pca', verbose=False)

# random search
res_dict, l_valid_models, best_model_index, df_model_res = auto_df.train_model(clf='XGBOOST', n_comb=20,
                                                                               comb_seed=None, verbose=True)
"""
