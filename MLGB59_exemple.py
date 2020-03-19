from MLBG59 import *
from MLBG59.Utils.Display import print_dict

# Import data
df_raw = import_data('data/bank-additional-full.csv', verbose=True)
df, target = category_to_target(df_raw, "y", "yes")

"""
# encode target
new_df, target = category_to_target(df_raw, var=df, cat=target)

# instantiate AutoML object
auto_df = AutoML(new_df.copy(), target=target)

# explore data
auto_df.explore(verbose=True)
print_dict(auto_df.d_features)
print(help(AutoML.explore))

# data preparation
auto_df.preprocess(process_outliers=True, cat_method='deep_encoder', verbose=False)
print_dict(auto_df.d_preprocess)
# features selection
auto_df.select_features(method='pca', verbose=False)


# random search
res_dict, l_valid_models, best_model_index, df_model_res = auto_df.train_predict(clf='XGBOOST', n_comb=20,
                                                                            comb_seed=None, verbose=True)
"""
