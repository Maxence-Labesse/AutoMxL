from MLBG59 import *
from MLBG59.Utils.Display import print_dict, color_print, print_title1
from MLBG59.Modelisation.Utils import classifier_evaluate

# Import data
df_raw = import_data('data/bank-additional-full.csv', verbose=False)
df, target = category_to_target(df_raw, "y", "yes")

# df_raw = import_data('data/covtype.csv', verbose=False)
# df, target = category_to_target(df_raw, "target", 2)
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


# data preparation
auto_df.preprocess(process_outliers=True, cat_method='one_hot', verbose=False)
print('\n\n')
print_dict(auto_df.d_preprocess)

# features selection
auto_df.select_features(method='pca', verbose=False)

# random search
#res_dict, l_valid_models, best_model_index, df_model_res = auto_df.train_model(clf='XGBOOST', n_comb=2,
#                                                                               comb_seed=None, verbose=True)


auto_df.train(clf='RF', n_comb=2, comb_seed=None, verbose=True)


# dev
print_title1("Apply")
df_apply = auto_df.preprocess_apply(df, verbose=False)
df_apply = auto_df.select_features_apply(df_apply, verbose=False)

res_dict, l_valid_models, best_model_index, df_model_res = auto_df.predict(df_apply, metric='F1', verbose=True)

# test
y = df_apply[target]
X = df_apply.drop(target, axis=1)

y_proba = res_dict[best_model_index]['model'].predict_proba(X)[:, 1]
y_pred = res_dict[best_model_index]['model'].predict(X)

eval_dict = classifier_evaluate(y, y_pred, y_proba, verbose=0)

color_print("eval_dict")
print_dict(eval_dict)