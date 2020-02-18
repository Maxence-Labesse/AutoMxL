from MLBG59 import *
from MLBG59.Utils.Utils import print_dict

#########################
# Start and process target
#########################
# data/bank-additional-full.csv
# MLBG59/tests/df_test.csv
df_raw = import_data('data/bank-additional-full.csv', verbose=True)
trgt = 'y'

df_raw, target = category_to_target(df_raw, trgt, cat='yes')

# help(load_data)
#print(import_data.__doc__)

print(import_data.__globals__)

print(import_data.__code__.co_varnames[:import_data.__code__.co_argcount])

"""
############################
# instantiate MLBG59 object
############################
# print(df_raw.columns)

df = AutoML(df_raw.copy(), target='y_yes')

df.recap(verbose=True)

print_dict(df.d_features)

df.get_outliers(verbose=True)

print_dict(df.d_num_outliers)
print_dict(df.d_cat_outliers)

df.preprocess(process_outliers=True, verbose=True)

_, _, _, df_test = df.train_predict(n_comb=10, comb_seed=None, verbose=True)

print(df_test.TOP_feat1)
print(df_test.TOP_feat2)
"""
