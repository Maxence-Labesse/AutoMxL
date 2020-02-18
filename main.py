from MLBG59 import *
from MLBG59.Utils.Utils import print_dict

#########################
# Start and process target
#########################
# data/bank-additional-full.csv
# MLBG59/tests/df_test.csv
df_raw = import_data('data/bank-additional-full.csv', verbose=True)
trgt = 'y'

df_raw_bis, target = category_to_target(df_raw, trgt, cat='yes')

print(df_raw[trgt].value_counts()['yes'])
print(df_raw_bis['y_yes'].sum())
print(df_raw[trgt].value_counts()['yes']==df_raw_bis['y_yes'].sum())


df = import_data('tests/df_test.csv', verbose=True)
print(df.columns)
print(df.job.value_counts())

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
