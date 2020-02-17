from MLBG59 import *

#########################
# Start and process target
#########################
# data/bank-additional-full.csv
# MLBG59/tests/df_test.csv
df_raw = import_data('data/bank-additional-full.csv', verbose=0)
trgt = 'y'

df_raw, target = category_to_target(df_raw,trgt,cat='yes')

#help(load_data)
print(import_data.__doc__)

############################
# instantiate MLBG59 object
############################
#print(df_raw.columns)

#df = AutoML(df_raw.copy(),target = 'y_yes')
#df.audit(verbose=1)

"""
df.get_outliers(verbose=1)

df.preprocess(process_outliers=True, verbose=1)

df.train_predict(verbose=1)
"""