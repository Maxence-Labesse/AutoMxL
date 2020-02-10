from MLBG59 import *

#########################
# Load and process target
#########################
# data/bank-additional-full.csv
# MLBG59/tests/df_test.csv
df_raw = load_data('data/bank-additional-full.csv', verbose=0)
trgt = 'y'

df_raw, target = parse_target(df_raw,trgt,modalite='yes')

#help(load_data)
print(load_data.__doc__)

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