from lib.Load.Load import load_data
from lib.autoML import AutoML
from lib.Utils.Utils import parse_target

#########################
# Load and process target
#########################
# data/bank-additional-full.csv
# lib/tests/df_test.csv
df_raw = load_data('lib/tests/df_test.csv')
trgt = 'y'

#df_raw, target = parse_target(df_raw,trgt,modalite='yes')

############################
# instantiate AutoML object
############################
#print(df_raw.columns)

df = AutoML(df_raw.copy(),target = 'y_yes')

df.audit(verbose=1)

df.get_outliers(verbose=1)

df.preprocess(process_outliers=True, verbose=1)

#df.train_predict(verbose=1)
