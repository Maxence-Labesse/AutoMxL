from MLKit.Load.Load import load_data
from MLKit.MLKit import AutoML
from MLKit.Utils.Utils import parse_target

#########################
# Load and process target
#########################
# data/bank-additional-full.csv
# MLKit/tests/df_test.csv
df_raw = load_data('data/bank-additional-full.csv')
trgt = 'y'

df_raw, target = parse_target(df_raw,trgt,modalite='yes')

############################
# instantiate MLKit object
############################
#print(df_raw.columns)

df = AutoML(df_raw.copy(),target = 'y_yes')

df.audit(verbose=1)

df.get_outliers(verbose=1)

df.preprocess(process_outliers=True, verbose=1)

df.train_predict(verbose=1)