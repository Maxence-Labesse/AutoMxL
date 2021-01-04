# Import
from time import time
from datetime import date
from AutoMxL import *
import pandas as pd
from AutoMxL.Utils.Display import print_title1, print_dict
from AutoMxL.Utils.Utils import random_from_dict
from data.data_config import d_files
from AutoMxL.param_config import n_epoch, learning_rate, batch_size, crit, optim

"""
AutoML test meta algo :
- Apply AutoML on multiple datasets with random tuning
- Store datasets informations, AutoML parameters and modelisation perfs in meta.csv
"""
# Meta config #
###############
d_all_param = dict()
# number of datasets over datasets
n_iter = 1
# number of model iterations for each dataset
model_iter = 5
# date reference for date transformation (timedelta
today_date = date.today().strftime("%d/%m/%Y")

# AutoML param #
################
# outliers processing
d_all_param['outliers'] = [True, False]
# available models
d_all_param['clf'] = ['RF', 'XGBOOST']
# features selection
d_all_param['select_method'] = ['pca', None]
# categorical encoding method
d_all_param['cat_method'] = ['deep_encoder', 'one_hot']
# bagging use for modelisation
d_all_param['bagging'] = [True, False]

start_meta = time()

# Extend existing storing file meta.csv
# df_meta = pd.read_csv('meta.csv', sep=",")

# Create new storing file meta.csv
df_meta = pd.DataFrame(
    columns=['file_name', 'date', 'exec_time', 'nrow', 'ncol', 'nnum', 'ncat', 'ndate', 'cat_method',
             'n_epoch', 'learning_rate', 'batch_size', 'crit', 'optim', 'process_outliers', 'select_method', 'bagging',
             'clf', 'bagging', 'n_model', 'n_valid_models', 'AUC', 'delta_auc', 'precision', 'recall', 'F1'])

#############
# Meta Algo #
#############
for i in range(n_iter):
    for file in [file for file in d_files.keys()]:

        # import
        df_raw = import_data(file, verbose=False)

        # print
        print_title1(file + "  / niter : " + str(i))
        print("\nTaille du dataset brut", df_raw.shape)
        before = time()

        # pick random parameters from d_all_param
        d_param = random_from_dict(d_all_param, verbose=True)

        # encode target
        new_df, target = category_to_target(df_raw, var=d_files[file]['var'], cat=d_files[file]['cat'])

        # instantiate AML object
        auto_df = AML(new_df.copy(), target=target)

        # explore
        auto_df.explore(verbose=False)
        print_dict(auto_df.d_features)

        # preprocess
        auto_df.preprocess(process_outliers=d_param['outliers'], cat_method=d_param['cat_method'], verbose=False)

        # select features
        if d_param['select_method'] is not None:
            auto_df.select_features(method=d_param['select_method'], verbose=False)

        print("Taille du dataset avant modele :", auto_df.shape)

        # random search
        res_dict, l_valid_models, best_model_index, df_test = auto_df.model_train_test(clf=d_param['clf'],
                                                                                       top_bagging=d_param['bagging'],
                                                                                       n_comb=model_iter,
                                                                                       comb_seed=None, verbose=True)

        # if a best model is found, store metrics, else store -1
        if best_model_index is not None:
            HP = res_dict[best_model_index]['metrics']

            df_meta = df_meta.append(
                {'file_name': file, 'date': today_date, 'exec_time': str(round(time() - before, 4)),
                 'nrow': df_raw.shape[0], 'ncol': df_raw.shape[1], 'nnum': len(auto_df.d_features['numerical']),
                 'ncat': len(auto_df.d_features['categorical']), 'ndate': len(auto_df.d_features['date']),
                 'cat_method': d_param['cat_method'], 'n_epoch': n_epoch, 'learning_rate': learning_rate,
                 'batch_size': batch_size, 'crit': crit, 'optim': optim,
                 'process_outliers': str(d_param['outliers']), 'select_method': str(d_param['select_method']),
                 'clf': d_param['clf'], 'bagging': d_param['bagging'], 'n_model': model_iter,
                 'n_valid_models': len(l_valid_models), 'AUC': HP['Roc_auc'] if HP else -1,
                 'delta_auc': HP['delta_auc'] if HP else -1, 'precision': HP['Precision'] if HP else -1,
                 'recall': HP['Recall'] if HP else -1, 'F1': HP['F1'] if HP else -1
                 }, ignore_index=True)

    # store models results
    df_meta.to_csv('meta.csv', index=False)

stop_meta = time()
print(str(round(stop_meta - start_meta, 4)))
