"""
AutoML test meta algo :
- Apply AutoML class on /data files (datasets)
- Store datasets informations, AutoML parameters and modelisation perfs in meta.csv
"""

# Import
import random
from time import time
from datetime import date
from MLBG59 import *
import pandas as pd
import glob
from MLBG59.Utils.Display import print_title1, print_dict
from data._data_config import d_files
from MLBG59.param_config import n_epoch, learning_rate, batch_size, crit, optim

#### config
today_date = date.today().strftime("%d/%m/%Y")
l_process_outliers = [True, False]
model_iter = 2
l_clf = ['RF', 'XGBOOST']
l_select_method = ['pca', None]
l_cat_method = ['one_hot', 'deep_encoder']
l_top_bagging = [True, False]
n_iter = 1

start_meta = time()

# Prepare output DataFrame
# df_meta = pd.read_csv('meta.csv', sep=",")

df_meta = pd.DataFrame(columns=['file_name', 'date', 'exec_time',
                                'nrow', 'ncol',
                                'nnum', 'ncat', 'ndate',
                                'cat_method', 'cat_method',
                                'n_epoch', 'learning_rate',
                                'batch_size', 'crit', 'optim',
                                # 'encoder_loss', 'encoder_acc',
                                'process_outliers', 'select_method',
                                'bagging',
                                'clf', 'n_model', 'n_valid_models',
                                'AUC', 'delta_auc',
                                'precision', 'recall', 'F1'])


#############
# Meta Algo #
#############

# Get /data folder files names
def findfiles(path):
    return glob.glob(path)


files_path = findfiles('data/*.csv')

# Apply AutoML class on files
for i in range(n_iter):
    for file in [file for file in d_files.keys()]:
        # if file == 'data\\bank-additional-full.csv'
        # if file == 'data\kddcup99.csv'

        print('\n')
        print_title1(file + "  / niter : " + str(i))

        process_outliers = random.choice(l_process_outliers)
        select_method = random.choice(l_select_method)
        cat_method = random.choice(l_cat_method)
        clf = random.choice(l_clf)
        bagging = random.choice(l_top_bagging)
        print("classifier: " + clf)
        print("select method: " + str(select_method))
        print("cat method: " + cat_method)
        print("process outliers: " + str(process_outliers))
        print("bagging: " + str(bagging))

        # import file
        df_raw = import_data(file, verbose=False)
        print(df_raw.shape)

        print("\nTaille du dataset brut", df_raw.shape)

        # encode
        new_df, target = category_to_target(df_raw, var=d_files[file]['var'], cat=d_files[file]['cat'])

        # start execution-timer timer
        before = time()

        # instantiate AutoML object
        auto_df = AML(new_df.copy(), target=target)

        # get dataset info
        auto_df.explore(verbose=False)

        # data preparation
        auto_df.preprocess(process_outliers=process_outliers, cat_method=cat_method, verbose=False)

        # features selection
        if select_method is not None:
            auto_df.select_features(method=select_method, verbose=False)

        print("Taille du dataset avant modele :", auto_df.shape)

        # random search
        res_dict, l_valid_models, best_model_index, df_test = auto_df.train_model(clf=clf, top_bagging=bagging,
                                                                                  n_comb=model_iter,
                                                                                  comb_seed=None, verbose=True)

        # stop execution-timer tmer
        after = time()

        # store res in df_meta
        nnum, ncat, ndate, = len(auto_df.d_features['numerical']), \
                             len(auto_df.d_features['categorical']), \
                             len(auto_df.d_features['date'])

        if best_model_index is not None:

            HP = res_dict[best_model_index]['metrics']

            df_meta = df_meta.append({'file_name': file, 'date': today_date, 'exec_time': str(round(after - before, 4)),
                                      'nrow': df_raw.shape[0], 'ncol': df_raw.shape[1],
                                      'nnum': nnum, 'ncat': ncat, 'ndate': ndate,
                                      'cat_method': cat_method,
                                      'n_epoch': n_epoch, 'learning_rate': learning_rate,
                                      'batch_size': batch_size, 'crit': crit, 'optim': optim,
                                      'process_outliers': str(process_outliers), 'select_method': str(select_method),
                                      'clf': clf, 'n_model': model_iter, 'n_valid_models': len(l_valid_models),
                                      'AUC': HP['Roc_auc'], 'delta_auc': HP['delta_auc'],
                                      'precision': HP['Precision'], 'recall': HP['Recall'], 'F1': HP['F1']
                                      }, ignore_index=True)

            print(res_dict[best_model_index]['features_importance'])

        else:
            df_meta = df_meta.append({'file_name': file, 'date': today_date, 'exec_time': str(round(after - before, 4)),
                                      'nrow': df_raw.shape[0], 'ncol': df_raw.shape[1],
                                      'nnum': nnum, 'ncat': ncat, 'ndate': ndate,
                                      'cat_method': cat_method,
                                      'n_epoch': n_epoch, 'learning_rate': learning_rate,
                                      'batch_size': batch_size, 'crit': crit, 'optim': optim,
                                      'process_outliers': str(process_outliers), 'select_method': str(select_method),
                                      'bagging': str(bagging),
                                      'clf': clf, 'n_model': model_iter, 'n_valid_models': len(l_valid_models),
                                      'AUC': -1, 'delta_auc': -1,
                                      'precision': -1, 'recall': -1, 'F1': -1
                                      }, ignore_index=True)

        # store models results
        df_meta.to_csv('meta.csv', index=False)

stop_meta = time()
print(str(round(stop_meta - start_meta, 4)))
