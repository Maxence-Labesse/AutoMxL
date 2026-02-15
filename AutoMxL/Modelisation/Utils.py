import numpy as np
from sklearn.metrics import roc_curve, auc, log_loss, f1_score, precision_score, recall_score
from sklearn import metrics

def classifier_evaluate(y, y_pred, y_proba_pred, verbose=False):
    fpr, tpr, thresholds = roc_curve(y, y_proba_pred)
    acc = metrics.accuracy_score(y, y_pred)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y, y_pred)
    logloss = log_loss(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)

    eval_dict = {
        "fpr tpr": (fpr, tpr),
        "Accuracy": acc,
        "Roc_auc": roc_auc,
        "F1": f1,
        "Logloss": logloss,
        "Precision": precision,
        "Recall": recall
    }

    l_metrics = ["Accuracy", "Roc_auc", "F1", "Logloss", "Precision", "Recall"]

    if verbose:
        list(map(lambda x: print(x + " : ", eval_dict[x]), l_metrics))
        print(metrics.confusion_matrix(y, y_pred), "\n")

    return eval_dict

"""
-------------------------------------------------------------------------------------------------------------
"""

def train_test(df, test_size=0.2, seed=None):
    list_index_df = df.index

    if seed is not None :
        np.random.seed(seed)
    chosen_idx = np.random.choice(list_index_df, replace=False, size=int(len(list_index_df) * test_size))
    df_train = df.loc[~df.index.isin(chosen_idx)]
    df_test = df.loc[df.index.isin(chosen_idx)]

    return df_train, df_test
