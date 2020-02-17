import numpy as np
from sklearn.metrics import roc_curve, auc, log_loss, f1_score, precision_score, recall_score
from sklearn import metrics


def classifier_evaluate(y, y_pred, y_proba_pred, print_level=1):
    """
    affiche et renvoie les différentes métriques de notre classification avec bagging
        
    input
    -----
     > y : pandas.Series
          série qui contient les classifications réelles
     > list_proba_pred : numpy.ndarray
          contient les probabilités de classification obtenus par notre bagging
     > list_pred : numpy.ndarray
          contient les prédictions pour chaque ligne de X
            
    return
    ------
     > eval_dict : dictionnaire
          contient l'ensemble des métriques
        
    """
    # Calcul des métriques
    fpr, tpr, thresholds = roc_curve(y, y_proba_pred) 
    acc = metrics.accuracy_score(y, y_pred)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y, y_pred)
    logloss = log_loss(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)

    # Affichage des métriques selon top_print
    if print_level > 0:
        print("Accuracy : ", acc)
        print("F1 : ", f1)
        print("Logloss : ", logloss)
        print("Precision : ", precision)
        print("Recall : ", recall)
        print("\nMatrice de confusion : ")
        print(metrics.confusion_matrix(y, y_pred), "\n")

    # Stockage des métriques dans eval_dict
    eval_dict = {
        "fpr tpr": (fpr, tpr),
        "Accuracy": acc,
        "Roc_auc": roc_auc,
        "F1": f1,
        "Logloss": logloss,
        "Precision": precision,
        "Recall": recall
    }
    
    return eval_dict


"""
-------------------------------------------------------------------------------------------------------------
"""


def train_test(df, test_size):
    """
    Sépare la base de travail en échantillons Train et Test (sans exlure la variable cible de l'échantillon Train)
   
    input
    -----
     > df : dataframe
     > list_proba_pred : numpy.ndarray
          contient les probabilités de classification obtenus par notre bagging
     > list_pred : numpy.ndarray
          contient les prédictions pour chaque ligne de X
            
    return
    ------
     > eval_dict : dictionnaire
          contient l'ensemble des métriques

    """
    # Liste des index
    list_index_df = df.index
    
    # Tirage aléatoire dans la liste des index (en fonction de train_size)
    chosen_idx = np.random.choice(list_index_df, replace=False, size=int(len(list_index_df)*test_size))
    df_train = df.loc[~df.index.isin(chosen_idx)]
    df_test = df.loc[df.index.isin(chosen_idx)]
    
    return df_train, df_test


"""
----------------------------------------------------------------------------------------------------------------------
"""


def get_type_features(df, features_type, var_list=None):
    """
    - if var_list is filled : keep
    - if var_list is None :

    input
    -----
     > df : DataFrame
         dataset
     > var_list : list (Default : None)
         list of the features
     > features_type = string ('num, 'cat')
        type
    """
    if features_type == 'num':
        type_list = df._get_numeric_data().columns
    elif features_type == 'cat':
        type_list = df.select_dtypes(include=object).columns
    elif features_type == 'date':
        type_list = df.dtypes[df.dtypes == 'datetime64[ns]'].index.tolist()
    elif features_type == 'all':
        type_list = df.columns.tolist()
    else:
        type_list = None

    if type_list is not None:
        if var_list is None:
            # if var_list = None, get all features of type "type"
            var_list = type_list
        # else, exclude features from var_list whose type is not of type "type"
        else:
            var_list = [i for i in var_list if i in type_list]

    return var_list
