"""

print('\nTOP 10 variables avec variance les plus basses :')
print(df_bis[var_list].var().sort_values().reset_index().rename(columns={'index': 'variable', 0: 'variance'}).head(10))
"""


def set_params(self, **params):
    """
    modify object parameters

    input
    -----
    params : dict
        dict containing the new parameters {param : value}

    """
    for k, v in params.items():
        if k not in self.get_params():
            print("invalid parameter")
        else:
            setattr(self, k, v)


# ===================================================
#                 Default Paramaters
# ===================================================
default_RF_param = {'n_estimators': 122,
                    'max_features': 'sqrt',
                    'max_depth': 6,
                    'min_samples_split': 5,
                    }

"""
-------------------------------------------------------------------------------------------------------------
"""


def fast_RF(df, target, param=None, print_res=False):
    """
    Réalise un modèle Random Forest baseline
      Pour l'instant :
        - n'intègre que les features numériques
        - pas possibilité de fixer les HPs du modèle

    input
    -----
     > df : datraframe
     > target : string
         nom de la variable à prédire
     > param : A venir
     > print_res : Bbolean
         si True, affiche les résultats du modèle

    return
    ------
     > df_bis : dataframe
         le dataframe modifié
    """
    # Date to datetime
    df_corr = all_to_date(df)
    # Comlétude des valeurs manquantes par la médianne :
    df_corr1 = fill_all_num(df_corr, method='median', top_var_NA=0)
    # complétude variables catégorielles et datetime :
    df_corr2 = fill_all_cat(df_corr1)
    # On récupère l'année et le mois de la date.
    for col in df_corr2.columns:
        if df_corr2[col].dtype == 'datetime64[ns]':
            df_corr2[col + 'year'] = pd.Series(df_corr2[col].dt.year, index=df_corr2.index)
            df_corr2[col + 'mois'] = pd.Series(df_corr2[col].dt.month, index=df_corr2.index)
            del df_corr2[col]
    df_corr3 = process_cat_outliers(df_corr2, var_list=None, method="percent", threshold=0.05, verbose=0)
    # dichotomisation
    df_corr4 = dummy_all_var(df_corr3, var_list=None, prefix_list=None, keep=False)
    # Déf X et y
    X = df_corr4.copy()
    y = X[target]
    X = X.drop(target, axis=1)
    # Découpage train Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Création du modèle
    clf = RandomForestClassifier(**default_RF_param)
    # Entrainement sur X_train
    clf.fit(X_train, y_train)
    # Prédiction sur X_test
    # probabilités de classification sur X_test
    y_proba = clf.predict_proba(X_test)[:, 1]
    # Valeur prédites sur X_test
    y_pred = clf.predict(X_test)

    eval_dict = classifier_evaluate(y_test, y_pred, y_proba, print_res)

    eval_dict['feature_importances'] = pd.DataFrame(clf.feature_importances_, index=X_train.columns,
                                                    columns=['importance']).sort_values('importance', ascending=False)

    return eval_dict


"""
-------------------------------------------------------------------------------------------------------------
"""