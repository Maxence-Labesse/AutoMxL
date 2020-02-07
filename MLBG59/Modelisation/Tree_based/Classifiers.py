

def features_importance_select(eval_dict, treshold):
    """
    get most important features according to a threshold

    input
    -----
     > eval_dict : dict
        model evaluation dict
     > threshold : int/float
        - if int : number of top important features to get
        - if float : cumulative importance rate of top features

    return
    ------
     > n_feat_list : list
        most important features
    """
    rf_top_feat = eval_dict['feature_importances']
    rf_top_feat['Features'] = rf_top_feat.index
    rf_top_feat = rf_top_feat.reset_index(drop=True)
    rf_top_feat = rf_top_feat.sort_values('importance', ascending=False)

    if isinstance(treshold, int):
        n_feat_list = rf_top_feat['Features'].tolist()[0:treshold]

    elif isinstance(treshold, float):
        rf_top_feat['cum_importance'] = rf_top_feat['importance'].cumsum()
        val_ref = rf_top_feat['cum_importance'].loc[rf_top_feat['cum_importance'] >= treshold].min()
        n_feat_list = rf_top_feat['Features'].loc[rf_top_feat['cum_importance'] <= val_ref].tolist()
    else:
        print("il faut que le seuil soit un entier ou un décimal !")

    return n_feat_list
