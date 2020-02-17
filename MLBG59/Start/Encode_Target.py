import pandas as pd

def category_to_target(df, var, cat):
    """Transform target to boolean (1/0), choosing the reference modality

    Parameters
    ----------
    df : DataFrame
        input dataset
    var : string
        variable containing the target category
    cat : string
         target category

    Returns
    -------
    DataFrame : modified dataset
    string : new target name (var_cat)
    """
    # transform variable to string if numerical
    if var in df._get_numeric_data().columns:
        df[var] = df[var].apply(str)

    # one hot encoding
    target_dummies = pd.get_dummies(df[var])
    # select cat feature
    target_dummies[var + '_' + cat] = target_dummies[cat]

    # add encoded cat feature to dataset
    df_bis = pd.concat((df, target_dummies[var + '_' + cat]), axis=1)

    # remove var
    del df_bis[var]

    return df_bis, var + '_' + cat