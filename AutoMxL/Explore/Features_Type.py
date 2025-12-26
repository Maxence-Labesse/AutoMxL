import pandas as pd
from time import time
from AutoMxL.Utils.Decorators import timer

def features_from_type(df, typ, l_var=None, th=0.95):
    """
    Retourne les colonnes qui correspondent à un type donné.

    Args:
        df: DataFrame à analyser
        typ: Type recherché (date, identifier, verbatim, boolean, categorical)
        l_var: Liste des colonnes à analyser (toutes si None)
        th: Seuil d'unicité pour identifier/verbatim

    Returns:
        Liste des colonnes correspondant au type
    """
    assert typ in ['date', 'identifier', 'verbatim', 'boolean', 'categorical'], 'Invalid type'

    if l_var is None:
        df_local = df.copy()
    else:
        df_local = df[l_var].copy()

    if typ == 'date':
        l_var = [col for col in df_local.columns if is_date(df_local, col)]
    elif typ == 'identifier':
        l_var = [col for col in df_local.columns if is_identifier(df_local, col, th)]
    elif typ == 'verbatim':
        l_var = [col for col in df_local.columns if is_verbatim(df_local, col, th)]
    elif typ == 'boolean':
        l_var = [col for col in df_local.columns if is_boolean(df_local, col)]
    elif typ == 'categorical':
        l_var = [col for col in df_local.columns if is_categorical(df_local, col, th)]

    return l_var

"""
-------------------------------------------------------------------------------------------------------------------------
"""

def is_date(df, col):
    """
    Détecte si une colonne contient des dates.

    Échantillonne 10 valeurs et tente de les parser en datetime.
    Gère les types object et numériques.

    Args:
        df: DataFrame source
        col: Nom de la colonne à tester

    Returns:
        True si la colonne semble contenir des dates
    """
    sample_size = 10
    full_col = df[col].loc[~df[col].isna()]

    smpl_size = min(sample_size, len(full_col))
    smpl = full_col.sample(smpl_size).copy()
    if df[col].dtype != 'datetime64[ns]':
        try:
            if smpl.dtype == 'object':
                smpl = pd.to_datetime(smpl, errors='raise')
            else:
                smpl = pd.to_datetime(smpl.astype('Int32').astype(str), errors='raise')
        except ValueError:
            pass
        except OverflowError:
            pass
        except TypeError:
            pass
    return smpl.dtype == 'datetime64[ns]'

"""
-------------------------------------------------------------------------------------------------------------------------
"""

def is_identifier(df, col, th=0.95):
    """
    Détecte si une colonne est un identifiant.

    Une colonne est un identifiant si :
    - Taux d'unicité >= seuil (th)
    - Toutes les valeurs ont la même longueur
    - Ce n'est pas une date

    Args:
        df: DataFrame source
        col: Nom de la colonne à tester
        th: Seuil d'unicité (défaut: 0.95)

    Returns:
        True si la colonne semble être un identifiant
    """
    full_col = df[col].loc[~df[col].isna()]

    if full_col.nunique() / full_col.count() >= th:
        if df[col].dtype != 'object':
            try:
                full_col = full_col.astype('Int32').astype(str)
            except ValueError:
                return False
            except OverflowError:
                return False
            except TypeError:
                return False

        if full_col.apply(lambda x: len(x)).nunique() == 1:
            if not is_date(df, col):
                return True
            else:
                return False
        else:
            return False
    else:
        return False

"""
-------------------------------------------------------------------------------------------------------------------------
"""

def is_verbatim(df, col, th=0.95):
    """
    Détecte si une colonne contient du texte libre (verbatim).

    Une colonne est verbatim si :
    - Type object (texte)
    - Taux d'unicité >= seuil (th)
    - Longueurs des valeurs variables

    Args:
        df: DataFrame source
        col: Nom de la colonne à tester
        th: Seuil d'unicité (défaut: 0.95)

    Returns:
        True si la colonne semble être du texte libre
    """
    if df[col].dtype == 'object':
        full_col = df[col].loc[~df[col].isna()]
    else:
        return False

    if full_col.nunique() / full_col.count() >= th:
        if full_col.apply(lambda x: len(x)).nunique() > 1:
            return True
        else:
            return False
    else:
        return False

"""
-------------------------------------------------------------------------------------------------------------------------
"""

def is_boolean(df, col):
    """
    Détecte si une colonne est booléenne.

    Une colonne est booléenne si elle a exactement 2 valeurs
    uniques et plus de 2 observations.

    Args:
        df: DataFrame source
        col: Nom de la colonne à tester

    Returns:
        True si la colonne est booléenne
    """
    full_col = df[col].loc[~df[col].isna()]

    if full_col.nunique() == 2:
        if len(full_col) > 2:
            return True
        else:
            return False

    else:
        return False

"""
-------------------------------------------------------------------------------------------------------------------------
"""

def is_categorical(df, col, th=0.95):
    """
    Détecte si une colonne est catégorielle.

    Une colonne est catégorielle si :
    - Plus de 2 valeurs uniques
    - Taux d'unicité < seuil (th)
    - Type object OU moins de 5 valeurs uniques (numérique)

    Args:
        df: DataFrame source
        col: Nom de la colonne à tester
        th: Seuil d'unicité (défaut: 0.95)

    Returns:
        True si la colonne est catégorielle
    """
    full_col = df[col].loc[~df[col].isna()]
    if full_col.nunique() > 2:
        if (full_col.nunique() / full_col.count()) < th:
            if df[col].dtype == 'object':
                return True
            else:
                if full_col.nunique() < 5:
                    return True
                else:
                    return False
        else:
            return False
    else:
        return False
