"""Variables type identification function

- features_from_type (func): get all features for a selected type
- is_date (func): test if a variable is a date
- is_identifier (func): test if a variable is an identifier
- is_verbatim (func): test if a variable is a verbatim
- is_boolean (func): test if a variable is a boolean
- is_categorical (func): test if a variable is a categorical one (with more than 2 categories)
"""
import pandas as pd
from time import time
from MLBG59.Utils.Decorators import timer


@timer
def features_from_type(df, typ, l_var=None, th=0.95):
    """Get features of a selected type :

    - date : try to apply to_datetime
    - identifier :
        - #(unique values)/#(total values) > threshold (default 0.95)
        - AND length is the same for all values (for non NA)
    - verbatim :
        - #(unique values)/#(total values) >= threshold (default 0.95)
        - AND length is NOT the same for all values (for non NA)
    - boolean : #(distinct values) = 2
    - categorical :
        - not a date
        - #(unique values)/#(total values) < threshold (default 0.95)
        - AND #(uniques values)>2
        - AND for num values #(unique values)<30

    Parameters
    ----------
    df : DataFrame
        input dataset
    typ : string
        selected type to get features:

        - 'date'
        - 'identifier'
        - 'verbatim'
        - 'boolean'
        - categorical
        
    l_var : list (Default : None)
        variables names. If None, all dataset columns
    th : float (Default : 0.95)
        threshold used to identify identifiers/verbatims variables

    Returns
    -------
    list
        identified variables names
    """
    assert typ in ['date', 'identifier', 'verbatim', 'boolean', 'categorical'], 'Invalid type'
    print(typ)

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
    """Test if a variable is a date.

    Method : try to apply to_datetime

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name

    Returns
    -------
    res : boolean
        test result
    """
    sample_size = 10
    full_col = df[col].loc[~df[col].isna()]

    smpl_size = min(sample_size, len(full_col))
    smpl = full_col.sample(smpl_size).copy()
    # if col is numerical/object type, try apply to_datetime
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
    # if col is datetime type, res = True
    return smpl.dtype == 'datetime64[ns]'


def is_date_save(df, col):
    """Test if a variable is a date.

    Method : try to apply to_datetime

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name

    Returns
    -------
    res : boolean
        test result
    """
    before = time()
    smpl_size = min(10, len(df.loc[~df[col].isna()]))
    print(time() - before)
    df_smpl = df.loc[~df[col].isna()].sample(smpl_size).copy()
    print(time() - before)
    # if col is numerical/object type, try apply to_datetime
    if df[col].dtype != 'datetime64[ns]':
        try:
            if df_smpl[col].dtype == 'object':
                df_smpl[col] = pd.to_datetime(df_smpl[col], errors='raise')
                print(time() - before)
            else:
                df_smpl[col] = pd.to_datetime(df_smpl[col].astype('Int32').astype(str), errors='raise')
                print(time() - before)
        except ValueError:
            pass
        except OverflowError:
            pass
        except TypeError:
            pass
    # if col is datetime type, res = True
    return df_smpl[col].dtype == 'datetime64[ns]'


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def is_identifier(df, col, th=0.95):
    """Test if a variable is an identifier.

    - #(unique values)/#(total values) > threshold (default 0.95)
    - AND length is the same for all values (for non NA)
    - AND not date

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name
    th : float (Default : 0.95)
        threshold rate

    Returns
    -------
    res : boolean
        test result
    """
    full_col = df[col].loc[~df[col].isna()]

    # test if #(v unique values)/#(v,total,values) >= threshold (default 0.95)
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

        # test if all (non NA) values have the same length
        if full_col.apply(lambda x: len(x)).nunique() == 1:
            if not is_date(df, col):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def is_identifier_save(df, col, th=0.95):
    """Test if a variable is an identifier.

    - #(unique values)/#(total values) > threshold (default 0.95)
    - AND length is the same for all values (for non NA)
    - AND not date

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name
    th : float (Default : 0.95)
        threshold rate

    Returns
    -------
    res : boolean
        test result
    """
    # get variable serie with non NA values
    if df[col].dtype == 'object':
        full_col = df[col].loc[~df[col].isna()]
    else:
        try:
            full_col = df[col].loc[~df[col].isna()].astype('Int32').astype(str)
        except ValueError:
            return False
        except OverflowError:
            return False
        except TypeError:
            return False

    # test if all (non NA) values have the same length
    length_test = full_col.apply(lambda x: len(x)).nunique() == 1
    # test if #(v unique values)/#(v,total,values) >= threshold (default 0.95)
    diff_test = full_col.nunique() / full_col.count() >= th
    # test if not a date if other tests are True
    if length_test * diff_test:
        date_test = not is_date(df, col)

        return length_test * diff_test * date_test

    # True is both tests are respected
    return length_test * diff_test


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def is_verbatim(df, col, th=0.95):
    """Test if a variable is a verbatim.

    - #(unique values)/#(total values) >= threshold (default 0.95)
    - AND length is NOT the same for all values (for non NA)

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name
    th : float (Default : 0.95)
        threshold rate

    Returns
    -------
    res : boolean
        test result
    """
    # get variable serie with non NA values
    if df[col].dtype == 'object':
        full_col = df[col].loc[~df[col].isna()]
    else:
        return False

    # test if #(v unique values)/#(v,total,values) > threshold (default 0.95)
    if full_col.nunique() / full_col.count() >= th:
        # test if all (non NA) values have the same length
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
    """Test if a variable is a boolean.

    - #(distinct values) = 2

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name

    Returns
    -------
    res : boolean
        test result
    """
    full_col = df[col].loc[~df[col].isna()]
    # get variable serie with non NA values

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
    """Test if a variable is a categorical one (with more than 2 categories).

    - not a date
    - #(unique values)/#(total values) < threshold (default 0.95
    - AND #(uniques values)>2
    - AND for num values #(unique values)<30

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name
    th : float (Default : 0.95)
        threshold

    Returns
    -------
    res : boolean
        test result
    """
    # get variable serie with non NA values
    full_col = df[col].loc[~df[col].isna()]
    if full_col.nunique() > 2:
        if (full_col.nunique() / full_col.count()) < th:
            if df[col].dtype == 'object':
                return True
            else:
                if full_col.nunique() < 30:
                    return True
                else:
                    return False
        else:
            return False
    else:
        return False


def is_categorical_old(df, col, th=0.95):
    """Test if a variable is a categorical one (with more than 2 categories).

    - not a date
    - #(unique values)/#(total values) < threshold (default 0.95
    - AND #(uniques values)>2
    - AND for num values #(unique values)<30

    Parameters
    ----------
    df : DataFrame
        input dataset
    col : string
        variable name
    th : float (Default : 0.95)
        threshold

    Returns
    -------
    res : boolean
        test result
    """
    # get variable serie with non NA values
    if df[col].dtype == 'object':
        full_col = df[col].loc[~df[col].isna()]
    else:
        try:
            full_col = df[col].loc[~df[col].isna()].astype('Int32').astype(str)
        except ValueError:
            return False
        except OverflowError:
            return False
        except TypeError:
            return False

    cat_nb_test = full_col.nunique() > 2

    diff_test = (full_col.nunique() / full_col.count()) < th
    #
    nuniq_test = df[col].loc[~df[col].isna()].nunique() < 30

    if df[col].dtype == 'object':
        return cat_nb_test * diff_test

    else:
        return cat_nb_test * diff_test * nuniq_test
