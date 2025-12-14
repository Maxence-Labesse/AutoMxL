import pandas as pd
from datetime import datetime
from AutoMxL.Explore.Features_Type import features_from_type

class DateEncoder(object):

    def __init__(self,
                 method='timedelta',
                 date_ref=None,):

        assert method in ['timedelta'], "invalid method : select timedelta"

        self.method = method
        self.is_fitted = False
        self.l_var2encode = []
        if date_ref is None:
            self.date_ref = datetime.now()
        else:
            self.date_ref = date_ref

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit(self, df, l_var=None, verbose=False):
        l_date_var = features_from_type(df, typ='date', l_var=None)

        if l_var is None:
            self.l_var2encode = l_date_var
        else:
            self.l_var2encode = [col for col in l_var if col in l_date_var]

        self.is_fitted = True

        if verbose:
            if self.method == 'timedelta':
                print(" **method " + self.method + " / date ref : ", self.date_ref)

            print("  >", len(self.l_var2encode), "features to transform")
            if len(self.l_var2encode) > 0:
                print(" ", self.l_var2encode)

    """
    ----------------------------------------------------------------------------------------------
    """

    def transform(self, df, verbose=False):
        assert self.is_fitted, 'fit the encoding first using .fit method'

        df_local = df.copy()

        if len(self.l_var2encode) > 0:
            df_local = all_to_date(df_local, l_var=self.l_var2encode, verbose=verbose)

            if self.method == 'timedelta':
                df_local, _ = date_to_anc(df_local, l_var=self.l_var2encode, date_ref=self.date_ref, verbose=verbose)

        elif verbose:
            print("  > No date to transform")

        return df_local

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit_transform(self, df, l_var=None, verbose=False):
        df_local = df.copy()
        self.fit(df_local, l_var=l_var, verbose=verbose)
        df_local = self.transform(df_local, verbose=verbose)

        return df_local

"""
----------------------------------------------------------------------------------------------
"""

def all_to_date(df, l_var=None, verbose=False):
    if l_var is None:
        l_var = df.columns.tolist()
    else:
        l_var = [col for col in l_var if col in df.columns.tolist()]

    df_local = df.copy()

    if verbose:
        print('  > features : ', l_var)
        print('  > features conversion to date using "try .to_datetime')

    for col in l_var:
        try:
            if df_local[col].dtype == 'object':
                df_local[col] = pd.to_datetime(df_local[col], errors='raise')
            else:
                df_smpl = df.loc[~df[col].isna()].copy()
                df_smpl[col] = pd.to_datetime(df_smpl[col].astype('Int32').astype(str), errors='raise')
                df_local[col] = pd.to_datetime(df_local[col].astype('Int32').astype(str), errors='coerce')
        except ValueError:
            pass
        except OverflowError:
            pass
        except TypeError:
            pass

    return df_local

"""
-------------------------------------------------------------------------------------------------------------------------
"""

def date_to_anc(df, l_var=None, date_ref=None, verbose=False):
    if date_ref is None:
        date_ref = datetime.now()
    else:
        if isinstance(date_ref, datetime):
            pass
        else:
            date_ref = datetime.strptime(date_ref, '%d/%m/%Y')

    l_date = df.dtypes[df.dtypes == 'datetime64[ns]'].index.tolist()
    if l_var is None:
        l_var = l_date
    else:
        l_var = [col for col in l_var if col in l_date]

    df_local = df.copy()

    l_new_var_names = ['anc_' + col for col in l_var]
    df_local = df_local.apply(lambda x: (date_ref - x).dt.days / 365 if x.name in l_var else x)
    df_local = df_local.rename(columns=dict(zip(l_var, l_new_var_names)))

    if verbose:
        print('  ** Reference date for timelapse computing : ', date_ref)
        list(map(lambda x, y: print("  >", x + ' -> ' + y), l_var, l_new_var_names))

    return df_local, l_new_var_names
