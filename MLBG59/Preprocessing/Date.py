""" Date Features processing functions:

 - DateEncoder (class) : encode date features
 - all_to_date (func): detect dates from num/cat features and transform them to datetime format.
 - date_to_anc (func): transform datetime features to timedelta according to a ref date
"""
import pandas as pd
from datetime import datetime
from MLBG59.Explore.Features_Type import features_from_type


class DateEncoder(object):
    """Encode categorical features

    Available methods :

    - timedelta : compute time between date feature and parameter date_ref

    Parameters
    ----------
    method : string (Default : deep_encoder)
        method used to encode dates
        Available methods : "timedelta"
    date_ref : string '%d/%m/%y' (Default : None)
        Date to compute timedelta.
        If None, today date
    """

    def __init__(self,
                 method='timedelta',
                 date_ref=None,
                 ):

        assert method in ['timedelta'], "invalid method : select timedelta"

        self.method = method
        self.is_fitted = False
        self.l_var2encode = []

        if date_ref is None:
            self.date_ref = datetime.now()
        else:
            self.date_ref = datetime.strptime(date_ref, '%d/%m/%Y')

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit(self, df, l_var=None, verbose=False):
        """fit encoder

        Parameters
        ----------
        df : DataFrame
            input dataset
        l_var : list
            features to encode.
            If None, all features identified as dates (see Features_Type module)
        verbose : boolean (Default False)
            Get logging information
        """
        l_date_var = features_from_type(df, typ='date', l_var=None)

        if l_var is None:
            self.l_var2encode = l_date_var
        else:
            self.l_var2encode = [col for col in l_var if col in l_date_var]

        self.is_fitted = True

        # verbose
        if verbose:
            if self.method == 'timedelta':
                print(" **method " + self.method + " / date ref : ", self.date_ref)
            print("  >", len(self.l_var2encode), "features to transform")
            if len(self.l_var2encode) > 0:
                print(self.l_var2encode)

    """
    ----------------------------------------------------------------------------------------------
    """

    def transform(self, df, verbose=False):
        """ transform dataset date features using the encoder.
        Can be done only if encoder has been fitted

        Parameters
        ----------
        df : DataFrame
            dataset to transform
        verbose : boolean (Default False)
            Get logging information
        """
        assert self.is_fitted, 'fit the encoding first using .fit method'

        df_local = df.copy()

        if len(self.l_var2encode) > 0:
            # transform features to datetime
            df_local = all_to_date(df_local, l_var=self.l_var2encode, verbose=verbose)

            # method timedelta
            if self.method == 'timedelta':
                df_local, _ = date_to_anc(df_local, l_var=self.l_var2encode, date_ref=self.date_ref, verbose=verbose)

        else:
            print("  > No date to transform")

        return df_local

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit_transform(self, df, l_var=None, verbose=False):
        """fit and transform dataset with encoder

        Parameters
        ----------
        df : DataFrame
            input dataset
        l_var : list
            features to encode.
            If None, all features identified as dates (see Features_Type module)
        verbose : boolean (Default False)
            Get logging information
        """
        df_local = df.copy()
        self.fit(df_local, l_var=l_var, verbose=verbose)
        df_local = self.transform(df_local, verbose=verbose)

        return df_local


"""
----------------------------------------------------------------------------------------------
"""


def all_to_date(df, l_var=None, verbose=False):
    """Detect dates from selected/all features and transform them to datetime format.
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    l_var : list (Default : None)
        Names of the features
        If None, all the features
    verbose : boolean (Default False)
        Get logging information
        
    Return
    -------
    DataFrame
        Modified dataset
    """
    # if var_list = None, get all df features
    # else, exclude features if not in df
    if l_var is None:
        l_var = df.columns.tolist()
    else:
        l_var = [col for col in l_var if col in df.columns.tolist()]

    df_local = df.copy()

    if verbose:
        print('  > features : ', l_var)
        print('  > features conversion to date using "try .to_datetime')

    # for each feature in var_list, try to convert to datetime
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
    """Transform selected/all datetime features to timedelta according to a ref date
    
    Parameters
    ----------
    df : DataFrame
        Input dataset
    l_var : list (Default : None)
        List of the features to analyze.
        If None, contains all the datetime features
    date_ref : string '%d/%m/%y' (Default : None)
        Date to compute timedelta.
        If None, today date
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame
        Modified dataset
        
    list
        New timedelta features names
    """
    # if date_ref is None, use today date
    if date_ref is None:
        date_ref = datetime.now()
    else:
        date_ref = datetime.strptime(date_ref, '%d/%m/%Y')

    # if var_list = None, get all datetime features
    # else, exclude features from var_list whose type is not datetime
    l_date = df.dtypes[df.dtypes == 'datetime64[ns]'].index.tolist()
    if l_var is None:
        l_var = l_date
    else:
        l_var = [col for col in l_var if col in l_date]

    df_local = df.copy()

    # new variables names
    l_new_var_names = ['anc_' + col for col in l_var]
    # compute time delta for selected dates variables
    df_local = df_local.apply(lambda x: (date_ref - x).dt.days / 365 if x.name in l_var else x)
    # rename columns
    df_local = df_local.rename(columns=dict(zip(l_var, l_new_var_names)))

    if verbose:
        print('  ** Reference date for timelapse computing : ', date_ref)
        list(map(lambda x, y: print("  >", x + ' -> ' + y), l_var, l_new_var_names))

    return df_local, l_new_var_names
