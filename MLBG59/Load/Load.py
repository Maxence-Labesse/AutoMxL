""" Data importation :

 - get_delimiter : identify csv file delimiter
 - load data
"""
import pandas as pd


def get_delimiter(csvfile):
    """Identify the delimiter of a .csv file

    Parameters
    ----------
    csvfile : string
        Path and name of the file (Ex : "data/file.csv")

    Returns
    -------
    string
        Identified delimiter

    """
    # csv file reading
    with open(csvfile, 'r') as myCsvfile:
        # Reads one entire line from the file
        header = myCsvfile.readline()

        # Returns the lowest index of the substring if it is found in given string. (-1 = not found)
        if header.find(";") != -1:
            delimiter = ";"
        elif header.find(",") != -1:
            delimiter = ","

    return delimiter


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def load_data(file, index_col=None, verbose=1):
    """Import dataset as a DataFrame
    accept .csv, .xlsx, .xls files

    Parameters
    ----------
    file : string
        Path and name of the file (Ex : "data/file.csv")
        If file is .csv, automatically identify delimiter
    index_col : int, str, sequence of int / str, or False, default None
        Column(s) to use as the row labels of the DataFrame, either given as string name or column index.
        If a sequence of int / str is given, a MultiIndex is used.
    verbose : int (0/1) (Default : 1)
        Get more operations information

    Returns
    -------
    DataFrame :
        dataset imported as DataFrame

    """
    # CSV
    if file.endswith('.csv') or file.endswith('.txt'):
        # Find file delimiter
        file_sep = get_delimiter(file)
        # import 
        df = pd.read_csv(file, encoding="iso-8859-1", sep=file_sep, index_col=index_col)

    # Excel
    elif (file.endswith('.xlsx')) or (file.endswith('.xsl')):
        df = pd.read_excel(file)

    # JSON
    elif file.endswith('.json'):
        # to-do
        pass

    else:
        df = None

    if verbose == 1:
        if df is not None:
            print('-> File ' + file + ' successfully imported as DataFrame')
            print('-> DataFrame size  : ', df.shape)
        else:
            print("File couldn't be imported")

    return df


"""
-------------------------------------------------------------------------------------------------------------
"""


def parse_target(df, target, modalite):
    """
    Transforme la variable cible en variable binaire, en choisissant la modalité de référence

    input
    -----
     > df : dataframe
     > target : string
         variable cible
     > modalite : string
         nom de la modalité de référence

    return
    ------
     > df_bis : dataframe
          le dataframe modifié
     > target : string
          nom de la nouvelle variable

    """
    # Si la variable cible est numérique, transformation en string (nécessaire pour dichotomiser)
    if target in df._get_numeric_data().columns:
        df[target] = df[target].apply(str)

    # Dichotomisation
    target_dummies = pd.get_dummies(df[target])
    # Choix de la nouvelle variable cible et renommage
    target_dummies[target + '_' + modalite] = target_dummies[modalite]

    # Intégration de la nouvelle variable cible dans le dataset
    df_bis = pd.concat((df, target_dummies[target + '_' + modalite]), axis=1)

    # suppresion de l'ancienne variable cible
    del df_bis[target]

    return df_bis, target + '_' + modalite
