""" contains objets related to data importation :
- get_delimiter : identify csv file delimiter
- load data

"""
import pandas as pd


def get_delimiter(csvfile):
    """
    Identify the delimiter of a .csv file

    Parameters
    ----------
    csvfile : string
        path and name of the file (Ex : "data/file.csv")

    Returns
    -------
    string
        identified delimiter
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
    """
    Import dataset as a DataFrame
    accept .csv, .xlsx, .xls files

    Parameters
    ----------
    file : string
        path and name of the file (Ex : "data/file.csv")
        if file is .csv, automatically identify delimiter
    index_col : int, str, sequence of int / str, or False, default None
        Column(s) to use as the row labels of the DataFrame, either given as string name or column index.
        If a sequence of int / str is given, a MultiIndex is used.
    verbose : int (0/1) (Default : 1)
        get more operations information

    Returns
    -------
    DataFrame :
        dataset imported as dataset
    """
    # CSV
    if file.endswith('.csv'):
        # Find separator
        file_sep = get_delimiter(file)
        # import 
        df = pd.read_csv(file, encoding="iso-8859-1", sep=file_sep, index_col=index_col)
    
    # Excel
    elif (file.endswith('.xlsx')) or (file.endswith('.xsl')):
        df = pd.read_excel(file)

    # JSON
    elif file.endswith('.json'):
        pass

    else:
        df = None

    if verbose==1:
        if df is not None:
            print('-> Fichier '+file+' importé avec succès')
            print('-> Taille du dataframe créé : ', df.shape)
        else:
            print("Le Fichier n'a pas pu être importé (dommage)")

    return df
