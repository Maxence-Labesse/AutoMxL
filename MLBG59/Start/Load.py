"""

- get_delimiter : identify delimiter for a csv/txt file
- load_data :

"""
import pandas as pd


def get_delimiter(file):
    """Identify the delimiter for a csv/txt file

    Parameters
    ----------
    file : string
        Path and name of the file (Ex : "data/file.csv")

    Returns
    -------
    string - identified delimiter
    """
    if file.endswith('.csv') or file.endswith('.txt'):
        # file reading
        with open(file, 'r') as myCsvfile:
            # Reads one entire line from the file
            header = myCsvfile.readline()

            # Returns the lowest index of the substring if it is found in given string. (-1 = not found)
            if header.find(";") != -1:
                delimiter = ";"
            elif header.find(",") != -1:
                delimiter = ","

        return delimiter

    else :
        print('Please use a .csv or .txt file')


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def import_data(file, index_col=None, verbose=False):
    """Import dataset as a DataFrame.
    Accept .csv, .xlsx, .xls files

    Parameters
    ----------
    file : string
        Path and name of the file (Ex : "data/file.csv")
        If file is .csv, automatically identify delimiter
    index_col : int, str, sequence of int / str, or False (Default None)
        Column(s) to use as the row labels of the DataFrame, either given as string name or column index.
        If a sequence of int / str is given, a MultiIndex is used.
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame : dataset imported as DataFrame

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

    if verbose:
        if df is not None:
            print('-> File ' + file + ' successfully imported as DataFrame')
            print('-> DataFrame size  : ', df.shape)
        else:
            print("File couldn't be imported")

    return df
