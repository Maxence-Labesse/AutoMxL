import pandas as pd


def get_delimiter(csvfile):
    """
    identify delimiter of a file
    
    input
    -----
     > csvfile : string
          path+name of the file
    
    return
    ------
      > delimiter : char

    """
    # csv file reading
    with open(csvfile, 'r') as myCsvfile:
        # Reads one entire line from the file
        header = myCsvfile.readline()
        
        # Returns the lowest index of the substring if it is found in given string. (-1 = not found)
        if header.find(";") != -1:
            return ";"
        if header.find(",") != -1:
            return ","

    return ";"


"""
-------------------------------------------------------------------------------------------------------------------------
"""


def load_data(file, index_col=None, verbose=1):
    """
    import a file as a DataFrame, no matter what his format is (csv, xlsx, Json)
    
    input
    -----
     > path : string
          path of the file
     > file : string
          name of the file
     > index_col : int, str, sequence of int / str, or False, default None
          Column(s) to use as the row labels of the DataFrame, either given as string name or column index.
          If a sequence of int / str is given, a MultiIndex is used.
        
    return
    ------
     > df : DataFrame
          imported dataset
        
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
