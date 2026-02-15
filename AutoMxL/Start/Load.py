import pandas as pd

def get_delimiter(file):
    if file.endswith('.csv') or file.endswith('.txt'):
        with open(file, 'r') as myCsvfile:
            header = myCsvfile.readline()

            if header.find(";") != -1:
                delimiter = ";"
            elif header.find(",") != -1:
                delimiter = ","

        return delimiter

    else:
        print('Please use a .csv or .txt file')

def import_data(file, index_col=None, verbose=False):
    if file.endswith('.csv') or file.endswith('.txt'):
        file_sep = get_delimiter(file)
        df = pd.read_csv(file, encoding="iso-8859-1", sep=file_sep, index_col=index_col)

    elif (file.endswith('.xlsx')) or (file.endswith('.xsl')):
        df = pd.read_excel(file)

    elif file.endswith('.json'):
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
