def print_title1(titre, color_code=34):
    """
    print text as a title
    
    input
    -----
    titre : string
        text
        
    color_code : int
        code couleur  (voir http://ascii-table.com/ansi-escape-sequences.php)
        
    return
    ------
    print formatted text
    """
    col = '\033[' + str(color_code) + 'm'
    print(col + '-------------------')
    print(' ' + '\033[1m' + titre + '\033[0m' + col)
    print('-------------------' + '\033[0m')


def bold_print(text):
    """
    print bold text
    
    input
    -----
    text : string
       text to print bold
    
    """
    print('\033[1m' + text + '\033[0m')


def color_print(text, color_code=34):
    """
    print colorized text
    
    input
    -----
    text : string
       text to print crolorized

    """
    col = '\033[' + str(color_code) + 'm'
    print(col + text + '\033[0m')
