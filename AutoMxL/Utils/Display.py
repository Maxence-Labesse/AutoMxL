def print_title1(titre, color_code=34):
    col = '\033[' + str(color_code) + 'm'
    print(col + '-------------------')
    print(' ' + '\033[1m' + titre + '\033[0m' + col)
    print('-------------------' + '\033[0m')

def bold_print(text):
    print('\033[1m' + text + '\033[0m')

def color_print(text, color_code=34):
    col = '\033[' + str(color_code) + 'm'
    print(col + text + '\033[0m')

def print_dict(dic):
    for key, value in dic.items():
        print(key, ' : ', value)
