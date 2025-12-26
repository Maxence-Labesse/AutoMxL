def print_title1(titre, color_code=34):
    """
    Affiche un titre encadré pour marquer une étape du pipeline.

    Utilisé pour signaler le début d'une phase (Explore, Preprocessing, etc.)
    dans la sortie console.

    Args:
        titre: Texte du titre à afficher
        color_code: Code ANSI de couleur (défaut: 34 = bleu)
    """
    col = '\033[' + str(color_code) + 'm'
    print(col + '-------------------')
    print(' ' + '\033[1m' + titre + '\033[0m' + col)
    print('-------------------' + '\033[0m')

def bold_print(text):
    """
    Affiche du texte en gras dans la console.

    Args:
        text: Texte à afficher
    """
    print('\033[1m' + text + '\033[0m')

def color_print(text, color_code=34):
    """
    Affiche du texte en couleur dans la console.

    Args:
        text: Texte à afficher
        color_code: Code ANSI de couleur (défaut: 34 = bleu, 31 = rouge, 32 = vert)
    """
    col = '\033[' + str(color_code) + 'm'
    print(col + text + '\033[0m')

def print_dict(dic):
    """
    Affiche les clés et valeurs d'un dictionnaire.

    Utilisé pour afficher les résultats (features détectées, métriques, etc.).

    Args:
        dic: Dictionnaire à afficher
    """
    for key, value in dic.items():
        print(key, ' : ', value)
