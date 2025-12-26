import random

def random_from_dict(dic, verbose=False):
    """
    Sélectionne une valeur aléatoire pour chaque clé d'un dictionnaire.

    Pour chaque clé, si la valeur est une liste, choisit un élément au hasard.
    Sinon, conserve la valeur telle quelle. Utilisé pour le random search
    d'hyperparamètres.

    Args:
        dic: Dictionnaire avec des listes de valeurs possibles
        verbose: Affiche les valeurs sélectionnées

    Returns:
        Dictionnaire avec une valeur unique par clé
    """
    d_res = {}

    for k in dic.keys():

        if isinstance(dic[k], list):
            d_res[k] = random.choice(dic[k])
        else:
            d_res[k] = dic[k]

        if verbose:
            print(k + ":", d_res[k])

    return d_res
