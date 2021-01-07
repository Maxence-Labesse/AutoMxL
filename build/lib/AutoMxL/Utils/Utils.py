import random


def random_from_dict(dic, verbose=False):
    """ pick random item for each dict key if value is a list

    Parameters
    ----------
    dic : dict
        input dict
    verbose : bool (Default False)
        Get logging information

    Returns
    -------
        dict with picked values
    """
    d_res = {}

    # for each key of input dc
    for k in dic.keys():

        # if value is a list, pick random, else pick keep item
        if isinstance(dic[k], list):
            d_res[k] = random.choice(dic[k])
        else:
            d_res[k] = dic[k]

        if verbose:
            print(k + ":", d_res[k])

    return d_res
