import random

def random_from_dict(dic, verbose=False):
    d_res = {}

    for k in dic.keys():

        if isinstance(dic[k], list):
            d_res[k] = random.choice(dic[k])
        else:
            d_res[k] = dic[k]

        if verbose:
            print(k + ":", d_res[k])

    return d_res
