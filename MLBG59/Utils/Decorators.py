from time import time
from MLBG59.Utils.Display import print_title1
from types import CodeType

# Timer
def timer(func):
    def f(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print('\t\t>>>',func.__name__,'execution time:', round(after - before, 4),'secs. <<<')
        return rv
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__

    return f

"""
------------------------------------------------------------------------------------------------------------------------
"""
# def funct_title
def func_title(func):
    def f(*args, **kwargs):
        print('\n')
        print_title1(func.__name__)
        rv = func(*args, **kwargs)
        return rv
    f.__name__ = func.__name__
    f.__doc__ = func.__doc.__
    return f