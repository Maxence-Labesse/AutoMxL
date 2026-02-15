from time import time

def timer(func):

    def f(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print('\t\t>>>', func.__name__, 'execution time:', round(after - before, 4), 'secs. <<<')
        return rv

    f.__name__ = func.__name__
    f.__doc__ = func.__doc__

    return f
