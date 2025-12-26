from time import time

def timer(func):
    """
    Décorateur qui mesure et affiche le temps d'exécution d'une fonction.

    Utilisé pour monitorer les performances des étapes du pipeline
    (explore, preprocess, model_train, etc.).

    Args:
        func: Fonction à décorer

    Returns:
        Fonction wrappée qui affiche le temps d'exécution après l'appel
    """
    def f(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print('\t\t>>>', func.__name__, 'execution time:', round(after - before, 4), 'secs. <<<')
        return rv

    f.__name__ = func.__name__
    f.__doc__ = func.__doc__

    return f
