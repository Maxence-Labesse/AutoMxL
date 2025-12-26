import pytest
import random

from AutoMxL.Utils.Decorators import timer
from AutoMxL.Utils.Utils import random_from_dict


class TestTimer:

    def test_timer_measures_execution_time(self, capsys):
        """Vérifie que le décorateur mesure le temps et retourne le résultat."""
        @timer
        def dummy_function(x):
            return x * 2

        result = dummy_function(5)

        assert result == 10
        captured = capsys.readouterr()
        assert 'dummy_function' in captured.out
        assert 'execution time' in captured.out


class TestRandomFromDict:

    def test_random_from_dict_selects_from_lists(self):
        """Vérifie que les valeurs listes donnent une sélection aléatoire."""
        random.seed(42)
        dic = {
            'param1': [1, 2, 3, 4, 5],
            'param2': ['a', 'b', 'c']
        }

        result = random_from_dict(dic)

        assert result['param1'] in dic['param1']
        assert result['param2'] in dic['param2']

    def test_random_from_dict_keeps_non_list_values(self):
        """Vérifie que les valeurs non-listes sont conservées telles quelles."""
        dic = {
            'fixed_param': 'fixed_value',
            'list_param': [1, 2, 3]
        }

        result = random_from_dict(dic)

        assert result['fixed_param'] == 'fixed_value'
        assert result['list_param'] in [1, 2, 3]
