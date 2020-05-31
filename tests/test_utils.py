from collections import namedtuple
from unittest import TestCase
from utils import convert, load_params


class Test(TestCase):
    def test_convert(self):
        hyperparams = {
            'foo': 12,
            'bar': 13,
            'baz': {
                'bim': 'b', 'bam': 'c'
            }
        }
        res = convert(hyperparams)
        print(str(res))
        self.assertEqual(str(res), "HyperParameters(foo=12, bar=13, baz={'bim': 'b', 'bam': 'c'})")

    def test_load_params(self):
        res = load_params("../tests/test_data/test_params.yml")
        print(res)
        self.assertEqual(str(res), "HyperParameters(foo=1, bar=2, baz={'eenie': 'a', 'meenie': 'b', 'miny': 'c'})")

    def test_h_params(self):
        from utils import h_params
        print(h_params)
