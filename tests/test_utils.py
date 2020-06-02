from unittest import TestCase

import pandas as pd

from utils import convert, load_params, create_data_frame, parse_libri_meta, fetch_files


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
        res = load_params("tests/test_data/test_params.yml")
        self.assertEqual(str(res), "HyperParameters(foo=1, bar=2, baz={'eenie': 'a', 'meenie': 'b', 'miny': 'c'})")

    def test_h_params(self):
        from utils import h_params
        print(h_params)

    def test_parse_libri_meta(self):
        res = parse_libri_meta("tests/test_data/libri_dir_struct/SPEAKERS.txt")
        self.assertEqual(res, [['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'], ['84', 'F', 'dev-clean', '8.02', 'Christie Nowak'], ['174', 'M', 'dev-clean', '8.04', 'Peter Eastman']])

    def test_fetch_files(self):
        df = pd.read_csv("../tests/test_data/SPEAKERS.csv")
        row = next(df.itertuples())
        res = fetch_files(row=row, data_path="tests/test_data/libri_dir_struct")
        self.assertEqual(list(res), [('dev-clean', '84', 'dev-clean/84/121123/84-121123-0000.flac'), ('dev-clean', '84', 'dev-clean/84/121123/84-121123-0001.flac'), ('dev-clean', '84', 'dev-clean/84/121123/84-121123-0002.flac'), ('dev-clean', '84', 'dev-clean/84/121550/84-121550-0001.flac'), ('dev-clean', '84', 'dev-clean/84/121550/84-121550-0000.flac'), ('dev-clean', '84', 'dev-clean/84/121550/84-121550-0003.flac'), ('dev-clean', '84', 'dev-clean/84/121550/84-121550-0002.flac')])

    def test_create_data_frame(self):
        meta_path = "tests/test_data/libri_dir_struct/SPEAKERS.txt"
        root_path = "tests/test_data/libri_dir_struct/"
        res = create_data_frame(root_path, meta_path)
        print(res)
