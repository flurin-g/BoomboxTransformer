from unittest import TestCase

import pandas as pd
import yaml

from utils import create_libri_data_frame, parse_libri_meta, fetch_files, CWD, HyperParameters, create_urban_data_frame


class TestHyperParameters(TestCase):
    def test_from_dict(self):
        with open(CWD / "tests/test_data/hyper_parameters.yml") as file:
            param_dict = yaml.full_load(file)

        h_params = HyperParameters.from_dict(param_dict)
        self.assertIsInstance(h_params, HyperParameters)
        self.assertEqual(h_params.n_mels, 128)
        self.assertEqual(h_params.gamma, 0.95)
        self.assertEqual(h_params.libri_speakers, "data/LibriSpeech/SPEAKERS.txt")
        self.assertEqual(h_params.libri_drop_subsets, ['dev-other', 'test-other', 'train-other-500'])

    def test_from_yaml(self):
        res = HyperParameters.from_yaml(CWD / "tests/test_data/hyper_parameters.yml")
        self.assertIsInstance(res, HyperParameters)
        self.assertEqual(res.n_mels, 128)
        self.assertEqual(res.gamma, 0.95)
        self.assertEqual(res.libri_speakers, "data/LibriSpeech/SPEAKERS.txt")
        self.assertEqual(res.libri_drop_subsets, ['dev-other', 'test-other', 'train-other-500'])


class Test(TestCase):

    def test_h_params(self):
        from utils import h_params
        print(h_params)

    def test_parse_libri_meta(self):
        res = parse_libri_meta("tests/test_data/libri_dir_struct/SPEAKERS.txt")
        self.assertEqual(res, [['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'],
                               ['84', 'F', 'dev-clean', '8.02', 'Christie Nowak'],
                               ['174', 'M', 'dev-clean', '8.04', 'Peter Eastman']])

    def test_parse_libri_meta_cbw_simon(self):
        res = parse_libri_meta("tests/test_data/cbw_simon.txt")
        self.assertListEqual(res, [['60', 'M', 'train-clean-100', '20.18', 'CBW Simon'], ['61', 'M', 'test-clean', '8.08', 'Paul-Gabriel Wiener']])

    def test_fetch_files(self):
        df = pd.read_csv("../tests/test_data/SPEAKERS.csv")
        row = next(df.itertuples())
        res = fetch_files(row=row, data_path="tests/test_data/libri_dir_struct")
        self.assertEqual(list(res), [('dev-clean', '84', 'dev-clean/84/121123/84-121123-0000.flac'),
                                     ('dev-clean', '84', 'dev-clean/84/121123/84-121123-0001.flac'),
                                     ('dev-clean', '84', 'dev-clean/84/121123/84-121123-0002.flac'),
                                     ('dev-clean', '84', 'dev-clean/84/121550/84-121550-0001.flac'),
                                     ('dev-clean', '84', 'dev-clean/84/121550/84-121550-0000.flac'),
                                     ('dev-clean', '84', 'dev-clean/84/121550/84-121550-0003.flac'),
                                     ('dev-clean', '84', 'dev-clean/84/121550/84-121550-0002.flac')])

    def test_create_libri_data_frame(self):
        meta_path = "tests/test_data/libri_dir_struct/SPEAKERS.txt"
        root_path = "tests/test_data/libri_dir_struct/"
        res = create_libri_data_frame(root_path, meta_path, ['train-other-500'])
        print(res)

    def test_create_urban_meta(self):
        from utils import h_params
        res = create_urban_data_frame(h_params.urban_path)
        print(res["salience"].sample(20))

