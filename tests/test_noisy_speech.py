from unittest import TestCase
from unittest.case import TestCase

import pandas as pd
import torch
from hydra.experimental import initialize, compose

from noisy_speech import parse_libri_meta, fetch_files, create_libri_data_frame, create_urban_data_frame, \
    partition_urban_meta, select_libri_split, NoisySpeechDataset
from tests.test_model import CWD


class TestNoisySpeechFunctions(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("../")
        cls.settings = compose("parameters.yml")

    def test_parse_libri_meta(self):
        res = parse_libri_meta("tests/test_data/libri_dir_struct/SPEAKERS.txt", CWD)
        self.assertEqual(res, [['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'],
                               ['84', 'F', 'dev-clean', '8.02', 'Christie Nowak'],
                               ['174', 'M', 'dev-clean', '8.04', 'Peter Eastman']])

    def test_parse_libri_meta_cbw_simon(self):
        res = parse_libri_meta("tests/test_data/cbw_simon.txt", CWD)
        self.assertListEqual(res, [['60', 'M', 'train-clean-100', '20.18', 'CBW Simon'],
                                   ['61', 'M', 'test-clean', '8.08', 'Paul-Gabriel Wiener']])

    def test_fetch_files(self):
        df = pd.read_csv("../tests/test_data/SPEAKERS.csv")
        row = next(df.itertuples())
        res = fetch_files(row=row, data_path="tests/test_data/libri_dir_struct", cwd=CWD)
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
        res = create_libri_data_frame(root_path, meta_path, CWD, ['dev-clean'])
        print(res)

    def test_partition_urban_meta(self):
        urban_df = create_urban_data_frame(urban_path=self.settings.dataset.urban_path,
                                           cwd=CWD)
        res = partition_urban_meta(urban_df)
        print(res.groupby("split")["class"].count())
        self.assertEqual(2121, res.groupby("split")["class"].count().train)
        self.assertEqual(454, res.groupby("split")["class"].count().dev)
        self.assertEqual(455, res.groupby("split")["class"].count().test)

    def test_create_urban_meta(self):
        res = create_urban_data_frame(urban_path=self.settings.dataset.urban_path,
                                      cwd=CWD)
        assert (res["salience"] == 2).all()

    def test_select_libri_split_train(self):
        libri_meta = pd.read_csv(CWD / self.settings.dataset.libri_meta)
        df = select_libri_split(libri_meta, "train", self.settings.dataset.libri_subsets)
        assert df["SUBSET"].astype(str).str.contains("train").all()

    def test_select_libri_split_dev(self):
        libri_meta = pd.read_csv(CWD / self.settings.dataset.libri_meta)
        df = select_libri_split(libri_meta, "dev", self.settings.dataset.libri_subsets)
        assert df["SUBSET"].astype(str).str.contains("dev").all()

    def test_select_libri_split_illegal_mode(self):
        libri_meta = pd.read_csv(CWD / self.settings.dataset.libri_meta)
        with self.assertRaises(AssertionError):
            select_libri_split(libri_meta, "spam", self.settings.dataset.libri_subsets)


class TestNoisySpeechDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        initialize("../")
        settings = compose("parameters.yml")
        dataset = settings.dataset
        cls.dataset = NoisySpeechDataset(dataset.libri_meta,
                                         dataset.urban_meta,
                                         CWD,
                                         "train",
                                         dataset.libri_subsets,
                                         dataset.libri_urls,
                                         None,
                                         1,
                                         dataset.sr_libri,
                                         dataset.sr_urban,
                                         dataset.libri_path,
                                         dataset.urban_path,
                                         dataset.urban_url)

    def test_init(self):
        self.assertIsInstance(self.dataset, NoisySpeechDataset)

    def test_len(self):
        print(len(self.dataset))
        self.assertEqual(len(self.dataset), 132553)
        self.dataset.oversampling = 2
        print(len(self.dataset))
        self.assertEqual(len(self.dataset), 265106)

    def test_shift_no_oversampling(self):
        self.dataset.oversampling = 1
        res = self.dataset.shift(132553 + 1)
        print(res)
        self.assertEqual(res, 0)

    def test_shift_oversampling_0(self):
        self.dataset.oversampling = 2
        res = self.dataset.shift(132553 - 1)
        print(res)
        self.assertEqual(res, 0)

    def test_shift_oversampling_1(self):
        self.dataset.oversampling = 2
        res = self.dataset.shift(132553 + 0)
        self.assertEqual(res, 1)
        res = self.dataset.shift(132553 + 1)
        print(res)
        self.assertEqual(res, 1)

    def test_shift_oversampling_2(self):
        self.dataset.oversampling = 2
        res = self.dataset.shift(132553 * 2)
        print(res)
        self.assertEqual(res, 2)

    def test_path_at_libri(self):
        res = self.dataset.path_at(idx=0, of="libri")
        self.assertEqual(res, "train-clean-360/14/212/14-212-0005.flac")

    def test_path_at_urban(self):
        res = self.dataset.path_at(idx=0, of="urban")
        self.assertEqual(res, "audio/fold10/100648-1-2-0.wav")

    def test_path_at_urban_shift1(self):
        self.dataset.oversampling = 2
        res = self.dataset.path_at(idx=132553, of="urban")
        print(res)
        self.assertEqual(res, "audio/fold10/100648-1-4-0.wav")

    def test_get_item(self):
        mix, speech = self.dataset[0]
        self.assertIsInstance(mix, torch.Tensor)
        self.assertIsInstance(speech, torch.Tensor)
        self.assertEqual(mix.shape[1], speech.shape[1])
