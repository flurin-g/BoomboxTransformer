from unittest import TestCase
from unittest.case import TestCase

import pandas as pd
import torch
import torchaudio
from hydra.experimental import initialize, compose

from noisy_speech import parse_libri_meta, fetch_files, create_libri_data_frame, create_urban_data_frame, \
    partition_urban_meta, select_libri_split, NoisySpeechDataset
from tests.test_model import CWD


class TestNoisySpeechFunctions(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("../conf")
        cls.settings = compose("config")

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

    def test_partition_urban_meta(self):
        urban_df = create_urban_data_frame(urban_path=self.settings.dataset.urban_path,
                                           cwd=CWD)
        res = partition_urban_meta(urban_df)
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
        initialize("../conf")
        cfg = compose("config")
        cls.dataset = cfg.dataset
        cls.hparams = cfg.hparams
        cls.transform = torchaudio.transforms.MelSpectrogram(sample_rate=cls.dataset.sr_libri,
                                                         n_mels=cls.hparams["n_mels"])

    def setUp(self):
        self.noisy_speech = NoisySpeechDataset(cfg=self.dataset,
                                               cwd=CWD,
                                               mode="train",
                                               transform=self.transform)

    def test_init(self):
        self.assertIsInstance(self.noisy_speech, NoisySpeechDataset)

    def test_len(self):
        self.assertEqual(len(self.noisy_speech), 132553)
        self.noisy_speech.oversampling = 2
        self.assertEqual(len(self.noisy_speech), 265106)

    def test_shift(self):
        self.noisy_speech.oversampling = 3
        res = self.noisy_speech.shift(132553 + 1)
        self.assertEqual(res, 1)

        res = self.noisy_speech.shift(2 * 132553 + 1)
        self.assertEqual(res, 2)

        res = self.noisy_speech.shift(3 * 132553 + 1)
        self.assertEqual(res, 3)

    # ToDo: test comp_urban_idx
    def test_comp_urban_idx_8_3(self):
        self.noisy_speech.speech_len = 8
        self.noisy_speech.noise_len = 3

        self.assertEqual(0, self.noisy_speech.comp_urban_idx(4))
        self.assertEqual(1, self.noisy_speech.comp_urban_idx(8))
        self.assertEqual(2, self.noisy_speech.comp_urban_idx(16))
        self.assertEqual(2, self.noisy_speech.comp_urban_idx(12))
        self.assertEqual(2, self.noisy_speech.comp_urban_idx(15))

    def test_comp_urban_idx_8_3(self):
        self.noisy_speech.speech_len = 8
        self.noisy_speech.noise_len = 4

        self.assertEqual(0, self.noisy_speech.comp_urban_idx(0))
        self.assertEqual(3, self.noisy_speech.comp_urban_idx(7))
        self.assertEqual(3, self.noisy_speech.comp_urban_idx(21))
        self.assertEqual(2, self.noisy_speech.comp_urban_idx(27))
        self.assertEqual(1, self.noisy_speech.comp_urban_idx(30))

    def test_path_at_libri(self):
        res = self.noisy_speech.path_at(idx=0, of="libri")
        self.assertEqual(res, "train-clean-360/14/212/14-212-0005.flac")

    def test_path_at_urban(self):
        res = self.noisy_speech.path_at(idx=0, of="urban")
        self.assertEqual(res, "audio/fold10/100648-1-2-0.wav")

    def test_path_at_urban_shift1(self):
        self.noisy_speech.oversampling = 2
        res = self.noisy_speech.path_at(idx=132553, of="urban")
        self.assertEqual(res, "audio/fold10/100648-1-4-0.wav")

    def test_get_item(self):
        mix, speech = self.noisy_speech[0]
        self.assertIsInstance(mix, torch.Tensor)
        self.assertIsInstance(speech, torch.Tensor)
        self.assertEqual(mix.shape[1], speech.shape[1])

    def test_get_item_last(self):
        mix, speech = self.noisy_speech[2121]
        self.assertIsInstance(mix, torch.Tensor)
        self.assertIsInstance(speech, torch.Tensor)
        self.assertEqual(mix.shape[1], speech.shape[1])
