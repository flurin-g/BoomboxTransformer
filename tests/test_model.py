from unittest import TestCase

import torch

from model import NoisySpeechDataset
from utils import h_params, CWD


class TestNoisySpeechDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = NoisySpeechDataset(h_params.libri_meta,
                                         h_params.urban_meta,
                                         CWD,
                                         "train",
                                         h_params.libri_subsets,
                                         None,
                                         1,
                                         h_params.sr_libri,
                                         h_params.sr_urban,
                                         h_params.libri_path,
                                         h_params.urban_path)

    def test_init(self):
        self.assertIsInstance(self.dataset, NoisySpeechDataset)

    def test_len(self):
        print(len(self.dataset))
        self.assertEqual(len(self.dataset), 131502)

    def test_create_pairs(self):
        speech_len = 6
        ambient_len = 2
        oversampling = 1
        pairs = NoisySpeechDataset.create_pairs(speech_len, ambient_len, oversampling)
        self.assertListEqual(pairs, [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1)])

    def test_create_pairs_oversampling(self):
        speech_len = 6
        ambient_len = 2
        oversampling = 2
        pairs = NoisySpeechDataset.create_pairs(speech_len, ambient_len, oversampling)
        self.assertListEqual(pairs,
                             [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1), (0, 1), (1, 0), (2, 1), (3, 0), (4, 1),
                              (5, 0)])

    def test_path_at_libri(self):
        res = self.dataset.path_at(idx=0, of="libri")
        self.assertEqual(res, "train-clean-360/14/212/14-212-0005.flac")

    def test_path_at_urban(self):
        res = self.dataset.path_at(idx=0, of="urban")
        self.assertEqual(res, "audio/fold10/100648-1-4-0.wav")

    def test_get_item(self):
        mix, speech = self.dataset[0]
        self.assertIsInstance(mix, torch.Tensor)
        self.assertIsInstance(speech, torch.Tensor)
        self.assertEqual(mix.shape[1], speech.shape[1])
