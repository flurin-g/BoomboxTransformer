from unittest import TestCase
import numpy as np

from model import NoisySpeechDataset
from utils import h_params, CWD


class TestNoisySpeechDataset(TestCase):
    def test_init(self):
        dataset = NoisySpeechDataset(h_params.libri_meta,
                                     h_params.urban_meta,
                                     CWD)
        self.assertIsInstance(dataset, NoisySpeechDataset)

    def test_len(self):
        dataset = NoisySpeechDataset(h_params.libri_meta,
                                     h_params.urban_meta,
                                     CWD)
        print(len(dataset))
        self.assertEqual(len(dataset), 137876)

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
        self.assertListEqual(pairs, [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1), (0, 1), (1, 0), (2, 1), (3, 0), (4, 1), (5, 0)])


