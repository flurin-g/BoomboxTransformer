from unittest import TestCase

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
        self.assertEqual(len(dataset), 143679)
