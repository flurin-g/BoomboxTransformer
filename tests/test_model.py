import pathlib
from unittest import TestCase

import torch
from hydra.experimental import initialize, compose
from pytorch_lightning.core.lightning import DataLoader

from model import BoomboxTransformer

CWD = pathlib.Path(__file__).parent.parent.absolute()


class TestBoomboxTransformer(TestCase):

    @classmethod
    def setUpClass(cls):
        initialize("../conf")
        cls.settings = compose("config")
        cls.boombox = BoomboxTransformer(cfg=cls.settings, cwd=CWD)

    def test_init(self):
        self.assertIsInstance(self.boombox, BoomboxTransformer)

    def test_train_dataloader(self):
        res = self.boombox.train_dataloader()
        self.assertIsInstance(res, DataLoader)

    def test_pad_audio_seq(self):
        data = [(torch.rand(1, 128, i), torch.rand(1, 128, i)) for i in range(10, 20)]

        x_spectrograms, y_spectrograms = BoomboxTransformer.pad_audio_seq(data)

        self.assertEqual(True, all([x.size() == torch.Size([19, 128]) for x in x_spectrograms]))

    def test_loss(self):
        x = torch.Tensor([[7, 7, 7],
                          [1, 2, 3],
                          [1, 2, 3]])
        y = torch.Tensor([[7, 7, 7],
                          [1, 2, 3],
                          [1, 2, 3]])
        loss: torch.Tensor = torch.nn.MSELoss()(x, y)
        print(loss)
