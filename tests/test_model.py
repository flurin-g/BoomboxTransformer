import pathlib
from unittest import TestCase

from pytorch_lightning.core.lightning import DataLoader

from model import BoomboxTransformer

CWD = pathlib.Path(__file__).parent.parent.absolute()


class TestBoomboxTransformer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.boombox = BoomboxTransformer(settings, CWD, h_params)

    def test_init(self):
        self.assertIsInstance(self.boombox, BoomboxTransformer)

    def test_train_dataloader(self):
        res = self.boombox.train_dataloader()
        self.assertIsInstance(res, DataLoader)

