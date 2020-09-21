import pathlib
from unittest import TestCase
from hydra.experimental import initialize, compose

from autoencoder import BoomboxAE

CWD = pathlib.Path(__file__).parent.parent.absolute()


class TestBoomboxAE(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("../conf")
        cls.settings = compose("config")
        cls.boombox = BoomboxAE(cfg=cls.settings, cwd=CWD)

    def test_training_step(self):
        self.boombox.training_step()
