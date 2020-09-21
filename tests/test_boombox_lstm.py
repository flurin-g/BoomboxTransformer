import pathlib
from unittest import TestCase

import torch
from hydra.experimental import initialize, compose

from boombox_lstm import BoomboxLSTM


CWD = pathlib.Path(__file__).parent.parent.absolute()


class TestBoomboxLSTM(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("../conf")
        cls.settings = compose("config")
        cls.boombox = BoomboxLSTM(cfg=cls.settings, cwd=CWD)

    def test_forward(self):
        x = torch.randn(1, 512, 128)  # batch, seq, n_mels
        output, hidden = self.boombox(x)

        print(f'output size: {output.size()}\n\
        hidden size: {hidden[0].size()}\n\
        context size: {hidden[1].size()}')
