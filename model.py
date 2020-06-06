from collections import namedtuple
import pandas as pd
import torch
from pathlib import Path
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset
import torchaudio
from pytorch_lightning.core.lightning import LightningModule, DataLoader

from preprocessing import create_match_8k_libri


class NoisySpeechDataset(Dataset):

    def __init__(self, libri_meta: str, urban_meta: str, cwd: Path, transform: callable = None):
        self.cwd = cwd
        self.libri_meta = pd.read_csv(cwd / libri_meta)
        self.urban_meta = pd.read_csv(cwd / urban_meta)

        self.transform = transform

    def __len__(self):
        return len(self.libri_meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        speech, _ = torchaudio.load(self.libri_meta.at(self.libri_meta.index[idx], 'PATH'))
        noise, _ = torchaudio.load(self.urban_meta.at)


class BoomboxTransformer(LightningModule):

    def __init__(self, h_params: namedtuple):
        super().__init__()

        self.h_params = h_params
        # ToDo: maybe even abstract away, by creating a function that takes care of the whole audio stuff
        self.match_8k_libri = create_match_8k_libri(h_params.sr_urban, h_params.sr_libri)

    def forward(self, x):
        pass

    def prepare_data(self) -> None:
        # stuff here is done once at the very beginning of training
        # before any distributed training starts

        # download stuff
        # save to disk
        # etc...
        pass

    def train_dataloader(self) -> DataLoader:
        # data transforms
        # dataset creation
        # return a DataLoader
        pass
