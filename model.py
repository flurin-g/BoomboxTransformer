from itertools import chain, permutations
from collections import namedtuple
import pandas as pd
import torch
from pathlib import Path
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset
import torchaudio
from pytorch_lightning.core.lightning import LightningModule, DataLoader

from preprocessing import create_match_urban_to_libri, mix_samples
from utils import select_libri_split


class NoisySpeechDataset(Dataset):

    def __init__(self, libri_meta_path: str, urban_meta_path: str, cwd: Path,
                 mode: str, libri_subsets: list, transform: callable, oversampling: int,
                 libri_sr: int, urban_sr: int):

        oversampling = oversampling if oversampling else 1

        self.cwd = cwd

        libri_meta = pd.read_csv(cwd / libri_meta_path)
        self.libri_df = select_libri_split(libri_meta, mode, libri_subsets)

        urban_meta = pd.read_csv(cwd / urban_meta_path)
        self.urban_df = urban_meta[urban_meta["split"] == mode]

        self.pairs = self.create_pairs(len(self.libri_df.index), len(self.urban_df.index), oversampling)

        self.match_urban_to_libri = create_match_urban_to_libri(urban_sr, libri_sr)

        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def create_pairs(len_speech: int, len_ambient: int, oversampling: int):
        n = len_speech // len_ambient
        speech_idx = list(range(len_speech))
        ambient_idx = list(range(len_ambient))
        ambient_idx = (ambient_idx * n
                       if oversampling < 2
                       else chain.from_iterable([perm * n for perm in permutations(ambient_idx, oversampling)]))
        return list(zip(speech_idx * oversampling, ambient_idx))

    def path_at(self, idx: int, of: str) -> str:
        df, tup_idx = (self.libri_df, 0) if of == "libri" else (self.urban_df, 1)
        return df.at(df.index[self.pairs[tup_idx][idx]], "PATH")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        speech, _ = torchaudio.load(self.path_at(idx, "libri"))
        noise, _ = torchaudio.load(self.path_at(idx, "urban"))

        noise = self.match_urban_to_libri(noise)
        mixed = mix_samples(speech, noise)

        if self.transform:
            mixed = self.transform(mixed)
            speech = self.transform(speech)

        return mixed, speech


class BoomboxTransformer(LightningModule):

    def __init__(self, h_params: namedtuple):
        super().__init__()

        self.h_params = h_params
        # ToDo: maybe even abstract away, by creating a function that takes care of the whole audio stuff
        self.match_8k_libri = create_match_urban_to_libri(h_params.sr_urban, h_params.sr_libri)

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

        # 1. instantiate a transformation function

        # 2. instantiate a dataset instance and pass the transformation function
        # -

        # return a DataLoader
        pass
