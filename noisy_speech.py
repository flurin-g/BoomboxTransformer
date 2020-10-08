from math import gcd
import itertools
import pathlib
import re
from collections import namedtuple
from pathlib import Path
from typing import List, Iterator, Tuple

import pandas as pd
import torch
import torchaudio
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torchvision.datasets.utils import download_and_extract_archive, download_file_from_google_drive, extract_archive
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from audio_processing import create_match_urban_to_libri, mix_samples


def clean_line_libri(line: str) -> List[str]:
    """
    Clean a single line of the librispeech SPEAKERS.txt file
    """
    line = line.replace("|CBW|Simon", "CBW Simon")
    line = re.sub(r"(;|\s*\|\s*|\n)", "%", line)
    return [token for token in line.split("%") if token]


def parse_libri_meta(meta_path: str, cwd: Path) -> list:
    """
    Retains only header and data of the librispeech SPEAKERS.txt file

    @param cwd: working directory relative to project root
    @param meta_path: path of SPEAKERS.txt
    @return: A list of lists, where the lists contain a tokenized line
    """
    with open(cwd / meta_path) as file:
        res = [clean_line_libri(line) for line in file if not line.startswith(("; ", ";\n"))]
    return res


def fetch_files(row: namedtuple, data_path: str, cwd: Path) -> Iterator[Tuple[str, str, str]]:
    """
    Given row of DataFrame and the path of librispeech, construct paths for all files
    @param cwd: working directory relative to project root
    @param row: NamedTuple from df.iterrows, with keys: SUBSET, ID
    @param data_path: path of librispeech
    @return: list of tuples with (SUBSET, ID, PATH)
    """
    path = pathlib.Path(cwd / data_path / row.SUBSET) / str(row.ID)
    fetch_file_paths = lambda x: [(str(row.SUBSET),
                                   str(row.ID),
                                   str(file.relative_to(cwd / data_path)))
                                  for file in x.glob("*.flac")]
    return itertools.chain.from_iterable([fetch_file_paths(directory) for directory in path.iterdir()])


def create_libri_data_frame(root_path: str, meta_path: str, cwd: Path, subsets: list) -> pd.DataFrame:
    """
    @param subsets: list of subsets to be excluded
    @return: DataFrame containing meta-data and corresponding file-path
    """
    speakers_parse = parse_libri_meta(meta_path, cwd)
    df_raw = pd.DataFrame(speakers_parse[1:], columns=speakers_parse[0])
    if subsets:
        df_raw = df_raw[df_raw.SUBSET.isin(subsets)]
    paths = itertools.chain.from_iterable([fetch_files(row, root_path, cwd) for row in df_raw.itertuples()])
    df_paths = pd.DataFrame.from_records(paths, columns=['SUBSET', 'ID', 'PATH'])
    return df_raw.merge(df_paths, on=["SUBSET", "ID"])


def create_libri_meta(libri_path: str, libri_meta_path: str, file_name: str, subsets: list, cwd: Path) -> None:
    libri_df = create_libri_data_frame(libri_path, libri_meta_path, cwd, subsets)
    libri_df.to_csv(path_or_buf=file_name, index=False)


def create_urban_data_frame(urban_path: str, cwd: Path, background_only: bool = True) -> pd.DataFrame:
    """
    Adds the filepath relative to the data-folder to the urban-sound DataFrame
    @param cwd: working directory relative to project root
    @param urban_path: base director of UrbanSound8K
    @param background_only: selects only rows where salience == 2
    @return: File is written to disk
    """
    urban = pd.read_csv(cwd / urban_path / "metadata" / "UrbanSound8K.csv")
    if background_only:
        urban = urban[urban["salience"] == 2]
    urban["PATH"] = urban.apply(lambda row: f'audio/fold{row.fold}/{row.slice_file_name}', axis=1)
    return urban


def partition_urban_meta(urban_meta: pd.DataFrame) -> pd.DataFrame:
    train, tmp = train_test_split(urban_meta, test_size=0.3, stratify=urban_meta["class"], random_state=42)
    dev, test = train_test_split(tmp, test_size=0.5, stratify=tmp["class"], random_state=42)
    urban_meta.loc[train.index, "split"] = "train"
    urban_meta.loc[dev.index, "split"] = "dev"
    urban_meta.loc[test.index, "split"] = "test"
    return urban_meta


def select_libri_split(df: pd.DataFrame, mode: str, subsets: list) -> pd.DataFrame:
    assert mode in ["train", "dev", "test"], "mode must be one of train, dev, test"
    subsets = [subset for subset in subsets if mode in subset]
    return df[df.SUBSET.isin(subsets)]


def is_prime(a):
    return all(a % i for i in range(2, a))


class NoisySpeechDataset(Dataset):

    def __init__(self, cfg: DictConfig, cwd: Path, mode: str, transform: callable):
        self.cfg = cfg
        self.cwd = cwd
        self.mode = mode
        self.transform = transform

        libri_meta = pd.read_csv(cwd / self.cfg.libri_meta)
        self.libri_df = select_libri_split(libri_meta, self.mode, self.cfg.libri_subsets)

        urban_meta = pd.read_csv(cwd / self.cfg.urban_meta)
        self.urban_df = urban_meta[urban_meta["split"] == self.mode].reset_index(drop=True)

        self.match_urban_to_libri = create_match_urban_to_libri(self.cfg.sr_urban, self.cfg.sr_libri)

        self.speech_len = len(self.libri_df.index)
        self.noise_len = len(self.urban_df.index)
        self.oversampling = None
        self.compute_oversampling(self.cfg.oversampling)

    def __len__(self) -> int:
        return self.speech_len * self.oversampling

    def compute_oversampling(self, oversampling: int) -> None:
        """
        Makes sure no out of bounds occurs when shifting
        @param oversampling: number of speech samples a given
            noise sample should be paired with
        @return: The maximum number of shifts for noise_idx
            relative to speech_idx
        """
        if oversampling > 1:
            self.oversampling = oversampling if oversampling < self.noise_len else self.noise_len - 1
        else:
            self.oversampling = 1

    def shift(self, idx: int) -> int:
        """
        Computes the offset for noise samples, such that when all speech
            samples have been paired, the noise samples are offset by +1 relative
            to the previous pairings
        @param idx: absolute idx of batch
        @return: offset for the noise samples
        """
        return idx // self.speech_len

    def comp_urban_idx(self, idx: int):
        """
        Assures unique pairings between speech samples and noise samples
            for the current batch
        @param idx: absolute idx of batch
        @return: idx for self.urban_df
        """
        return ((idx % self.speech_len) + self.shift(idx)) % self.noise_len

    @staticmethod
    def create_libri_meta(libri_path: str, libri_meta_path: str, file_name: str, cwd: Path, subsets: list) -> None:
        libri_df = create_libri_data_frame(libri_path, libri_meta_path, cwd, subsets)
        libri_df.to_csv(path_or_buf=cwd / file_name, index=False)

    @staticmethod
    def create_urban_meta(urban_path: str, file_name: str, cwd: Path) -> None:
        urban_df = create_urban_data_frame(urban_path, cwd)
        urban_df = partition_urban_meta(urban_df)
        urban_df.to_csv(path_or_buf=cwd / file_name, index=False)

    @staticmethod
    def download_libri(cfg: DictConfig, cwd: Path) -> None:
        for subset in cfg.libri_subsets:
            download_and_extract_archive(url=cfg.libri_urls[subset],
                                         download_root=cwd / "data",
                                         filename=subset + ".tar.gz",
                                         remove_finished=True)

    @staticmethod
    def download_urban(cfg: DictConfig, cwd: Path) -> None:
        urban_path = Path(cfg.urban_path)
        Path.mkdir(cwd / urban_path.name, parents=True, exist_ok=True)
        download_file_from_google_drive(file_id=cfg.urban_url,
                                        root=cwd / urban_path.parent,
                                        filename="UrbanSound8K.tar.gz")
        extract_archive(from_path=str(cwd / urban_path.parent / "UrbanSound8K.tar.gz"),
                        remove_finished=True)

    def path_at(self, idx: int, of: str) -> str:
        df, idx_val = (self.libri_df, idx % self.speech_len) \
            if of == "libri" else (self.urban_df, self.comp_urban_idx(idx))
        return df.at[df.index[idx_val], "PATH"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        speech, _ = torchaudio.load(self.cwd / self.cfg.libri_path / self.path_at(idx, "libri"))
        noise, _ = torchaudio.load(self.cwd / self.cfg.urban_path / self.path_at(idx, "urban"))

        noise = self.match_urban_to_libri(noise)
        mixed = mix_samples(speech, noise)

        if self.transform:
            mixed = self.transform(mixed)
            speech = self.transform(speech)

        return mixed, speech


class NoisySpeechModule(LightningDataModule):

    def __init__(self, cfg: DictConfig, batch_size: int, n_mels: int, cwd: Path):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.cwd = cwd

        self.train: Dataset = None
        self.val: Dataset = None

    def prepare_data(self) -> None:
        if self.cfg.download:
            NoisySpeechDataset.download_libri(cfg=self.cfg, cwd=self.cwd)
            NoisySpeechDataset.download_urban(cfg=self.cfg, cwd=self.cwd)

        if self.cfg.create_meta:
            NoisySpeechDataset.create_libri_meta(libri_path=self.cfg.libri_path,
                                                 libri_meta_path=self.cfg.libri_speakers,
                                                 file_name=self.cfg.libri_meta,
                                                 cwd=self.cwd,
                                                 subsets=self.cfg.libri_subsets)

            NoisySpeechDataset.create_urban_meta(urban_path=self.cfg.urban_path,
                                                 file_name=self.cfg.urban_meta,
                                                 cwd=self.cwd)

    def setup(self, stage=None):
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.cfg.sr_libri,
                                                         n_mels=self.n_mels)

        self.train = NoisySpeechDataset(cfg=self.cfg,
                                        cwd=self.cwd,
                                        mode="train",
                                        transform=transform)
        self.val = NoisySpeechDataset(cfg=self.cfg,
                                      cwd=self.cwd,
                                      mode="dev",
                                      transform=transform)

    @staticmethod
    def pad_audio_seq(data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_spectrograms = []
        y_spectrograms = []
        input_lengths = []
        for (mixed, speech) in data:
            x_spectrograms.append(mixed.squeeze(0).transpose(0, 1))
            y_spectrograms.append(speech.squeeze(0).transpose(0, 1))
            input_lengths.append(mixed.shape[0])

        # sort by sequence length, for pad_sequence to work
        ixs = sorted((ix for ix in range(len(input_lengths))), key=lambda ix: input_lengths[ix], reverse=True)
        x_spectrograms = [x_spectrograms[ix] for ix in ixs]
        y_spectrograms = [y_spectrograms[ix] for ix in ixs]

        x_spectrograms = torch.nn.utils.rnn.pad_sequence(x_spectrograms, batch_first=True)
        y_spectrograms = torch.nn.utils.rnn.pad_sequence(y_spectrograms, batch_first=True)
        # ToDo: check if needed:
        # seq_len = x_spectrograms.shape[0]
        return x_spectrograms, y_spectrograms  # , seq_len

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          collate_fn=self.pad_audio_seq)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          collate_fn=self.pad_audio_seq)

    def test_dataloader(self):
        # ToDo: Implement
        pass
