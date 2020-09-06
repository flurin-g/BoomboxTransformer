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
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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

    def __init__(self, libri_meta_path: str, urban_meta_path: str, cwd: Path,
                 mode: str, libri_subsets: list, transform: callable, oversampling: int,
                 libri_sr: int, urban_sr: int, libri_path: str, urban_path: str):

        self.cwd = cwd

        libri_meta = pd.read_csv(cwd / libri_meta_path)
        self.libri_df = select_libri_split(libri_meta, mode, libri_subsets)

        urban_meta = pd.read_csv(cwd / urban_meta_path)
        self.urban_df = urban_meta[urban_meta["split"] == mode]

        self.match_urban_to_libri = create_match_urban_to_libri(urban_sr, libri_sr)

        self.transform = transform

        self.libri_path = libri_path
        self.urban_path = urban_path

        self.speech_len = len(self.libri_df.index)
        self.noise_len = len(self.urban_df.index)
        self.oversampling = None
        self.compute_oversampling(oversampling)
        self.comp_urban_idx = self.create_comp_urban_idx()

    def __len__(self):
        return self.speech_len * self.oversampling

    def compute_oversampling(self, oversampling):
        if oversampling > 1:
            self.oversampling = oversampling if oversampling < self.noise_len else self.noise_len - 1
            self.speech_len = self.speech_len - 1 if is_prime(self.speech_len) else self.speech_len
            self.noise_len = gcd(self.speech_len, self.noise_len)
        else:
            self.oversampling = 1

    def shift(self, idx):
        if self.oversampling == 1:
            shift = 0
        else:
            shift = idx // self.speech_len
        return shift

    def create_comp_urban_idx(self):
        if self.oversampling >= 2:
            def comp_urban_idx(idx):
                return (idx + self.shift(idx)) % self.speech_len % self.noise_len
        else:
            def comp_urban_idx(idx):
                return (idx + self.shift(idx)) % self.speech_len

        return comp_urban_idx

    @staticmethod
    def create_libri_meta(libri_path: str, libri_meta_path: str, file_name: str, cwd: Path, subsets: list) -> None:
        libri_df = create_libri_data_frame(libri_path, libri_meta_path, cwd, subsets)
        libri_df.to_csv(path_or_buf=cwd / file_name, index=False)

    @staticmethod
    def create_urban_meta(urban_path: str, file_name: str, cwd: Path) -> None:
        urban_df = create_urban_data_frame(urban_path, cwd)
        urban_df = partition_urban_meta(urban_df)
        urban_df.to_csv(path_or_buf=cwd / file_name, index=False)

    def path_at(self, idx: int, of: str) -> str:
        df, idx_val = (self.libri_df, idx % self.speech_len) \
            if of == "libri" else (self.urban_df, self.comp_urban_idx(idx))
        return df.at[df.index[idx_val], "PATH"]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        speech, _ = torchaudio.load(self.cwd / self.libri_path / self.path_at(idx, "libri"))
        noise, _ = torchaudio.load(self.cwd / self.urban_path / self.path_at(idx, "urban"))

        noise = self.match_urban_to_libri(noise)
        mixed = mix_samples(speech, noise)

        if self.transform:
            mixed = self.transform(mixed)
            speech = self.transform(speech)

        return mixed, speech
