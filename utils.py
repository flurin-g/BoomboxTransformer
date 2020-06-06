from collections import namedtuple
from typing import List, Iterator, Tuple, NamedTuple
import itertools

import pandas as pd
import yaml
import pathlib
import re

CWD = pathlib.Path(__file__).parent.absolute()


class HyperParameters(NamedTuple):
    batch_size: int
    shuffle: bool
    n_hid: int
    n_mels: int
    n_head: int
    dropout: float
    n_layers: int
    n_epochs: int
    lr: float
    gamma: float
    sr_libri: int
    sr_urban: int
    libri_path: str
    libri_drop_subsets: list
    libri_speakers: str
    libri_meta: str
    urban_path: str
    urban_meta: str

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**params)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as file:
            param_dict = yaml.full_load(file)
        return cls(**param_dict)


h_params = HyperParameters.from_yaml(CWD / "hyper_parameters.yml")


def clean_line_libri(line: str) -> List[str]:
    """
    Clean a single line of the librispeech SPEAKERS.txt file
    """
    line = line.replace("|CBW|Simon", "CBW Simon")
    line = re.sub(r"(;|\s*\|\s*|\n)", "%", line)
    return [token for token in line.split("%") if token]


def parse_libri_meta(meta_path: str) -> list:
    """
    Retains only header and data of the librispeech SPEAKERS.txt file

    @param meta_path: path of SPEAKERS.txt
    @return: A list of lists, where the lists contain a tokenized line
    """
    with open(CWD / meta_path) as file:
        res = [clean_line_libri(line) for line in file if not line.startswith(("; ", ";\n"))]
    return res


def fetch_files(row: namedtuple, data_path: str) -> Iterator[Tuple[str, str, str]]:
    """
    Given row of DataFrame and the path of librispeech, construct paths for all files
    @param row: NamedTuple from df.iterrows, with keys: SUBSET, ID
    @param data_path: path of librispeech
    @return: list of tuples with (SUBSET, ID, PATH)
    """
    path = pathlib.Path(CWD / data_path / row.SUBSET) / str(row.ID)
    fetch_file_paths = lambda x: [(str(row.SUBSET),
                                   str(row.ID),
                                   str(file.relative_to(CWD / data_path)))
                                  for file in x.glob("*.flac")]
    return itertools.chain.from_iterable([fetch_file_paths(directory) for directory in path.iterdir()])


def create_libri_data_frame(root_path: str, meta_path: str, drop_subsets: list) -> pd.DataFrame:
    """
    @param drop_subsets: list of subsets to be excluded
    @return: DataFrame containing meta-data and corresponding file-path
    """
    speakers_parse = parse_libri_meta(meta_path)
    df_raw = pd.DataFrame(speakers_parse[1:], columns=speakers_parse[0])
    if drop_subsets:
        df_raw = df_raw[~df_raw.SUBSET.isin(drop_subsets)]
    paths = itertools.chain.from_iterable([fetch_files(row, root_path) for row in df_raw.itertuples()])
    df_paths = pd.DataFrame.from_records(paths, columns=['SUBSET', 'ID', 'PATH'])
    return df_raw.merge(df_paths, on=["SUBSET", "ID"])


def create_libri_meta(libri_path: str, libri_meta_path: str, file_name: str, drop_subsets: list) -> None:
    libri_df = create_libri_data_frame(libri_path, libri_meta_path, drop_subsets)
    libri_df.to_csv(path_or_buf=file_name, index=False)


def create_urban_data_frame(urban_path: str, background_only: bool = True) -> pd.DataFrame:
    """
    Adds the filepath relative to the data-folder to the urban-sound DataFrame
    @param urban_path: base director of UrbanSound8K
    @param background_only: selects only rows where salience == 2
    @return: File is written to disk
    """
    urban = pd.read_csv(CWD / urban_path / "metadata" / "UrbanSound8K.csv")
    if background_only:
        urban = urban[urban["salience"] == 2]
    urban["PATH"] = urban.apply(lambda row: f'audio/fold{row.fold}/{row.slice_file_name}', axis=1)
    return urban


def create_urban_meta(urban_path: str, file_name: str = h_params.urban_meta) -> None:
    urban_df = create_urban_data_frame(urban_path)
    urban_df.to_csv(path_or_buf=file_name, index=False)