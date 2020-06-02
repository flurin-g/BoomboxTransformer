from collections import namedtuple
from typing import List, Iterator, Tuple
import itertools

import pandas as pd
import yaml
import pathlib
import re

CWD = pathlib.Path(__file__).parent.absolute()


def convert(dictionary: dict) -> namedtuple:
    """
    @param dictionary: dictionary containing hyper-parameters, can be nested
    @return: namedtuple of type HyperParameters
    """
    return namedtuple('HyperParameters', dictionary.keys())(**dictionary)


def load_params(path: str) -> namedtuple:
    """
    @param path: path of the yaml-file containing the hyper-parameters
    @return: namedtuple of type HyperParameters
    """
    with open(CWD / path) as file:
        param_dict = yaml.full_load(file)
    return convert(param_dict)


h_params = load_params("hyper_parameters.yml")


def clean_line_libri(line: str) -> List[str]:
    """
    Clean a single line of the librispeech SPEAKERS.txt file
    """
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


def create_data_frame(root_path: str, meta_path: str) -> pd.DataFrame:
    speakers_parse = parse_libri_meta(meta_path)
    df_raw = pd.DataFrame(speakers_parse[1:], columns=speakers_parse[0])
    paths = itertools.chain.from_iterable([fetch_files(row, root_path) for row in df_raw.itertuples()])
    df_paths = pd.DataFrame.from_records(paths, columns=['SUBSET', 'ID', 'PATH'])
    return df_raw.merge(df_paths, on=["SUBSET", "ID"])
