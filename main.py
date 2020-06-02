import argparse
import pandas as pd
from utils import create_data_frame


def create_libri_meta(libri_path: str, libri_meta_path: str, file_name: str = "libri_meta.csv") -> None:
    libri_df = create_data_frame(libri_path, libri_meta_path)
    libri_df.to_csv(path_or_buf=file_name, index=False)


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('--task', type=str, choices=['train', 'create-meta'], required=True,
                        help='Choose task to run')

    return parser


def main():
    parser = argparse.ArgumentParser(description='PyTorch BoomboxTransformer')
    parser = parse_args(parser)
    args, unknown_args = parser.parse_known_args()

    if "create-meta" in args.task:
        create_libri_meta(libri_path="data/LibriSpeech",
                          libri_meta_path="data/LibriSpeech/SPEAKERS.TXT")


if __name__ == '__main__':
    main()

