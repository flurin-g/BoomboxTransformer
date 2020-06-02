import pandas as pd
from utils import create_data_frame


def create_libri_meta(libri_path: str, libri_meta_path: str, file_name: str = "libri_meta.csv") -> None:
    libri_df = create_data_frame(libri_path, libri_meta_path)
    libri_df.to_csv(path_or_buf=file_name, index=False)


if __name__ == "__main__":
    # ToDo: include arg-parse
    create_libri_meta(libri_path="data/LibriSpeech",
                      libri_meta_path="data/LibriSpeech/SPEAKERS.TXT")