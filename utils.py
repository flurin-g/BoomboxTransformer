from typing import Tuple

import torch
from torch import nn


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

    x_spectrograms = nn.utils.rnn.pad_sequence(x_spectrograms, batch_first=True)
    y_spectrograms = nn.utils.rnn.pad_sequence(y_spectrograms, batch_first=True)
    # ToDo: check if needed:
    # seq_len = x_spectrograms.shape[0]
    return x_spectrograms, y_spectrograms  # , seq_len