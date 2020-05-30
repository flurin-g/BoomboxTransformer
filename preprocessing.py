import math

import torch
import torchaudio


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Subtract the mean, and scale to the interval [-1,1]
    """
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()


def match_length(sample: torch.Tensor, length: int) -> torch.Tensor:
    """
    Trims an audio-sample to the desired length, if the sample is shorter
    then the desired length, it will be repeated n times and then trimmed
    @param sample: audio-sample to be trimmed
    @param length: desired length
    @return: trimmed audio-sample
    """
    if len(sample) < length:
        times = math.ceil(length / len(sample))
        sample = sample.repeat(times)
    return sample[:length]


def mix_samples(sample_a: torch.Tensor, sample_b: torch.Tensor, trim_to: str = "a") -> torch.Tensor:
    """
    @param sample_a: first audio-sample
    @param sample_b: second audio-sample
    @param trim_to: resulting length will be length of sample_a or sample_b,
                    if the sample not specified is shorter than the specified one,
                    it will be repeated to match the specified samples length
    @return: sum of both samples, normalized to avoid clipping
    """
    if trim_to == "b":
        sample_a, sample_b = sample_b, sample_a
    sample_b = match_length(sample_b, len(sample_a))
    add = torch.add(sample_a, sample_b)
    return normalize(add)
