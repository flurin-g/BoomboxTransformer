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
    than the desired length, it will be repeated n times and then trimmed
    @param sample: audio-sample to be trimmed
    @param length: desired length
    @return: trimmed audio-sample
    """
    if sample.shape[1] < length:
        times = math.ceil(length / sample.shape[1])
        sample = sample.repeat(1, times)
    return sample[:, :length]


def stereo_to_mono(sample: torch.Tensor) -> torch.Tensor:
    return torch.mean(sample, 0).unsqueeze(0)


def match_sample_rate(sample: torch.Tensor, src_sr: int, trg_sr: int) -> torch.Tensor:
    """
    Check if src_sr and trg_sr are equal, if not, the sample-rate of the sample is adjusted
    @param sample: the audio-sample whose sample-rate is to be changed
    @param src_sr: sample rate of the sample
    @param trg_sr: desired sample rate
    @return: audio-sample with trg_sr as sample rate
    """
    if src_sr != trg_sr:
        return torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=trg_sr)(sample)
    else:
        return sample


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
    sample_b = match_length(sample_b, sample_a.shape[1])
    add = torch.add(sample_a, sample_b)
    return normalize(add)
