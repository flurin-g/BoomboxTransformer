from unittest import TestCase
from preprocessing import match_length, mix_samples, stereo_to_mono

import torch
import torchaudio


class Test(TestCase):
    def test_match_length(self):
        sample = torch.ones(1, 4)
        res = match_length(sample, 3)
        print(res.shape)
        self.assertEqual(res.shape, torch.Size([1, 3]))

    def test_match_length_repeat(self):
        sample = torch.tensor([[.1, .2, .3, .4]])
        res = match_length(sample, 6)
        self.assertTrue(torch.equal(res, torch.tensor([[.1, .2, .3, .4, .1, .2]])))

    def test_stereo_to_mono(self):
        sample = torch.tensor([[.1, .1, .1],
                               [.3, .3, .3]])
        res = stereo_to_mono(sample)
        print(res)
        self.assertTrue(torch.equal(res, torch.tensor([[0.2, 0.2, 0.2]])))

    def test_mix_samples(self):
        speech = torch.tensor([[-.5, .5, -.5, .5]])
        noise = torch.tensor([[-.2, .2, -.2, .2]])
        res = mix_samples(speech, noise)
        self.assertTrue(torch.equal(res, torch.tensor([[-1., 1., -1., 1.]])))

    def test_mix_samples_overflow(self):
        speech = torch.tensor([[-.6, .6, -.6, .6]])
        noise = torch.tensor([[-.6, .6, -.6, .6]])
        res = mix_samples(speech, noise)
        self.assertTrue(torch.equal(res, torch.tensor([[-1., 1., -1., 1.]])))

    def test_mix_samples_trim_to_a(self):
        speech = torch.tensor([[-.2, .2, -.2, .2]])
        noise = torch.tensor([[-.6, .6]])
        res = mix_samples(speech, noise)
        self.assertTrue(torch.equal(res, torch.tensor([[-1., 1., -1., 1.]])))

    def test_mix_samples_trim_to_b(self):
        speech = torch.tensor([[-.2, .2, -.2, .2]])
        noise = torch.tensor([[-.6, .6]])
        res = mix_samples(speech, noise, "b")
        print(res)
        self.assertTrue(torch.equal(res, torch.tensor([[-1., 1.]])))


