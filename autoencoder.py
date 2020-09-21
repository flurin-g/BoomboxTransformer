from pathlib import Path

import torchaudio
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader

from noisy_speech import NoisySpeechDataset
from utils import pad_audio_seq


class BoomboxAE(pl.LightningModule):
    def __init__(self, cfg: DictConfig, cwd: Path):
        super().__init__()
        self.model_type = 'Autoencoder'

        self.hparams = cfg.hparams
        self.dataset = cfg.dataset
        self.cwd = cwd

    def train_dataloader(self) -> DataLoader:
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.dataset.sr_libri,
                                                         n_mels=self.hparams["n_mels"])
        dataset = NoisySpeechDataset(cfg=self.dataset,
                                     cwd=self.cwd,
                                     mode="train",
                                     transform=transform)

        return DataLoader(dataset,
                          batch_size=self.hparams["batch_size"],
                          collate_fn=pad_audio_seq)

    def training_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('val_loss', loss)
        return result
