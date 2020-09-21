from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


class BoomboxLSTM (pl.LightningModule):
    def __init__(self, cfg: DictConfig, cwd: Path):
        super().__init__()
        self.model_type = 'LSTM'

        self.cfg = cfg
        self.dataset = cfg.dataset
        self.hparams = cfg.hparams
        self.cwd = cwd

        self.lstm = nn.LSTM(input_size=self.hparams.n_mels,
                            hidden_size=self.hparams.hidden_dim,
                            num_layers=self.hparams.num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(in_features=self.hparams.n_mels * 2,
                            out_features=self.hparams.n_mels)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        out = self.fc(lstm_out)

        return out

    def configure_optimizers(self) -> callable:
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_pred = self(x)

        loss = F.mse_loss(y_pred, y)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_pred = self(x)

        val_loss = F.mse_loss(y_pred, y)

        result = pl.EvalResult(checkpoint_on=val_loss)
        result.log('val_loss', val_loss)
        return result
