import math
from pathlib import Path
from typing import Tuple, Union, List

import torch
import torchaudio
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule, DataLoader
from torch import nn
from torch.nn import functional as F, Dropout, TransformerEncoderLayer, TransformerEncoder, \
    TransformerDecoderLayer, TransformerDecoder
from torch.optim import Adam

from noisy_speech import NoisySpeechDataset


class PositionalEncoding(LightningModule):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BoomboxTransformer(LightningModule):

    def __init__(self, cfg: DictConfig, cwd: Path):
        super().__init__()

        #self.example_input_array = torch.rand(2, 1, 1024, 128)

        self.cfg = cfg
        self.dataset = cfg.dataset
        self.hparams = cfg.hparams
        self.cwd = cwd

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.hparams["n_mels"], self.hparams["dropout"])

        encoder_layers = TransformerEncoderLayer(self.hparams["n_mels"],
                                                 self.hparams["n_head"],
                                                 self.hparams["n_hid"],
                                                 self.hparams["dropout"])
        self.encoder = TransformerEncoder(encoder_layers, self.hparams["n_layers"])

        decoder_layers = TransformerDecoderLayer(self.hparams["n_mels"],
                                                 self.hparams["n_head"],
                                                 self.hparams["n_hid"],
                                                 self.hparams["dropout"])
        self.decoder = TransformerDecoder(decoder_layers, self.hparams["n_layers"])

        self.init_weights()

    @staticmethod
    def _generate_square_subsequent_mask(sz) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        self.encoder.weight.data.uniform_(-self.hparams.weight_init_range,
                                          self.hparams.weight_init_range)
        self.decoder.bias.data.zero_() # ToDo: what is this for?
        self.decoder.weight.data.uniform_(-self.hparams.weight_init_range,
                                          self.hparams.weight_init_range)

    def prepare_data(self) -> None:
        if self.dataset.download:
            NoisySpeechDataset.download_libri(cfg=self.dataset, cwd=self.cwd)
            NoisySpeechDataset.download_urban(cfg=self.dataset, cwd=self.cwd)

        if self.dataset.create_meta:
            NoisySpeechDataset.create_libri_meta(libri_path=self.dataset.libri_path,
                                                 libri_meta_path=self.dataset.libri_speakers,
                                                 file_name=self.dataset.libri_meta,
                                                 cwd=self.cwd,
                                                 subsets=self.dataset.libri_subsets)

            NoisySpeechDataset.create_urban_meta(urban_path=self.dataset.urban_path,
                                                 file_name=self.dataset.urban_meta,
                                                 cwd=self.cwd)

    @staticmethod
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
        return x_spectrograms, y_spectrograms #, seq_len

    def train_dataloader(self) -> DataLoader:
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.dataset.sr_libri,
                                                         n_mels=self.hparams["n_mels"])
        dataset = NoisySpeechDataset(cfg=self.dataset,
                                     cwd=self.cwd,
                                     mode="train",
                                     transform=transform)

        return DataLoader(dataset,
                          batch_size=self.hparams["batch_size"],
                          collate_fn=self.pad_audio_seq)

    def val_dataloader(self) -> DataLoader:
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.dataset.sr_libri,
                                                         n_mels=self.hparams["n_mels"])
        dataset = NoisySpeechDataset(cfg=self.dataset,
                                     cwd=self.cwd,
                                     mode="dev",
                                     transform=transform)

        return DataLoader(dataset,
                          batch_size=self.hparams["batch_size"],
                          collate_fn=self.pad_audio_seq)

    def configure_optimizers(self) -> callable:
        return Adam(self.parameters(), lr=self.hparams["lr"])

    def on_fit_start(self):
        metric_placeholder = {'val_loss': 0}
        self.logger.log_hyperparams(self.hparams, metrics=metric_placeholder)

    def forward(self, src, tgt):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = self._generate_square_subsequent_mask(len(src))

        src *= math.sqrt(self.hparams.n_mels)
        src = self.pos_encoder(src)
        memory = self.encoder(src, self.src_mask)
        output = self.decoder(tgt, memory)
        return output

    def training_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        logits = self(x, y)
        loss = F.l1_loss(logits, y)

        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        logits = self(x, y)
        val_loss = F.l1_loss(logits, y)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        log = {"avg_val_loss": val_loss}
        # val loss triggers checkpointing automatically on min val_loss
        return {"log": log, "val_loss": log}

