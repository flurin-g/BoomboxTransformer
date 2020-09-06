import math
from pathlib import Path

import torch
import torchaudio
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule, DataLoader
from torch import nn
from torch.nn import functional as F, Dropout, Module, TransformerEncoderLayer, TransformerEncoder, \
    TransformerDecoderLayer, TransformerDecoder
from torch.optim import Adam

from noisy_speech import NoisySpeechDataset


class PositionalEncoding(Module):

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

        self.settings = cfg.dataset
        self.cwd = cwd
        self.hparams = cfg.hparams

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

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = self._generate_square_subsequent_mask(len(src))

        src *= math.sqrt(self.nmels)
        src = self.pos_encoder(src)
        memory = self.encoder(src, self.src_mask)
        output = self.decoder(tgt, memory)
        return output

    def prepare_data(self) -> None:
        NoisySpeechDataset.create_libri_meta(libri_path=self.settings.libri_path,
                                             libri_meta_path=self.settings.libri_speakers,
                                             file_name=self.settings.libri_meta,
                                             cwd=self.cwd,
                                             subsets=self.settings.libri_subsets)

        NoisySpeechDataset.create_urban_meta(urban_path=self.settings.urban_path,
                                             file_name=self.settings.urban_meta,
                                             cwd=self.cwd)

    # ToDo: integrate collate function
    def data_processing(data):
        x_spectrograms = []
        y_spectrograms = []
        input_lengths = []
        for (waveform, _, _, _, _, _) in data:
            x_spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            y_spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            x_spectrograms.append(x_spec)
            y_spectrograms.append(y_spec)
            input_lengths.append(x_spec.shape[0])

        # sort by sequence length, for pad_sequence to work
        ixs = sorted((ix for ix in range(len(input_lengths))), key=lambda ix: input_lengths[ix], reverse=True)
        x_spectrograms = [x_spectrograms[ix] for ix in ixs]
        y_spectrograms = [y_spectrograms[ix] for ix in ixs]

        x_spectrograms = nn.utils.rnn.pad_sequence(x_spectrograms, batch_first=False)
        y_spectrograms = nn.utils.rnn.pad_sequence(y_spectrograms, batch_first=False)
        seq_len = x_spectrograms.shape[0]
        return x_spectrograms, y_spectrograms, seq_len

    def train_dataloader(self) -> DataLoader:
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.settings.sr_libri,
                                                         n_mels=self.hparams["n_mels"])
        dataset = NoisySpeechDataset(self.settings.libri_meta,
                                     self.settings.urban_meta,
                                     self.cwd,
                                     "train",
                                     self.settings.libri_subsets,
                                     transform,
                                     1,
                                     self.settings.sr_libri,
                                     self.settings.sr_urban,
                                     self.settings.libri_path,
                                     self.settings.urban_path)

        return DataLoader(dataset, batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams["lr"])

    def training_step(self, batch, batch_idx) -> dict:
        x, y = batch
        logits = self(x)
        loss = F.l1_loss(logits, y)

        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}
