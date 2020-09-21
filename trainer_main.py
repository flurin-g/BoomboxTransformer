import pathlib

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from boombox_lstm import BoomboxLSTM
from model import BoomboxTransformer
from noisy_speech import NoisySpeechModule

CWD = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    noisy_speech = NoisySpeechModule(cfg.dataset,
                                     cfg.hparams.batch_size,
                                     cfg.hparams.n_mels,
                                     CWD)

    if cfg.hparams.model_name == "LSTM":
        boom = BoomboxLSTM(cfg, CWD)

    trainer = Trainer(**cfg.lightning)
    trainer.fit(boom, noisy_speech)


if __name__ == "__main__":
    train()
