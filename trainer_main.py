# ----------------
# trainer_main.py
# ----------------
import pathlib

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import BoomboxTransformer

CWD = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    boom = BoomboxTransformer(cfg, CWD)

    boom.prepare_data()

    trainer = Trainer(**cfg.lightning)

    trainer.fit(boom)


if __name__ == "__main__":
    my_app()
