import pathlib

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from model import BoomboxTransformer

CWD = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    boom = BoomboxTransformer(cfg, CWD)

    boom.prepare_data()

    trainer = Trainer(**cfg.lightning)

    trainer.fit(boom)


if __name__ == "__main__":
    train()
