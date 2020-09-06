# ----------------
# trainer_main.py
# ----------------
import pathlib

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from model import BoomboxTransformer

CWD = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path=".", config_name="parameters.yml")
def my_app(cfg: DictConfig) -> None:
    boom: BoomboxTransformer = BoomboxTransformer(cfg, CWD)
    # print(cfg.pretty()) to view the parameters
    if cfg.create_meta:
        boom.prepare_data()

    trainer = Trainer()
    trainer.fit(boom)


if __name__ == "__main__":
    my_app()
