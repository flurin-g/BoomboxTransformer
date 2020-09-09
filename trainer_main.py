# ----------------
# trainer_main.py
# ----------------
import pathlib

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import BoomboxTransformer

CWD = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    tb_logger = TensorBoardLogger(save_dir=cfg.save_dir)
    boom: BoomboxTransformer = BoomboxTransformer(cfg, CWD)
    # print(cfg.pretty()) #to view the parameters
    if cfg.create_meta:
        boom.prepare_data()

    trainer = Trainer(logger=tb_logger)
    trainer.fit(boom)


if __name__ == "__main__":
    my_app()
