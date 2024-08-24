from torch.utils.data import DataLoader

from utils.config import Config


from .base_trainer import BaseTrainer
from .SP_trainer_base import SPTrainerBase


def get_trainer(net, train_loader: DataLoader, config: Config):
    if type(train_loader) is DataLoader:
        trainers = {
            'base': BaseTrainer,
            'SP_base': SPTrainerBase,
        }
        return trainers[config.trainer.name](net, train_loader, config)
