import utils.comm as comm
from datasets import get_dataloader_SP
from evaluators.utils import get_evaluator
from networks.utils import get_network
from recorders.utils import get_recorder
from trainers.utils import get_trainer
from utils.logger import setup_logger

from postprocessors import get_postprocessor

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import Config

class Test_SP_Pipeline_base:
    def __init__(self, config: Config):
        self.config = config
        self.loader_dict = get_dataloader_SP(self.config)
        self.test_loader = self.loader_dict['test']
        
        self.net = get_network(self.config.network)

        self.nets = dict()
        self.nets['net'] = self.net

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        evaluator = get_evaluator(self.config)
        postprocessor = get_postprocessor(self.config)

        recorder = get_recorder(self.config)
        print('Start testing...', flush=True)

        test_metrics = evaluator.eval_acc_SP(self.nets, self.test_loader, postprocessor, -1, -1)
        recorder.report(None, test_metrics)

        print('Completed!', flush=True)
