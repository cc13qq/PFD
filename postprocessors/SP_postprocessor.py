from typing import Any

import torch
import torch.nn as nn
import numpy as np
from .base_postprocessor import BasePostprocessor
from torch.utils.data import DataLoader
import torch.nn.functional as F


class SP_Postprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.mode = self.config.postprocessor.mode

    @torch.no_grad()
    def postprocess_energy(self, nets, data: Any):
        logit = nets['net'](data) #logit
        score = torch.softmax(logit, dim=1)
        _, pred = torch.max(score, dim=1)
        energy = self.temperature * torch.logsumexp(logit / self.temperature, dim=1) # energy
        # energy_score = energy / (1 + energy)
        return logit, pred, energy
    
    @torch.no_grad()
    def postprocess_loss(self,nets, data: Any, label: Any):
        logit = nets['net'](data)
        loss = F.cross_entropy(logit, label)
        # pred = logit.data.max(1)[1]
        _, pred = torch.max(logit, dim=1)
        return logit, pred, loss
    
    def inference(self, nets, data_loader: DataLoader, mode: str, eval : bool = True):
        if eval:
            data_choise = 'data_aux'
        else:
            data_choise = 'data'
        pred_list, logit_list, label_list, score_list = [], [], [], []
        if self.mode != mode:
            self.mode = mode
        for batch in data_loader:
            data = batch[data_choise].cuda()
            label = batch['label'].cuda()
            if self.mode == 'loss':
                logit, pred, score = self.postprocess_loss(nets, data, label)
            else:
                logit, pred, score = self.postprocess_energy(nets, data)
                
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                logit_list.append(logit[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())
            if self.mode == 'loss':
                score_list.append(score.cpu().tolist())
            else:
                for idx in range(len(data)):
                    score_list.append(score[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        logit_list = np.array(logit_list)
        label_list = np.array(label_list, dtype=int)
        score_list = np.array(score_list)

        return pred_list, logit_list, label_list, score_list
