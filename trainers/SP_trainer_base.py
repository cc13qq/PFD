import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import Config
import utils.comm as comm

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class SPTrainerBase: # vos only on real
    def __init__(self, net, train_loader, config: Config):
        self.train_loader = train_loader
        self.config = config
        self.net = net
        self.weight_energy = torch.nn.Linear(config.num_classes, 1).cuda()
        torch.nn.init.uniform_(self.weight_energy.weight)
        self.logistic_regression = torch.nn.Linear(1, 2).cuda()

        # self.optimizer = torch.optim.SGD(
        #     list(net.parameters()) + list(self.weight_energy.parameters()) + list(self.logistic_regression.parameters()),
        #     config.optimizer['learning_rate'],
        #     momentum=config.optimizer['momentum'],
        #     weight_decay=config.optimizer['weight_decay'], nesterov=True)
        
        self.optimizer = torch.optim.Adam(list(net.parameters()) + list(self.weight_energy.parameters()) + list(self.logistic_regression.parameters()),
            lr=config.optimizer['learning_rate'], 
            betas=(0.9, 0.999),
            weight_decay=config.optimizer['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(step, config.optimizer['num_epochs'] * len(train_loader), 1, 1e-6 / config.optimizer['learning_rate']))

        self.nets = dict()
        self.nets['net'] = self.net

    def train_epoch(self, epoch_idx, batch_idx, evaluator, val_loader, postprocessor, recorder):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        criterion = torch.nn.CrossEntropyLoss()


        for train_step in tqdm(range(1, len(train_dataiter) + 1),
                               desc='Epoch {:03d}'.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            images = batch['data'].cuda()
            labels = batch['label'].cuda()

            # x, output = self.net.forward(images, return_feature=True) # x == logits, output == feature?
            logit = self.net.forward(images, return_feature=False) # x == logits, output == feature?
            # loss = F.cross_entropy(logit, labels)
            loss = criterion(logit, labels)
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if self.config.optimizer['lr_scheduler']:
            self.scheduler.step()

            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

            batch_idx += 1

            if batch_idx % self.config.optimizer.logbatch == 0:
                metrics = {}
                metrics['loss'] = loss_avg
                metrics['epoch_idx'] = epoch_idx
                metrics['batch_idx'] = batch_idx
                val_metrics = evaluator.eval_acc_SP(self.nets, val_loader, postprocessor, epoch_idx, batch_idx)
                comm.synchronize()
                if comm.is_main_process():
                    recorder.report(metrics, val_metrics)
                
                for net_name in self.nets:
                        self.nets[net_name].train()

        metrics = {}
        metrics['loss'] = loss_avg
        metrics['epoch_idx'] = epoch_idx
        metrics['batch_idx'] = batch_idx

        # return self.net, metrics
        return self.nets, metrics, batch_idx

