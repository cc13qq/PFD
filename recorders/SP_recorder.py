import os
import time
from pathlib import Path

import torch
import csv


class SPRecorder:
    def __init__(self, config) -> None:
        self.config = config

        self.best_acc = 0.0
        self.best_auc = 0.0
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics, val_metrics):
        # train_metrics: ['loss', 'epoch_idx']
        '''
        val/test metrics:   dict[metrics_set, loss, auc, epoch_idx, batch_idx]
                 metrics_set:    dict[metrics_subset, loss, auc]
                 metrics_subset:    dict[loss, auc, num_loss, num_acc]
        '''

        if train_metrics is not None:
            print('\nEpoch {:03d} | Batch {:03d} | Time {:5d}s | Train Loss {:.4f} | '.format(
                    train_metrics['epoch_idx'], train_metrics['batch_idx'], int(time.time() - self.begin_time), train_metrics['loss']), flush=True)

        if val_metrics is not None:
            print('Val AvgLoss {:.4f} | Val AvgAUC {:.4f}'.format(
                val_metrics['loss'], val_metrics['auc']), flush=True)

            for set in val_metrics:
                if set in ['loss', 'auc', 'epoch_idx', 'batch_idx']:
                    continue
                if set == 'Real':
                    print('\t Val metrics on set '+ set + ': Loss {:.4f}'.format(
                        val_metrics[set]['loss']), flush=True)
                    continue
                else:
                    print('\t Val metrics on set '+ set + ': AvgLoss {:.4f} | AvgAUC {:.4f}'.format(
                        val_metrics[set]['loss'], val_metrics[set]['auc']), flush=True)

                    for subset in val_metrics[set]:
                        if subset in ['loss', 'auc']:
                            continue
                        print('\t \t Val metrics on subset '+ subset + ': Loss {:.4f} | AUC {:.4f}'.format(
                            val_metrics[set][subset]['loss'], val_metrics[set][subset]['auc']), flush=True)

    def save_model(self, nets, val_metrics):

        net_name='net'

        if self.config.recorder.save_all_models:
            # for net_name in nets:
            save_fname = 'model_epoch{}_batch{}_auc{:.4f}.ckpt'.format(val_metrics['epoch_idx'], val_metrics['batch_idx'], val_metrics['auc'])
            save_pth = os.path.join(self.output_dir, net_name+'-'+save_fname)
            print('\nsaving ' + net_name+'-'+save_fname)
            torch.save(nets[net_name].state_dict(), save_pth)

        # enter only if better accuracy occurs
        # elif val_metrics['acc'] >= self.best_acc:
        elif val_metrics['auc'] >= self.best_auc:
            # update the best model
            self.best_epoch_idx = val_metrics['epoch_idx']
            self.best_batch_idx = val_metrics['batch_idx']
            # self.best_acc = val_metrics['acc']
            self.best_auc = val_metrics['auc']

            # save_fname = 'best_epoch{}_batch{}acc{:.4f}.ckpt'.format(self.best_epoch_idx, self.best_batch_idx, self.best_acc)
            save_fname = 'best_epoch{}_batch{}auc{:.4f}.ckpt'.format(self.best_epoch_idx, self.best_batch_idx, self.best_auc)
            
            # for net_name in nets:
            save_pth = os.path.join(self.output_dir, net_name+'-'+save_fname)
            print('\nsaving ' + net_name+'-'+save_fname)
            torch.save(nets[net_name].state_dict(), save_pth)

        # save last path
        if val_metrics['epoch_idx'] == self.config.optimizer.num_epochs:
            save_fname = 'last_epoch{}_auc{:.4f}.ckpt'.format(val_metrics['epoch_idx'], val_metrics['auc'])

            # for net_name in nets:
            save_pth = os.path.join(self.output_dir, net_name+'-'+save_fname)
            print('\nsaving ' + net_name+'-'+save_fname)
            torch.save(nets[net_name].state_dict(), save_pth)

    def summary(self):
        print('Training Completed! '
              'Best AUC: {:.4f} '
              'at epoch {:d}'.format(self.best_auc, self.best_epoch_idx),
              flush=True)
