import numpy as np
from utils import Config

from .base_evaluator import BaseEvaluator
from sklearn import metrics

class SP_Evaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super(SP_Evaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None
    
    def auc_simple(self, logit, label):
        auroc = metrics.roc_auc_score(label, logit)
        return auroc
        
    def eval_acc_SP(self, nets, data_loader_split,
                 postprocessor: BaseEvaluator = None,
                 epoch_idx: int = -1, batch_idx: int = -1):

        for net_name in nets:

            if type(nets[net_name]) is dict:
                nets[net_name]['backbone'].eval()
            else:
                nets[net_name].eval()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['batch_idx'] = batch_idx

        loss_sum = 0.0
        loss_num = 0

        auc_sum = 0
        auc_num = 0

        metrics['Real'], logit_real, gt_real = self._eval_acc_Real(nets, data_loader_split['Real'], postprocessor, detailed_return=True)

        for set in data_loader_split:

            if set == 'Real':
                metrics[set]['auc'] = None
                continue

            metrics[set] = self._eval_acc_SP(nets,
                        data_loader_split[set],
                        logit_real, gt_real, 
                        postprocessor)
            
            loss_sum_set = 0.0
            loss_num_set = 0
            auc_sum_set = 0
            for subset_name in metrics[set]:
                loss_sum_set += metrics[set][subset_name]['loss'] * metrics[set][subset_name]['num_loss']

                loss_num_set += metrics[set][subset_name]['num_loss']

                auc_sum_set += metrics[set][subset_name]['auc']

            auc_num_set = len(metrics[set])
            
            metrics[set]['loss'] = loss_sum_set / loss_num_set if loss_num_set !=0 else 0
            metrics[set]['auc'] = auc_sum_set / auc_num_set if auc_num_set !=0 else 0

            loss_sum += loss_sum_set

            loss_num += loss_num_set

            auc_sum += metrics[set]['auc']
            auc_num += 1

        metrics['loss'] = loss_sum / loss_num
        metrics['auc'] = auc_sum / auc_num

        return metrics
    
    def _eval_acc_SP(self, nets, data_loader_set, logit_real, gt_real,
                 postprocessor: BaseEvaluator = None):

        metrics_set = {}
        for subset_name, data_loader_subset in data_loader_set.items():
            metrics_set[subset_name] = {}
            pred_subset, logit_subset, gt_subset, loss_subset  = postprocessor.inference(nets, data_loader_subset, 'loss')
            loss_sum = sum(loss_subset)
            loss_num = len(data_loader_subset) # batch num
            metrics_set[subset_name]['loss'] = loss_sum / loss_num
            metrics_set[subset_name]['num_loss'] = loss_num

            pred_p = np.concatenate((logit_real[:,1], logit_subset[:,1]), axis=0)
            gt_cat = np.concatenate((gt_real, gt_subset), axis=0)
            metrics_set[subset_name]['auc'] = self.auc_simple(pred_p, gt_cat)

        return metrics_set
    
    def _eval_acc_Real(self, nets, data_loader_set,
                 postprocessor: BaseEvaluator = None,
                 detailed_return: bool = False):
        metrics_set = {}
        pred_set, logit_set, gt_set, loss_set  = postprocessor.inference(nets, data_loader_set, 'loss')
        loss_sum = sum(loss_set)
        loss_num = len(data_loader_set) # batch num
        metrics_set['loss'] = loss_sum / loss_num

        if not detailed_return:
            return metrics_set
        else:
            return metrics_set, logit_set, gt_set

