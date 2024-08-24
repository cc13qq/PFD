
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import utils.comm as comm

from .xception_net import xception_net, xception_net_sep


def get_network(network_config):

    num_classes = network_config.num_classes
    
    if network_config.name == 'X':
        net = xception_net(num_classes=num_classes)
    
    elif network_config.name == 'X_sep':
        net = xception_net_sep(num_classes=num_classes)
    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        if type(net) is dict:
            for subnet, checkpoint in zip(net.values(), network_config.checkpoint):
                if checkpoint is not None:
                    if checkpoint != 'none':
                        subnet.load_state_dict(torch.load(checkpoint),
                                               strict=False)
        elif network_config.name == 'bit' and not network_config.normal_load:
            net.load_from(np.load(network_config.checkpoint))
        elif network_config.name == 'vit':
            pass
        else:
            try:
                net.load_state_dict(torch.load(network_config.checkpoint), strict=False)
            except RuntimeError:
                # sometimes fc should not be loaded
                loaded_pth = torch.load(network_config.checkpoint)
                loaded_pth.pop('fc.weight')
                loaded_pth.pop('fc.bias')
                net.load_state_dict(loaded_pth, strict=False)
        print('Model Loading {} Completed!'.format(network_config.name))

    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()
        torch.cuda.manual_seed(1)
        np.random.seed(1)
    cudnn.benchmark = True
    return net
