from pipelines import get_pipeline
from utils import launch, setup_config
import torch
import random
import numpy as np

def main(config):

    pipeline = get_pipeline(config)
    pipeline.run()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    config = setup_config()

    setup_seed(config.seed)

    launch(
        main,
        config.num_gpus,
        num_machines=config.num_machines,
        machine_rank=config.machine_rank,
        dist_url='auto',
        args=(config, ),
    )
