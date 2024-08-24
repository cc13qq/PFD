from utils.config import Config

from .base_preprocessor import BasePreprocessor, ImageNetBasePreprocessor
# from .cutpaste_preprocessor import CutPastePreprocessor
# from .draem_preprocessor import DRAEMPreprocessor
# from .pixmix_preprocessor import PixMixPreprocessor
from .test_preprocessor import TestStandardPreProcessor, ImageNetTestStandardPreProcessor


def get_preprocessor(config: Config, split):
    train_preprocessors = {
        'base': BasePreprocessor,
        'ImageNet': ImageNetBasePreprocessor,
        # 'draem': DRAEMPreprocessor,
        # 'cutpaste': CutPastePreprocessor,
        # 'pixmix': PixMixPreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
        'ImageNet': ImageNetTestStandardPreProcessor,
        # 'draem': DRAEMPreprocessor,
        # 'cutpaste': CutPastePreprocessor,
        # 'pixmix': TestStandardPreProcessor,
    }

    if split == 'train':
        return train_preprocessors[config.preprocessor.name](config, split)
    else:
        return test_preprocessors[config.preprocessor.name](config, split)
