import torchvision.transforms as tvs_trans

from utils.config import Config

from .base_preprocessor import BasePreprocessor
from .transform import Convert, interpolation_modes, normalization_dict


class TestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""
    def __init__(self, config: Config, split):
        dataset_name = config.dataset.name.split('_')[0]
        image_size = config.dataset.image_size
        pre_size = config.dataset.pre_size
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        try:
            interpolation = config.dataset[split].interpolation
        except KeyError:
            interpolation = config.ood_dataset.interpolation

        interpolation = interpolation_modes[interpolation]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize((pre_size, pre_size), interpolation=interpolation),
            # tvs_trans.CenterCrop(pre_size), # for general image
            tvs_trans.Resize((image_size, image_size), interpolation=interpolation),
            # tvs_trans.CenterCrop(image_size),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=mean, std=std),
        ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)

class ImageNetTestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""
    def __init__(self, config: Config, split):
        dataset_name = config.dataset.name.split('_')[0]
        image_size = config.dataset.image_size
        pre_size = config.dataset.pre_size
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        try:
            interpolation = config.dataset[split].interpolation
        except KeyError:
            interpolation = config.ood_dataset.interpolation

        interpolation = interpolation_modes[interpolation]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize((256, 256), interpolation=interpolation),
            tvs_trans.CenterCrop(pre_size), # for general image
            tvs_trans.Resize((image_size, image_size), interpolation=interpolation),
            # tvs_trans.CenterCrop(image_size),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=mean, std=std),
        ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)
