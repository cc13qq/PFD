import torchvision.transforms as tvs_trans

normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
    'SP_CIFAR10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'SP_ImageNet10': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'DIS_CIFAR10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'DIS_ImageNet100': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'CIFAR10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'ImageNet100': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
}

interpolation_modes = {
    'nearest': tvs_trans.InterpolationMode.NEAREST,
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)