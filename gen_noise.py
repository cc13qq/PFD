import torch
import numpy as np
import os
import time
import random
from torchvision import transforms
import argparse
import cv2
from torch.utils.data import Dataset
from PIL import Image
import latexify
import math


def run_normal_cls_save_pic(atk_method, dataset_name, num_per_cls, params, split, save_img):
    # save: torch.pt, normalized adversarial input tensor

    print('\n Attack dataset ', dataset_name)

    if dataset_name == 'CIFAR10':
        LABEL = CIFAR10_LABEL
        adv_data_root = 'E:/Files/code/Dataset/cifar10_adv_wrn28_' + split
        clean_data_root = 'E:/Files/code/Dataset/cifar10/' + split
        transform = transforms.Compose([])
    elif dataset_name == 'ImageNet100':
        LABEL = IMAGENET100_LABEL
        adv_data_root = 'E:/Files/code/Dataset/imagenet100_adv_resnet50_' + split
        clean_data_root = 'E:/Files/code/Dataset/imagenet100/' + split
        transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224)
        ])
    
    print(' Attack algorithm ', str(atk_method))

    class_names = os.listdir(clean_data_root)
    num_class = len(class_names)

    for cls in class_names:
        
        clean_root = os.path.join(clean_data_root, cls)
        sub_root = 'pt_advs/' + atk_method + '/eps' + str(params['eps'])
        adv_root = os.path.join(adv_data_root, sub_root, cls)
        noise_sub_root = 'noises/' + atk_method + '/eps' + str(params['eps'])
        save_root = os.path.join(adv_data_root, noise_sub_root, cls)
        if not os.path.exists(save_root):
            os.makedirs(save_root) # 创建数据保存路径
        else:
            return
        
        img_list = os.listdir(clean_root)
        selected_img_list = img_list[:num_per_cls]
        for img_name in selected_img_list:
            ori_img_dir = os.path.join(clean_root, img_name)
            if not img_name.endswith('.png'):
                img_name = img_name.split('.')[0] + '.png'
            adv_img_dir = os.path.join(adv_root, img_name)
            
            ori_img = Image.open(ori_img_dir)
            ori_img = ori_img.convert("RGB")
            if dataset_name == 'ImageNet100':
                ori_img = transform(ori_img) 
            
            adv_img = Image.open(adv_img_dir)
            adv_img = adv_img.convert("RGB")
            
            ori_img = np.array(ori_img)
            adv_img = np.array(adv_img)
            
            noise = adv_img.astype(np.int32) - ori_img.astype(np.int32)
            noise = np.absolute(noise)
            
            if save_img:
                # save image
                _ = save_adv_image(noise[:,:,[2,1,0]], img_name, save_root)

    print('noise generation compete')
  
  
def gen_noise(atk):
    
    return

IMAGENET100_LABEL = {'Americanegret': 0, 'Bread': 1, 'Grasshopper': 2, 'ModelT': 3, 'Oriole': 4, 'Persiancat': 5, 'Recordsheet': 6, 'Rottweiler': 7, 'SaintBernard': 8, 'Siamese': 9, 
                     'Tram': 10, 'artichoke': 11, 'badger': 12, 'beerbottle': 13, 'birdhouse': 14, 'bottlecap': 15, 'butterfly': 16, 'butternutsquash': 17, 'carrier': 18, 'carton': 19, 
                     'carwheel': 20, 'cassetteplayer': 21, 'catamaran': 22, 'chickadee': 23, 'chiffonier': 24, 'chimpanzee': 25, 'coffeepot': 26, 'convertible': 27, 'crocodile': 28, 'cucumber': 29, 
                     'daybed': 30, 'desktopcomputer': 31, 'digitalclock': 32, 'dishrag': 33, 'dustcart': 34, 'firetruck': 35, 'fly': 36, 'freightcar': 37, 'gartersnake': 38, 'gibbon': 39, 
                     'goldfish': 40, 'golfball': 41, 'gorilla': 42, 'greatgreyowl': 43, 'horn': 44, 'hourglass': 45, 'husky': 46, 'hyaena': 47, 'jeep': 48, 'keypad': 49, 
                     'leatherbackturtle': 50, 'leopard': 51, 'limpkin': 52, 'lizard': 53, 'malamute': 54, 'mask': 55, 'matchstick': 56, 'measuringcup': 57, 'minibus': 58, 'missile': 59, 
                     'mobilehome': 60, 'monitor': 61, 'moped': 62, 'numbfish': 63, 'ostrich': 64, 'person': 65, 'piano': 66, 'pooltable': 67, 'projector': 68, 'radiotelescope': 69, 
                     'recreationalvehicle': 70, 'reflexcamera': 71, 'revolver': 72, 'rocker': 73, 'rule': 74, 'runningshoe': 75, 'schoolbus': 76, 'soapdispenser': 77, 'stonewall': 78, 'stoplight': 79, 
                     'streetsign': 80, 'submarine': 81, 'suspensionbridge': 82, 'tabby': 83, 'tablelamp': 84, 'tank': 85, 'taxi': 86, 'thatch': 87, 'tileroof': 88, 'trifle': 89, 
                     'trimaran': 90, 'vase': 91, 'warplane': 92, 'washer': 93, 'watertower': 94, 'web': 95, 'whitefox': 96, 'wolfspider': 97, 'wool': 98, 'yurt': 99}

CIFAR10_LABEL = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, 
                 "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}

def save_adv_image(image, img_name, output_dir):#保存图片
    """Saves images to the output directory.
    input numpy.array (H,W,C) image, img_name
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # path = src + '.jpg'
    image = np.clip(image, 0, 255).astype(np.uint8)
    if img_name.endswith('.png'):
        cv2.imwrite(os.path.join(output_dir, img_name), image.astype(np.uint8))
        # print('saving ', img_name)
    elif img_name.endswith('.jpg'):
        img_name = img_name.split('.jpg')[0]
        cv2.imwrite(os.path.join(output_dir, img_name+'.png'), image.astype(np.uint8))
    elif img_name.endswith('.JPEG'):
        img_name = img_name.split('.JPEG')[0]
        cv2.imwrite(os.path.join(output_dir, img_name+'.png'), image.astype(np.uint8))
        # print('saving ', img_name+'.png')
    return os.path.join(output_dir, img_name)

class ImageList_Dataset(Dataset):
    def __init__(self, data_root, transform, label, max_num=None):

        self.transform = transform

        if max_num is not None:
            self.image_list = os.listdir(data_root)[:max_num]
        else:
            self.image_list = os.listdir(data_root)

        self.label = label
        self.data_root = data_root
        # random.shuffle(self.image_list)

    def __getitem__(self, item):
        # [img_path, gt_str] = self.image_list[item].split('\t')
        img_name = self.image_list[item]
        img_path = os.path.join(self.data_root, img_name)
        img = Image.open(img_path)
        img = img.convert("RGB")
        label = self.label
        img = self.transform(img)
        return img, label, img_name
    
    def __len__(self):
        return len(self.image_list)
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda_id", type=int, default=0,
                        help="The GPU ID")
    parser.add_argument("--save_dir", type=str, default='./data',
                        help="data list save path")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="training & validation batch size")
    parser.add_argument("--save_img", type=str2bool, default=True,
                        help="save images or not")
    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        help="base dataset name")
    parser.add_argument("--split", type=str, default='train',
                        help="train or test")
    parser.add_argument("--targeted", type=str, default=False,
                        help="calculate attack success rate or not")
    parser.add_argument("--attack_list", type=str, default=['FGSM', 'PGD', 'VNIFGSM','DIFGSM','MIFGSM', 'NIFGSM', 'SINIFGSM','BIM','RFGSM'],
                        help="face model ")
    parser.add_argument("--attack", type=str, default=None,
                        help="face model ")
    parser.add_argument("--gpu", type=str, default='0',
                        help="set gpu ")
    parser.add_argument("--num_per_cls", type=int, default=-1,
                        help="number [er class] ")
    parser.add_argument("--eps", type=int, default=8,
                        help="eps ")
    # clf model
    parser.add_argument("--clf_dir", type=str, default='E:/Files/code/Adv_Defense/Adv_Training/ensemble/net_weights/Clean/wrn-28-10-dropout0.3.pth')

    return parser.parse_args()

def main():
    args = input_args()
    dataset_name = args.dataset
    if args.num_per_cls == -1:
        num_per_cls = 100 if dataset_name == 'CIFAR10' else 10
    else:
        num_per_cls = args.num_per_cls
    
    eps = 8 if args.eps is None else args.eps#[5,10,15] 
    params = {'eps':eps, 'alpha':2, 'steps':10, 'gamma':0}
    
    if args.attack is not None:
        atk_method = args.attack
        run_normal_cls_save_pic(atk_method, dataset_name, num_per_cls, params, args.split, args.save_img)
    else:
        for atk_method in args.attack_list:
            run_normal_cls_save_pic(atk_method, dataset_name, num_per_cls, params, args.split, args.save_img)
    
if __name__=='__main__':
    main()