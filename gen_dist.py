import json
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import math
from PIL import Image, ImageFilter
import os
import cv2
import imageio
import torch.nn as nn

import io
import os, sys
import requests
import PIL

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

    
def cal_dis(features, memory = None, dis_type = 'MultivariateNormal', device = 'cpu'):
    '''
    features : [batch, dim]
    '''
    
    if memory is None:
        memory = features
    else:
        memory = torch.cat((memory, features), 0)
    X = memory - memory.mean(0)
    mean_embed = memory.mean(0).view(1, -1) # empirical class mean
    
    # print('X', X.shape, X)
    # print('mean_embed', mean_embed.shape, mean_embed)
    
    eye_matrix = torch.eye(memory.shape[-1], device=device)
    
    temp_precision = torch.mm(X.t(), X) / len(X) # covariance
    temp_precision += 0.0001 * eye_matrix
    
    # print('temp_precision', temp_precision.shape, temp_precision)
    # print('eye_matrix', eye_matrix.shape, eye_matrix)
    
    if dis_type == 'MultivariateNormal':
        new_dis = torch.distributions.MultivariateNormal(loc=mean_embed, covariance_matrix=temp_precision) # distribution
    elif dis_type == 'Normal':
        new_dis = torch.distributions.Normal(loc=mean_embed, scale=temp_precision) # distribution
    
    return new_dis, memory

def gen_dist(dataset_name='imagenet100', atk_method='FGSM', eps=4):
    
    print('Generating noise distribution of {} in {}'.format(atk_method, dataset_name))
    
    if dataset_name == 'imagenet100':
        adv_data_root = 'E:/Files/code/Dataset/imagenet100_adv_resnet50_train/noises'
        num_per_cls = 10
    elif dataset_name =='cifar10':
        adv_data_root = 'E:/Files/code/Dataset/cifar10_adv_wrn28_train/noises'
        num_per_cls = 100

    params = {'eps':eps, 'alpha':2, 'steps':10, 'gamma':0}

    scaller = 1

    sub_root = atk_method + '/eps' + str(params['eps'])
    adv_root = os.path.join(adv_data_root, sub_root)
    class_names = os.listdir(adv_root)
    num_class = len(class_names)

    feas_i = []

    for cls in class_names:

        # print('class', cls, ' attack_id', idx)

        cls_root = os.path.join(adv_root, cls)

        img_list = os.listdir(cls_root)
        selected_img_list = img_list[:num_per_cls]

        for img_name in selected_img_list:
            img_path = os.path.join(cls_root, img_name)
            img = cv2.imread(img_path)
            fea = img.reshape(-1)*scaller # flatten
            feas_i.append(fea)

    feas_i = np.array(feas_i)   

    new_dis, _ = cal_dis(torch.Tensor(feas_i), dis_type = 'MultivariateNormal')
    
    dis_dict = {}
    dis_dict['loc'] = new_dis[0].loc
    dis_dict['covariance_matrix'] = new_dis[0].covariance_matrix
    sub_root = dataset_name + '_' + atk_method + '_eps' + str(params['eps']) + '_noise_dist.pt'
    save_root = './dist'
    save_root = os.path.join(save_root, sub_root)
    torch.save(dis_dict, save_root)


if __name__=='__main__':
    adv_class_list = ['FGSM']
    for atk_method in adv_class_list:
        gen_dist('imagenet100', atk_method, eps = 4)