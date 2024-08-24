import ast
import io
import logging
import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
import random
from .base_dataset import BaseDataset
from .RAP import rap_process_opt, add_grad_noise_GC, get_GC_scaler, add_pixel_GC_continue_opt, load_dis, dis_process, dis_process_padding#get_face_matrix, add_pixel_GC_continue, RAP
# to fix "OSError: image file is truncated"

from torchvision.models import resnet50
from BASNet.model import BASNet
from torchcam.methods import SmoothGradCAMpp

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ImglistDataset_SP(BaseDataset):
    def __init__(self,
                 name,
                 split,
                 interpolation,
                 image_size,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 mode = 'RAP',
                 pre_size = 256, 
                 eps = None,
                 dataset_name = None,
                 params = None,
                 **kwargs):
        super(ImglistDataset_SP, self).__init__(**kwargs)

        self.name = name
        self.image_size = image_size
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        self.eps = eps
        self.dataset_name = dataset_name
        
        self.params = params

        self.mode = mode
        if mode in ['RAP', 'RAP_pixel', 'RAP_block']: # 'RAP' or 'GC'
            self.gc_prob = -1
            self.rap = 0.5
        elif mode == 'GC':
            self.gc_prob = 1
            self.rap = 0.5
        elif mode in ["GC_n","RC_point","RC_block","RC_mix","mix"]:
            self.gc_prob = 1
            self.rap = 0.5
        elif mode in ["RC_point_mask","RC_block_mask","RC_mix_mask","mix_mask"]:
            self.gc_prob = 1
            self.rap = 0.5
            self.cam_model = resnet50(pretrained=True).eval()
            self.cam_model.cuda()
            self.cam_model.eval()
            self.saliency_model = BASNet(3,1)
            self.saliency_model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'BASNet/saved_models/basnet.pth')))
            self.saliency_model.cuda()
            self.saliency_model.eval()
            self.cam_extractor = SmoothGradCAMpp(self.cam_model ,"layer4")
        elif mode == 'DIS':
            self.dis_dict = load_dis(atk_method = 'FGSM', dataset_name = self.dataset_name, eps = 4)
            self.rap = 0.5
        elif mode == None:
            self.rap = -1
        else:
            raise ValueError('mode must be "RAP", "GC" or None')
        
        self.pre_size = pre_size
            
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')
        
        self.GC_scaler = get_GC_scaler(256)

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)
    
    # def SP_process(self, ori_img, gc_prob):
    #     if np.random.random() <= gc_prob:
    #         # landmark, _  = get_landmark(ori_img)
    #         # # 5.五官区域添加渐变patch
    #         # _, m = get_face_matrix(ori_img, loc = ['left_eye', 'right_eye', 'nose_tip', 'mouse'], landmark= landmark)
    #         # ori_img, mask = add_pixel_GC_continue(ori_img, eps=random.randint(10, 50), prob=random.randint(1, 5)/1000, sl = [10, 100], loc = 'local', GC_size = 256, th = 50, face_matrix =None, GC_scaler=None, image_post = True) # for 1024x1024
    #         # sp_img, mask = add_pixel_GC_continue(ori_img, eps=random.randint(10, 50), prob=random.randint(4, 20)/1000, sl = [2, 25], loc = 'local', GC_size = 256, th = 50, face_matrix =None, GC_scaler=None, image_post = True) # for 256x256

    #         # ori_img, mask = add_pixel_GC_continue_opt(ori_img, eps=random.randint(10, 50), prob=random.randint(1, 5)/1000, sl = [10, 100], loc = 'local', GC_size = 256, th = 50, face_matrix=None, GC_scaler=None, image_post = True) # for 1024x1024
    #         sp_img, mask = add_pixel_GC_continue_opt(ori_img, eps=random.randint(10, 50), prob=random.randint(4, 20)/1000, sl = [2, 25], loc = 'local', GC_size = 256, th = 50, face_matrix=None, GC_scaler=self.GC_scaller, image_post = True) # for 256x256
    #     else:
    #         # sp_img, mask = self.rap_process(ori_img, type = 'random')
    #         sp_img, mask = rap_process_opt(ori_img, eps=None, sl=6, noise_type=None, image_post = True, al=3)

    #     sp_img = sp_img.astype(np.uint8)

    #     return sp_img, mask
    def SP_process(self, ori_img, mode, area_mask=None):
        if mode == 'GC':
            ori_img = cv2.resize(ori_img, (self.pre_size, self.pre_size))
            sp_img, mask = add_pixel_GC_continue_opt(ori_img, eps=random.randint(10, 70), prob=random.randint(16, 40)/1000, sl = [2, 25], loc = 'local', GC_size = 256, th = 50, face_matrix=None, GC_scaler=self.GC_scaler, image_post = True) # for 256x256
        elif mode == 'RAP':
            ori_img = cv2.resize(ori_img, (self.pre_size, self.pre_size)) # (112, 112) for faces
            sp_img, mask = rap_process_opt(ori_img, eps=self.eps, sl=6, noise_type=None, image_post = True, al=3)
        elif mode == 'GC_n':
            sp_img, mask = add_grad_noise_GC(ori_img, eps=random.randint(10, 70), prob=random.randint(16, 40)/1000, sl = [2, 25], mode = mode, GC_size = 256, th = 50, GC_scaler=self.GC_scaler, image_post = True) # for 256x256
        elif mode in ["RC_point","RC_block","RC_mix","mix"]:
            sp_img, mask = add_grad_noise_GC(ori_img, eps=self.eps, prob=random.randint(16, 40)/1000, sl = [2, 25], mode = mode, GC_size = 256, th = 50, GC_scaler=self.GC_scaler, image_post = True) # for 256x256
        elif mode in ["RC_point_mask","RC_block_mask","RC_mix_mask","mix_mask"]:
            sp_img, mask = add_grad_noise_GC(ori_img, eps=self.eps, prob=random.randint(16, 40)/1000, sl = [2, 25], mode = mode, GC_size = 256, th = 50, GC_scaler=self.GC_scaler, image_post = True, mask_area=True, mask = area_mask, cam_model=self.cam_model, saliency_model=self.saliency_model, cam_extractor=self.cam_extractor) # for 256x256
        elif mode == 'DIS':
            if 'ImageNet' in self.dataset_name:
                ori_img = cv2.resize(ori_img, (256, 256)) # (112, 112) for faces
                sp_img, mask = dis_process_padding(ori_img, self.dis_dict, eps=self.eps, eps_low=self.params.eps_low, eps_high = self.params.eps_high, image_post = True, padding = True, var_loc = self.params.var_loc, var_cov = self.params.var_cov)
            else:
                ori_img = cv2.resize(ori_img, (self.pre_size, self.pre_size)) # (112, 112) for faces
                sp_img, mask = dis_process(ori_img, self.dis_dict, eps=self.eps, eps_low=1, eps_high = 20, image_post = True)
        sp_img = sp_img.astype(np.uint8)

        return sp_img, mask
    
    def SP_process_split(self, ori_img, mode):
        if mode == 'GC':
            ori_img = cv2.resize(ori_img, (self.pre_size, self.pre_size))
            sp_img, mask = add_pixel_GC_continue_opt(ori_img, eps=random.randint(10, 70), prob=random.randint(16, 40)/1000, sl = [2, 25], loc = 'local', GC_size = 256, th = 50, face_matrix=None, GC_scaler=self.GC_scaler, image_post = True) # for 256x256
        elif mode == 'RAP_pixel':
            ori_img = cv2.resize(ori_img, (self.pre_size, self.pre_size)) # (112, 112) for faces
            sp_img, mask = rap_process_opt(ori_img, eps=None, sl=6, noise_type='point', image_post = True, al=3)
        elif mode == 'RAP_block':
            ori_img = cv2.resize(ori_img, (self.pre_size, self.pre_size)) # (112, 112) for faces
            sp_img, mask = rap_process_opt(ori_img, eps=None, sl=6, noise_type='block', image_post = True, al=3)

        sp_img = sp_img.astype(np.uint8)

        return sp_img, mask
    
    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        # print(line)
        tokens = line.split('\t', 1)
        # print(tokens)
        image_name, extra_str = tokens[0], tokens[1]
        # print('image_name',image_name)
        # print('extra_str',extra_str)
        if self.data_dir == None:
            path = image_name
        else:
            path = os.path.join(self.data_dir, image_name)

        # change dir
        # if path.startswith('E:'):
        #     path = 'D:' + path.split('E:')[-1]

        # print('self.data_dir,',self.data_dir,)
        # print('path',path)
        sample = dict()
        sample['image_name'] = image_name

        # TODO: comments
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        self.preprocessor.setup(**kwargs)
        try:
            img = cv2.imread(path)
            # img = cv2.resize(img, (256, 256))
            if np.random.random() <= self.rap:
                
                if self.mode in ["RC_point_mask","RC_block_mask","RC_mix_mask","mix_mask"]:
                    
                    # mask_root, img_prex = path.split('train')
                    # cls_name, img_name = img_prex.split('\\')[1:]
                    # img_name = 'n' + img_name
                    # mask_root = mask_root + 'train_masks'
                    # mask_dir = os.path.join(mask_root, cls_name, 'mask_'+img_name)
                    
                    # if mask_dir.endswith('.JPEG'):
                    #     mask_dir0 = mask_dir.split('.JPEG')[0]
                    #     mask_dir = mask_dir0 + '.png'
                    # elif mask_dir.endswith('.jpg'):
                    #     mask_dir0 = mask_dir.split('.jpg')[0]
                    #     mask_dir = mask_dir0 + '.png'
        
                    # mask_img = cv2.imread(mask_dir)
                    
                    mask_img = None
                    
                    img, _ = self.SP_process(img, mode=self.mode, area_mask = mask_img)

                else:
                    img, _ = self.SP_process(img, mode=self.mode)
                # img, _ = self.SP_process_split(img, mode=self.mode)
                sample['label'] = 0
                # print(sample['label'])
            else:
                sample['label'] = 1
            image = Image.fromarray(img)
            sample['data'] = self.transform_image(image)
            sample['data_aux'] = self.transform_aux_image(image)

            # sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e

        # print(sample['data'].shape)
        return sample

    def getitem_pre(self, index):
        line = self.imglist[index].strip('\n')
        # print(line)
        tokens = line.split('\t', 1)
        # print(tokens)
        image_name, extra_str = tokens[0], tokens[1]
        # print('image_name',image_name)
        # print('extra_str',extra_str)
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        if self.data_dir == None:
            path = image_name
        else:
            path = os.path.join(self.data_dir, image_name)

        # change dir
        # if path.startswith('E:'):
        #     path = 'D:' + path.split('E:')[-1]

        # print('self.data_dir,',self.data_dir,)
        # print('path',path)
        sample = dict()
        sample['image_name'] = image_name

        # TODO: comments
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        self.preprocessor.setup(**kwargs)
        try:
            if not self.dummy_read:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            extras = ast.literal_eval(extra_str)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                # image = Image.open(buff).convert('RGB')
                img = cv2.imread(path)
                img = cv2.resize(img, (256, 256))
                if np.random.random() <= self.rap:
                    img, _ = self.SP_process(img, gc_prob=0.5)
                    extra_str = 0
                image = Image.fromarray(img)
                sample['data'] = self.transform_image(image)
                sample['data_aux'] = self.transform_aux_image(image)
            # extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample
    
