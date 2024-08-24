import torch
import torchvision as tv
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import dlib
import torch
import cv2
import torch.nn as nn
import glob
import cv2
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc
from PIL import Image, ImageDraw
import albumentations as alb
import time
import argparse
from tqdm import tqdm
# from .utils_wq import save_adv_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
import matplotlib

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
    elif img_name.endswith('.jpg'):
        img_name = img_name.split('.jpg')[0]
        cv2.imwrite(os.path.join(output_dir, img_name+'.png'), image.astype(np.uint8))
    return os.path.join(output_dir, img_name)

def shuffle_tensor(input, label):
    l = input.size(0)
    order = torch.randperm(l)
    input = input[order]
    label = label[order]
    return input, label

def get_landmark(img):
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = os.path.join(os.path.dirname(__file__), './shape_predictor_81_face_landmarks.dat')
    face_predictor = dlib.shape_predictor(predictor_path)
    faces = face_detector(img, 1)
    if len(faces) == 0:
        # print('detect no faces')
        faces = [dlib.rectangle(0,0,img.shape[0],img.shape[1])]
    landmark = face_predictor(img, faces[0])

    return landmark, faces

def get_face_area(landmark):
    face_lmk = [i for i in range(17)] + [78, 74, 79 ,73, 72, 80, 71, 70, 69, 68, 76, 75, 77]
    ex = []
    ey = []
    for idx in face_lmk:
        pt = landmark.parts()[idx]
        ex.append(pt.x)
        ey.append(pt.y)
    top = min(ey)
    left = min(ex)
    right = max(ex)
    bottom = max(ey)
    return [top, bottom, left, right]

def get_mouse_area(landmark):
    ex = []
    ey = []
    for pt in landmark.parts()[48:59+1]:
        ex.append(pt.x)
        ey.append(pt.y)
    top = min(ey)
    left = min(ex)
    right = max(ex)
    bottom = max(ey)
    return [top, bottom, left, right]

def get_nose_area(landmark):
    ey = []
    for pt in landmark.parts()[31:35+1]:
        ey.append(pt.y)
    top = landmark.parts()[27].y
    left = min([landmark.parts()[39].x, landmark.parts()[31].x])
    right = max([landmark.parts()[42].x, landmark.parts()[35].x])
    bottom = max(ey)
    return [top, bottom, left, right]

def get_eye_area(landmark):
    ex = []
    ey = []
    for pt in landmark.parts()[17:26+1]:
        ex.append(pt.x)
        ey.append(pt.y)
    top = min(ey)
    left = min(ex)
    right = max(ex)
    bottom = landmark.parts()[28].y
    return [top, bottom, left, right]

def area_expand(area, size = 1024, param = 0.005):
    # area: top,bottom,left,right
    eps = size * param
    area_ex = area.copy()
    area_ex[0] = int(area[0] - eps)
    area_ex[1] = int(area[1] + eps)
    area_ex[2] = int(area[2] - eps)
    area_ex[3] = int(area[3] + eps)
    
    return area_ex

def get_local_area(landmark, expand_param=0.005):
    eye_area = get_eye_area(landmark)
    nose_area = get_nose_area(landmark)
    mouse_area = get_mouse_area(landmark)
    eye_area_ex = area_expand(eye_area, size = 1024, param = expand_param)
    nose_area_ex = area_expand(nose_area, size = 1024, param = expand_param)
    mouse_area_ex = area_expand(mouse_area, size = 1024, param = expand_param)
    return [eye_area_ex, nose_area_ex, mouse_area_ex]

def get_face_matrix(ori_img, loc = 'face', landmark=None):
    # loc: face, left_eye, right_eye, nose_tip, mouse, outlie, or list[]    
    face_lmk_fa = [i for i in range(17)] + [78, 74, 79 ,73, 72, 80, 71, 70, 69, 68, 76, 75, 77]
    face_lmk_le = [i for i in range(17,22)] + [39, 40, 41, 36]
    face_lmk_re = [i for i in range(22,27)] + [45,46,47,42]
    face_lmk_nt = [i for i in range(30,36)]
    face_lmk_mo = [i for i in range(48,60)]
    
    face_lmk = dict()
    face_lmk.update({'face': face_lmk_fa})
    face_lmk.update({'left_eye': face_lmk_le})
    face_lmk.update({'right_eye': face_lmk_re})
    face_lmk.update({'nose_tip': face_lmk_nt})
    face_lmk.update({'mouse': face_lmk_mo})
    face_lmk.update({'outlie': face_lmk_fa})
        
    if landmark == None:
        landmark, _ = get_landmark(ori_img)
    
    img_tmp = Image.new('L', (1024, 1024), 0)
    
    if isinstance(loc, list):
        for pos in loc:
            face_pt = []
            for idx in face_lmk[pos]:
                pt = landmark.parts()[idx]
                face_pt.append((pt.x, pt.y))
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=None, fill=1)
    elif isinstance(loc, str):
        face_pt = []
        for idx in face_lmk[loc]:
            pt = landmark.parts()[idx]
            face_pt.append((pt.x, pt.y))
        if loc == 'outlie':
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=1, fill=None)
        else:
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=None, fill=1)
    convex_mask = np.asarray(img_tmp)
    
    face_matrix = []
    
    for i in range(convex_mask.shape[0]):
        for j in range(convex_mask.shape[1]):
            if convex_mask[i,j] != 0:
                face_matrix.append((i,j))
    
    return convex_mask, face_matrix

def random_gradient_color(size, low=0, high=255):
    # 生成随机渐变色
    # size: 渐变色图像大小  low: 最小色值  high: 最大色值, 偶数
    radius = int(size/2) # 渐变色块半径
    assert radius*2 == size
    
    arr = np.zeros((radius, radius, 3),dtype=np.uint8) # 渐变色块右下角
    #bed = int(np.sqrt(arr.shape[0]**2+arr.shape[1]**2))

    center_color = np.random.randint(low, high, size=(3))
    center_color = tuple([int(x) for x in center_color])  #设置为整数
    #print('color ', center_color)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r = int(np.sqrt(i**2+j**2)) # 距离中心点半径
            partial = 1 - r / radius
            if partial > 0:
                arr[i,j,0]= int(center_color[0]*partial) # 像素值递减
                arr[i,j,1] = int(center_color[1]*partial)
                arr[i,j,2] = int(center_color[2]*partial)
            else:
                continue

    GC11 = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB ) # 渐变色块右下角
    GC10 = cv2.flip(GC11, 1) # 渐变色块左下角

    GC1 = np.concatenate((GC10, GC11), 1) # 渐变色块下部
    GC0 = cv2.flip(GC1, 0) # 渐变色块上部
    GC = np.concatenate((GC0, GC1), 0)
    return GC

def random_resize(img, scale):
    # 随机缩放
    scales = np.random.randint(low=int(scale/2), high=scale*2, size=(2))
    #print('scales ', scales)
    imgr = cv2.resize(img, (scales[0], scales[1]))
    return imgr

def random_rotate_image(img, crop=False):
    '''
    随机旋转
    angle_vari是旋转角度的范围[-angle_vari, angle_vari)
    p_crop是要进行去黑边裁剪的比例
    '''
    angle = np.random.uniform(-360, 360)
    #print('angle ', angle)
    h, w = img.shape[:2]
    
    if h<w: # 添加空白
        blank = np.zeros((int((w-h)/2),w,3), dtype=np.uint8)
        img = np.concatenate((blank, img, blank),0)
        h = int((w-h)/2)*2
    elif h>w:
        blank = np.zeros((h,int((h-w)/2),3), dtype=np.uint8)
        img = np.concatenate((blank, img, blank),1)
        w = int((h-w)/2)*2    
    h, w = img.shape[:2]
    
    angle %= 360# 旋转角度的周期是360°
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)# 用OpenCV内置函数计算仿射矩阵
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))# 得到旋转后的图像
    if crop:# 如果需要裁剪去除黑边
        angle_crop = angle % 180# 对于裁剪角度的等效周期是180°
        if angle_crop > 90:# 并且关于90°对称
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180.0# 转化角度为弧度
        hw_ratio = float(h) / float(w)# 计算高宽比
        
        tan_theta = np.tan(theta)# 计算裁剪边长系数的分子项
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        r = hw_ratio if h > w else 1 / hw_ratio# 计算分母项中和宽高比相关的项
        denominator = r * tan_theta + 1# 计算分母项
        crop_mult = numerator / denominator# 计算最终的边长系数
        w_crop = int(round(crop_mult*w))# 得到裁剪区域
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = img[y0:y0+h_crop, x0:x0+w_crop]
    return img_rotated

def random_crop(img, area_ratio, hw_vari):
    '''
    随机裁剪
    area_ratio为裁剪画面占原画面的比例
    hw_vari是扰动占原高宽比的比例范围
    '''
    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
    
    # 下标进行裁剪，宽高必须是正整数
    w_crop = int(round(w*np.sqrt(area_ratio*hw_mult)))
    
    # 裁剪宽度不可超过原图可裁剪宽度
    if w_crop > w:
        w_crop = w
        
    h_crop = int(round(h*np.sqrt(area_ratio/hw_mult)))
    if h_crop > h:
        h_crop = h
    
    # 随机生成左上角的位置
    x0 = np.random.randint(0, w-w_crop+1)
    y0 = np.random.randint(0, h-h_crop+1)

    return img[y0:y0+h_crop, x0:x0+w_crop]

def hsv_transform(img, hue_delta, sat_mult, val_mult):
    '''
    定义hsv变换函数：
    hue_delta是色调变化比例
    sat_delta是饱和度变化比例
    val_delta是明度变化比例
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    '''
    随机hsv变换
    hue_vari是色调变化比例的范围
    sat_vari是饱和度变化比例的范围
    val_vari是明度变化比例的范围
    '''
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)

def gamma_transform(img, gamma):
    # gamma变换函数
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    # 随机gamma变换
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

def random_transform(img, num_high):
    # 随机执行拉伸-旋转变换
    num = np.random.randint(low=1, high=num_high)
    scale=max(img.shape[0], img.shape[1])
    for i in range(num):
        img = random_resize(img, scale)
        img = random_rotate_image(img, crop=False)
    img = random_rotate_image(img, crop=True)
    return img
    
def add_patch_GC(ori_img, eps=100, prob_num = [10,2,4], sl = [10, 200], loc = 'local'):
    """
    Add scattered gradient color patch to image
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        prob_num (list): 各区域可能的噪声数目，-1则不加噪声, 极端值会造成连续噪声
        loc (string): "local" for areas (eye, nose, mouse), "global" for whole face area
        sl: patch length, random int [sl_low, sl_high] or user specified.
    """
    h, w, c = ori_img.shape
    mask = np.zeros([h, w, c])

    landmark = get_landmark(ori_img)
    if loc == 'local':
        areas = get_local_area(landmark, expand_param = 0.005) # 获取高频区域
    elif loc == 'global':
        areas = [get_face_area(landmark)]
    
    for (idx, area) in enumerate(areas):
        num_points = (area[1]-area[0])*(area[3]-area[2])
        for i in range(area[0],area[1]):
            for j in range(area[2],area[3]):
                if np.random.rand() < prob_num[idx]/num_points: # 取随机点
                    
                    GC = random_gradient_color(256, 0, 100) # 随机渐变色
                    GC = random_transform(GC) # 随机变缓
                    scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
                    GC = cv2.resize(GC, (scales[0],scales[1])) # 随机缩放
                    
                    centerx = i
                    centery = j
                    
                    tx = max(int(centerx - GC.shape[0]/2), 0) # top x
                    ly = max(int(centery - GC.shape[1]/2), 0) # left y
                    bx = min(tx + GC.shape[0], h) # bottom x
                    ry = min(ly + GC.shape[1], w) # right y
                    #print(tl_x, tl_y, br_x, br_y)
                    
                    mask[tx:bx, ly:ry] += GC[:min(GC.shape[0], h - tx), :min(GC.shape[1], w - ly)] # 添加噪声

    mask = mask.clip(min=-eps, max=eps)
    ori_img = (ori_img + mask).clip(min=0, max=255)
    return ori_img, mask

def add_patch_RC(ori_img, eps=10, prob_num = [100,100,100], sl = [5, 20], loc = 'local'):
    """
    Add scattered rectangle color patch to image
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        prob_num (list): 各区域可能的噪声数目，-1则不加噪声, 极端值会造成连续噪声
        loc (string): "local" for areas (eye, nose, mouse), "global" for whole face area
        sl: patch length, random int [sl_low, sl_high] or user specified.
    """
    h, w, c = ori_img.shape
    mask = np.zeros([h, w, c])
    
    landmark = get_landmark(ori_img)
    if loc == 'local':
        areas = get_local_area(landmark, expand_param = 0.005) # 获取高频区域
    elif loc == 'global':
        areas = [get_face_area(landmark)]
    
    for (idx, area) in enumerate(areas):
        num_points = (area[1]-area[0])*(area[3]-area[2])
        for i in range(area[0],area[1]):
            for j in range(area[2],area[3]):
                if np.random.rand() < prob_num[idx]/num_points: # 取随机点
                    sign = np.where(np.random.random(size=c) <= 0.5, -1, 1)
                    alpha = np.random.randint(low=1, high=eps)
                    scale = np.random.randint(low=sl[0], high=sl[1])
                    top = max(i-scale, 0)
                    bot = min(i+scale, h)
                    lef = max(j-scale, 0)
                    rig = min(j+scale, w)
                    mask[top:bot, lef:rig] += alpha * sign

    mask = mask.clip(min=0, max=eps)
    ori_img = (ori_img + mask).clip(min=0, max=255)
    return ori_img, mask

def add_patch_mix(ori_img, eps = [100, 10], prob_num = [[10,5,5], [10,5,5]], sl = [[10, 200], [5, 20]], loc = ['local', 'local']):
    """
    Add mixed scattered color patch to image
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        prob_num (list): 各区域可能的噪声数目，-1则不加噪声, 极端值会造成连续噪声
        loc (string): "local" for areas (eye, nose, mouse), "global" for whole face area
        sl: patch length, random int [sl_low, sl_high] or user specified.
    """

    _, mask_GC = add_patch_GC(ori_img, eps=eps[0], prob_num = prob_num[0], sl = sl[0], loc = loc[0])
    _, mask_RC = add_patch_RC(ori_img, eps=eps[1], prob_num = prob_num[1], sl = sl[1], loc = loc[1])

    mask = mask_GC + mask_RC

    mask = mask.clip(min=0, max=max(eps))
    ori_img = (ori_img + mask).clip(min=0, max=255)

    return ori_img, mask

def grad_image(ori_img, mode = 'plus'):
    # mode: plus, weighted
    grad_x = cv2.Sobel(ori_img, -1, 1, 0)
    grad_y = cv2.Sobel(ori_img, -1, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    if mode == 'werighted':
        gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    elif mode == 'plus':
        gradxy = gradx + grady
    return gradxy

def add_pixel_RC(ori_img, eps=10, prob=0.5, sl = [1, 5], loc = 'local', th = 50, m=None):
    # 对特定区域点集添加噪声
    # loc: face, left_eye, right_eye, nose_tip, mouse, outlie, local
    h, w, c = ori_img.shape
    mask = np.zeros([h, w, c])
    gradxy = grad_image(ori_img, mode = 'plus')
    
    if m == None:
        if loc == 'local':
            _, m = get_face_matrix(ori_img, loc = ['left_eye', 'right_eye', 'nose_tip', 'mouse']) # 获取高频区域
        elif loc == 'global':
            _, m = get_face_matrix(ori_img, loc = 'face')
        else:
            _, m = get_face_matrix(ori_img, loc = loc)

    for (i,j) in m:
        if np.mean(gradxy[i,j,:]) > th:
            if np.random.rand() < prob: # 取随机点
                sign = np.where(np.random.random(size=c) <= 0.5, -1, 1)
                alpha = np.random.randint(low=1, high=eps)
                scale = np.random.randint(low=sl[0], high=sl[1])
                top = max(i-scale, 0)
                bot = min(i+scale, h)
                lef = max(j-scale, 0)
                rig = min(j+scale, w)

                mask[top:bot, lef:rig] += alpha * sign

    mask = mask.clip(min=0, max=eps)
    ori_img = (ori_img + mask).clip(min=0, max=255)
    return ori_img, mask

def add_pixel_GC(ori_img, eps=100, prob=0.5, sl = [10, 200], loc = 'local', th = 50):
    # 对特定区域点集添加噪声
    # loc: face, left_eye, right_eye, nose_tip, mouse, outlie, local
    h, w, c = ori_img.shape
    mask = np.zeros([h, w, c])
    gradxy = grad_image(ori_img, mode = 'plus')
    
    if m == None:
        if loc == 'local':
            _, m = get_face_matrix(ori_img, loc = ['left_eye', 'right_eye', 'nose_tip', 'mouse']) # 获取高频区域
        elif loc == 'global':
            _, m = get_face_matrix(ori_img, loc = 'face')
        else:
            _, m = get_face_matrix(ori_img, loc = loc)
    
    for (i,j) in m:
        if np.mean(gradxy[i,j,:]) > th:
            if np.random.rand() < prob: # 取随机点
                
                GC = random_gradient_color(256, 0, 100) # 随机渐变色
                GC = random_transform(GC) # 随机变缓
                scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
                GC = cv2.resize(GC, (scales[0],scales[1])) # 随机缩放
                
                centerx = i
                centery = j
                
                tx = max(int(centerx - GC.shape[0]/2), 0) # top x
                ly = max(int(centery - GC.shape[1]/2), 0) # left y
                bx = min(tx + GC.shape[0], h) # bottom x
                ry = min(ly + GC.shape[1], w) # right y
                #print(tl_x, tl_y, br_x, br_y)
                
                mask[tx:bx, ly:ry] += GC[:min(GC.shape[0], h - tx), :min(GC.shape[1], w - ly)] # 添加噪声

    mask = mask.clip(min=-eps, max=eps)
    ori_img = (ori_img + mask).clip(min=0, max=255)
    return ori_img, mask

def add_pixel_mix(ori_img, eps = [100, 10], prob = [0.5, 0.5], sl = [[10, 200], [5, 20]], loc = ['local', 'local'], th = 50):
    # 对特定区域点集添加噪声
    # loc: face, left_eye, right_eye, nose_tip, mouse, outlie, local
    _, mask_GC = add_pixel_GC(ori_img, eps=eps[0], prob = prob[0], sl = sl[0], loc = loc[0], th = th)
    _, mask_RC = add_pixel_RC(ori_img, eps=eps[1], prob = prob[1], sl = sl[1], loc = loc[1], th = th)

    mask = mask_GC + mask_RC

    mask = mask.clip(min=0, max=max(eps))
    ori_img = (ori_img + mask).clip(min=0, max=255)

    return ori_img, mask

def add_pixel_GC_continue(ori_img, eps=100, prob=0.5, sl = [10, 200], loc = 'local', GC_size = 256, th = 50, face_matrix=None, image_post = True):
    """
    Add scattered gradient color patch to image
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        prob: 噪声添加概率
        sl: patch length, random int [sl_low, sl_high] or user specified.
        loc (string): "local" for areas (eye, nose, mouse), "global" for whole face area
        gcs: 生成初始渐变色块的size
        th:阈值
    """
    # 对特定区域点集添加连续噪声
    # loc: face, left_eye, right_eye, nose_tip, mouse, outlie, local
    h, w, c = ori_img.shape
    mask = np.zeros([h, w, c])
    gradxy = grad_image(ori_img, mode = 'plus')
    
    if face_matrix == None:
        if loc == 'local':
            _, face_matrix = get_face_matrix(ori_img, loc = ['left_eye', 'right_eye', 'nose_tip', 'mouse']) # 获取高频区域
        elif loc == 'global':
            _, face_matrix = get_face_matrix(ori_img, loc = 'face')
        else:
            _, face_matrix = get_face_matrix(ori_img, loc = loc)
    
    GCs = [] # 渐变色块
    mask_poss = [] # 噪声添加位置，坐上角、右下角坐标
    GC_poss = [] # 渐变色块右下角坐标
    
    for (i,j) in face_matrix:
        if np.mean(gradxy[i,j,:]) > th:
            if np.random.rand() < prob: # 取随机点
                
                GC = random_gradient_color(GC_size, 0, 100) # 随机渐变色
                GC = random_transform(GC, 2) # 随机变换
                scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
                GC = cv2.resize(GC, (scales[0],scales[1])) # 随机缩放
                
                centerx = i
                centery = j
                
                tx = max(int(centerx - GC.shape[0]/2), 0) # top x
                ly = max(int(centery - GC.shape[1]/2), 0) # left y
                bx = min(tx + GC.shape[0], h) # bottom x
                ry = min(ly + GC.shape[1], w) # right y
                
                #mask[tx:bx, ly:ry] += GC[:min(GC.shape[0], h - tx), :min(GC.shape[1], w - ly)] # 添加噪声
                
                GCs.append(GC)
                mask_poss.append((tx,bx,ly,ry))
                GC_poss.append((min(GC.shape[0], h - tx), min(GC.shape[1], w - ly)))
    
    if len(GCs) > 0:
        select_idx = np.random.randint(low = 0, high = len(GCs))
        GC_m = GCs[select_idx] # 随机选取一个色块作为连续主色块
        for (GC, mask_pos, GC_pos) in zip (GCs, mask_poss, GC_poss):
            (tx,bx,ly,ry) = mask_pos
            (GC_x1, GC_y1) = GC_pos
            if np.random.rand() < 0.8: # 添加连续主色块
                GC_t = cv2.resize(GC_m, (GC.shape[1], GC.shape[0]))
                mask[tx:bx, ly:ry] += GC_t[:GC_x1, :GC_y1] # 添加噪声
            else: # 添加其他色块
                mask[tx:bx, ly:ry] += GC[:GC_x1, :GC_y1] # 添加噪声

    mask = mask.clip(min=-eps, max=eps)
    GC_img = (ori_img.astype(np.uint32) + mask).clip(min=0, max=255)
    if image_post :
        return GC_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return GC_img.astype(np.uint32), mask.astype(np.uint32)


'''
************************************************************************************************************************************************
优化GC process:
处理10张1024x1024大小图像的用时/秒(优化前,优化后):
33.54, 8.24
'''
def get_face_matrix_opt(ori_img, loc = 'face', landmark=None, size = 1024): # 优化后
    # loc: face, left_eye, right_eye, nose_tip, mouse, outlie, or list[]    
    face_lmk_fa = [i for i in range(17)] + [78, 74, 79 ,73, 72, 80, 71, 70, 69, 68, 76, 75, 77]
    face_lmk_le = [i for i in range(17,22)] + [39, 40, 41, 36]
    face_lmk_re = [i for i in range(22,27)] + [45,46,47,42]
    face_lmk_nt = [i for i in range(30,36)]
    face_lmk_mo = [i for i in range(48,60)]
    
    face_lmk = dict()
    face_lmk.update({'face': face_lmk_fa})
    face_lmk.update({'left_eye': face_lmk_le})
    face_lmk.update({'right_eye': face_lmk_re})
    face_lmk.update({'nose_tip': face_lmk_nt})
    face_lmk.update({'mouse': face_lmk_mo})
    face_lmk.update({'outlie': face_lmk_fa})
        
    if landmark == None:
        landmark, _ = get_landmark(ori_img)
    
    img_tmp = Image.new('L', (size, size), 0)
    
    if isinstance(loc, list):
        for pos in loc:
            face_pt = []
            for idx in face_lmk[pos]:
                pt = landmark.parts()[idx]
                face_pt.append((pt.x, pt.y))
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=None, fill=1)
    elif isinstance(loc, str):
        face_pt = []
        for idx in face_lmk[loc]:
            pt = landmark.parts()[idx]
            face_pt.append((pt.x, pt.y))
        if loc == 'outlie':
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=1, fill=None)
        else:
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=None, fill=1)
    convex_mask = np.asarray(img_tmp) # 高频区域mask，高频区域点值为1，其余为0
    
    # face_matrix = []
    
    # for i in range(convex_mask.shape[0]):
    #     for j in range(convex_mask.shape[1]):
    #         if convex_mask[i,j] != 0:
    #             face_matrix.append((i,j))
    
    idxs_tmp = np.where(convex_mask != 0)
    face_matrix = np.concatenate((np.expand_dims(idxs_tmp[0], axis=-1), np.expand_dims(idxs_tmp[1], axis=-1)), axis = -1) # (len, 2)
    
    return convex_mask, face_matrix

def get_GC_scaler(size):
    radius = int(size/2) # 渐变色块半径
    assert radius*2 == size
    GC_scaler_11 = np.zeros((radius, radius)) # 渐变色块scaller右下角 (radius, radius)

    for i in range(GC_scaler_11.shape[0]):
        for j in range(GC_scaler_11.shape[1]):
            r = int(np.sqrt(i**2+j**2)) # 距离中心点半径
            partial = 1 - r / radius
            if partial > 0:
                GC_scaler_11[i,j]= partial # 像素值递减
            else:
                continue
    GC_scaler_10 = cv2.flip(GC_scaler_11, 1) # 渐变色块scaller左下角
    GC_scaler_1 = np.concatenate((GC_scaler_10, GC_scaler_11), 1) # 渐变色块scaller下部
    GC_scaler_0 = cv2.flip(GC_scaler_1, 0) # 渐变色块scaller上部
    GC_scaler = np.concatenate((GC_scaler_0, GC_scaler_1), 0) # 渐变色块scaller(size, size)

    GC_scaler = np.expand_dims(GC_scaler, axis=-1)
    GC_scaler = np.repeat(GC_scaler, 3, axis = -1) # (size, size, 3)
    return GC_scaler

def random_gradient_color_opt(size, GC_scaler, low=0, high=255, image_post = True):
    GC = np.ones((size, size, 3))
    assert GC.shape == GC_scaler.shape
    center_color = np.random.randint(low, high, size=(3))
    GC = GC * center_color * GC_scaler
    if image_post :
        return GC.astype(np.uint8)
    else:
        return GC.astype(np.int32)

def add_pixel_GC_continue_opt(ori_img, eps=100, prob=0.5, sl = [10, 200], loc = 'local', GC_size = 256, th = 50, face_matrix=None, GC_scaler=None, image_post = True):
    """
    Add scattered gradient color patch to image
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        prob: 噪声添加概率
        sl: patch length, random int [sl_low, sl_high] or user specified.
        loc (string): "local" for areas (eye, nose, mouse), "global" for whole face area
        GC_size: 生成初始渐变色块的size
        th: 高频点梯度阈值
        face_matrix: 高频区域矩阵
        GC_scaler: 生成随机渐变色块的变化矩阵
        image_post: 是否处理输出图像为cv2格式
    """
    # 对特定区域点集添加连续噪声
    # loc: face, left_eye, right_eye, nose_tip, mouse, outlie, local
    h, w, c = ori_img.shape
    mask = np.zeros([h, w, c])
    gradxy = grad_image(ori_img, mode = 'plus')
    
    if face_matrix is None:
        face_matrix_size = np.max(ori_img.shape)
        if loc == 'local':
            _, face_matrix = get_face_matrix_opt(ori_img, loc = ['left_eye', 'right_eye', 'nose_tip', 'mouse'], size = face_matrix_size) # 获取高频区域
        elif loc == 'global':
            _, face_matrix = get_face_matrix_opt(ori_img, loc = 'face', size = face_matrix_size)
        else:
            _, face_matrix = get_face_matrix_opt(ori_img, loc = loc, size = face_matrix_size)
    
    if GC_scaler is None:
        GC_scaler = get_GC_scaler(GC_size)
    
    GCs = [] # 渐变色块
    mask_poss = [] # 噪声添加位置，坐上角、右下角坐标
    GC_poss = [] # 渐变色块右下角坐标

    gradxy_m = gradxy[face_matrix[:,0],face_matrix[:,1],:] # 梯度图中的高频区域点
    gradxy_m_idx = np.where(np.mean(gradxy_m, axis=-1)>th)[0] # 高频区域点梯度值大于阈值的索引
    idxs_i = face_matrix[:,0][gradxy_m_idx] # 高频区域点梯度值大于阈值的下标索引
    idxs_j = face_matrix[:,1][gradxy_m_idx]
    idx_m = np.concatenate((np.expand_dims(idxs_i,axis=-1), np.expand_dims(idxs_j,axis=-1)), axis = -1) # 下标索引2d矩阵(len,len)
    num_s_m = int(idx_m.shape[0] * prob) # 随机选择高频点的数量
    np.random.shuffle(idx_m)
    idx_selected = idx_m[:num_s_m,:] # 随机选择的高频点

    for idx in idx_selected:
                
        GC = random_gradient_color_opt(GC_size, GC_scaler, 0, 100, image_post = True) # 随机渐变色, np.int32
        GC = random_transform(GC, 2) # 随机变换
        scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
        GC = cv2.resize(GC, (scales[0],scales[1])) # 随机缩放
        
        tx = max(int(idx[0] - GC.shape[0]/2), 0) # top x
        ly = max(int(idx[1] - GC.shape[1]/2), 0) # left y
        bx = min(tx + GC.shape[0], h) # bottom x
        ry = min(ly + GC.shape[1], w) # right y
        
        #mask[tx:bx, ly:ry] += GC[:min(GC.shape[0], h - tx), :min(GC.shape[1], w - ly)] # 添加噪声
        
        GCs.append(GC)
        mask_poss.append((tx,bx,ly,ry))
        GC_poss.append((min(GC.shape[0], h - tx), min(GC.shape[1], w - ly)))
    
    if len(GCs) > 0:
        select_idx = np.random.randint(low = 0, high = len(GCs))
        GC_m = GCs[select_idx] # 随机选取一个色块作为连续主色块
        for (GC, mask_pos, GC_pos) in zip (GCs, mask_poss, GC_poss):
            (tx,bx,ly,ry) = mask_pos
            (GC_x1, GC_y1) = GC_pos
            if np.random.rand() < 0.8: # 添加连续主色块
                GC_t = cv2.resize(GC_m, (GC.shape[1], GC.shape[0]))
                mask[tx:bx, ly:ry] += GC_t.astype(np.uint32)[:GC_x1, :GC_y1] # 添加噪声
            else: # 添加其他色块
                mask[tx:bx, ly:ry] += GC.astype(np.uint32)[:GC_x1, :GC_y1] # 添加噪声

    mask = mask.clip(min=-eps, max=eps)
    GC_img = (ori_img.astype(np.uint32) + mask).clip(min=0, max=255)
    if image_post :
        return GC_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return GC_img.astype(np.uint32), mask.astype(np.uint32)

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


def compute_cam(ori_img, cam_model, cam_extractor):
    #compute_cam
    # model = resnet50(pretrained=True).eval()
    img = ori_img.transpose((2,0,1))
    img = torch.tensor(img)
    input_tensor_cam = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # cam_extractor =SmoothGradCAMpp(model,"layer4")
    if torch.cuda.is_available():
        # net.cuda()
        input_tensor_cam = input_tensor_cam.cuda()
    out = cam_model(input_tensor_cam.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    res1 = to_pil_image(activation_map[0].squeeze(0), mode='F')
    #result = overlay_mask(to_pil_image(img), res1 , alpha=0.5)
    cmap = matplotlib.colormaps.get_cmap('jet')
    # Resize mask and apply colormap
    overlay = res1.resize(to_pil_image(img).size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    #overlay[x][y][0] > 126 即为红色区域
    #print(overlay.shape)
    #print(type(overlay))
    #cam = Image.fromarray(overlay)
    #cam.save(f"./data/{t}_cam.png")
    #plt.imshow(overlay); plt.axis('off'); plt.tight_layout(); plt.show()
    return overlay


def compute_sailency(ori_img, saliency_model):
    #compute sailency map 
    img = ori_img.transpose((2,0,1))
    img = img/np.max(img)
    img = torch.tensor(img)
    img = normalize(resize(img, (256, 256)) , [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	#print(img)
    img = img.unsqueeze(0)
    inputs_test = img.type(torch.FloatTensor)
    # model_dir = './BASNet/saved_models/basnet.pth'
    # net = BASNet(3,1)
    # net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        # net.cuda()
        inputs_test = inputs_test.cuda()
    # net.eval()
    d1,d2,d3,d4,d5,d6,d7,d8 = saliency_model(inputs_test)
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    imo = im.resize((ori_img.shape[1],ori_img.shape[0]),resample=Image.BILINEAR)
    #imo.save(f"./data/{i}_sailency.png")
    pb_np = np.array(imo)
    del d1,d2,d3,d4,d5,d6,d7,d8
    return pb_np	

def get_area_mask(ori_img, cam_model, saliency_model, cam_extractor):
    cam = compute_cam(ori_img, cam_model, cam_extractor)
    sailency = compute_sailency(ori_img, saliency_model)
    #temp = np.zeros((cam.shape[0],cam.shape[1],cam.shape[2]))
    cam_0 = cam[...,0]
    cam_1 = cam[...,1]
    sailency_0 = sailency[...,0]
    cam_0 = (cam_0 > 125)
    cam_1 = (cam_1 ==255)
    sailency_0 = (sailency_0 != 0)
    res = np.bitwise_or(cam_0,cam_1)
    res = np.bitwise_or(res,sailency_0)
    #res = res.astype(np.int8)
    #print(len(res[res==1]))
    #temp[...,0] = 255*res
    """
    
    for x in range(cam.shape[0]):
        for y in range(cam.shape[1]):
            if (cam[x][y][0] >125 or cam[x][y][1] == 255) and sailency[x][y][0]!=0:
                res[x][y][0] = cam[x][y][0]
    """
    mask3d = np.expand_dims(res, -1)
    mask3d = np.repeat(mask3d, 3, -1 )
    
    return mask3d

def add_grad_noise_GC(ori_img, eps=100, prob=0.5, sl = [10, 200], mode = 'GC_n', GC_size = 256, th = 50, GC_scaler=None, image_post = True, mask_area=False, mask = None, cam_model=None, saliency_model=None, cam_extractor=None):
    """
    Add scattered gradient color patch to image
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        prob: 噪声添加概率
        sl: patch length, random int [sl_low, sl_high] or user specified.
        mode (string): "GC_n","RC_point","RC_block","RC_mix","mix"
        GC_size: 生成初始渐变色块的size
        th: 高频点梯度阈值
        GC_scaler: 生成随机渐变色块的变化矩阵
        image_post: 是否处理输出图像为cv2格式
    """
    # 对高频点集添加连续噪声
    h, w, c = ori_img.shape
    # mask = np.zeros([h, w, c])
    GC_mask = np.zeros([h, w, c])
    point_width_mask = np.zeros([h, w, c])
    block_width_mask = np.zeros([h, w, c])
    gradxy = grad_image(ori_img, mode = 'plus')
    
    if GC_scaler is None:
        GC_scaler = get_GC_scaler(GC_size)
        
    if mask_area:
        if mask is None:
            area_mask = get_area_mask(ori_img, cam_model, saliency_model, cam_extractor)
        else:
            area_mask = mask
        gradxy *= area_mask

    idxs_i, idxs_j = np.where(np.mean(gradxy, axis=-1)>th) # 高频区域点梯度值大于阈值的索引
    idx_m = np.concatenate((np.expand_dims(idxs_i,axis=-1), np.expand_dims(idxs_j,axis=-1)), axis = -1) # 下标索引2d矩阵(len,len)
    num_s_m = int(idx_m.shape[0] * prob) # 随机选择高频点的数量
    np.random.shuffle(idx_m)
    idx_selected = idx_m[:num_s_m,:] # 随机选择的高频点

    for idx in idx_selected:
                
        if mode == 'GC_n':
            GC = random_gradient_color_opt(GC_size, GC_scaler, 0, 100, image_post = True) # 随机渐变色, np.int32
            GC = random_transform(GC, 2) # 随机变换
            scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
            GC = cv2.resize(GC, (scales[0],scales[1])) # 随机缩放
            
            tx = max(int(idx[0] - GC.shape[0]/2), 0) # top x
            ly = max(int(idx[1] - GC.shape[1]/2), 0) # left y
            bx = min(tx + GC.shape[0], h) # bottom x
            ry = min(ly + GC.shape[1], w) # right y
            
            (GC_x1, GC_y1) = (min(GC.shape[0], h - tx), min(GC.shape[1], w - ly))
            GC_mask[tx:bx, ly:ry] += GC.astype(np.uint32)[:GC_x1, :GC_y1] # 添加噪声
        
        elif mode in ["RC_point","RC_block","RC_mix","RC_point_mask","RC_block_mask","RC_mix_mask","mix_mask"]:

            sign = np.where(np.random.random(size=c) <= 0.5, -1, 1)
            alpha = np.random.randint(low=1, high=eps)
            scale = np.random.randint(low=sl[0], high=sl[1])
            top = max(idx[0]-scale, 0)
            bot = min(idx[0]+scale, h)
            lef = max(idx[1]-scale, 0)
            rig = min(idx[1]+scale, w)

            block_width_mask[top:bot, lef:rig] += alpha * sign
            point_width_mask[idx[0]][idx[1]] += alpha * sign
        
        elif mode == "mix":
            if np.random.rand() < 0.5: # 取随机点
                GC = random_gradient_color_opt(GC_size, GC_scaler, 0, 100, image_post = True) # 随机渐变色, np.int32
                GC = random_transform(GC, 2) # 随机变换
                scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
                GC = cv2.resize(GC, (scales[0],scales[1])) # 随机缩放
                
                tx = max(int(idx[0] - GC.shape[0]/2), 0) # top x
                ly = max(int(idx[1] - GC.shape[1]/2), 0) # left y
                bx = min(tx + GC.shape[0], h) # bottom x
                ry = min(ly + GC.shape[1], w) # right y
                
                (GC_x1, GC_y1) = (min(GC.shape[0], h - tx), min(GC.shape[1], w - ly))
                GC_mask[tx:bx, ly:ry] += GC.astype(np.uint32)[:GC_x1, :GC_y1] # 添加噪声
                
            else:
                
                sign = np.where(np.random.random(size=c) <= 0.5, -1, 1)
                alpha = np.random.randint(low=1, high=eps)
                scale = np.random.randint(low=sl[0], high=sl[1])
                top = max(idx[0]-scale, 0)
                bot = min(idx[0]+scale, h)
                lef = max(idx[1]-scale, 0)
                rig = min(idx[1]+scale, w)

                block_width_mask[top:bot, lef:rig] += alpha * sign
                point_width_mask[idx[0]][idx[1]] += alpha * sign
                
        else:
            print("unknown mode!")
            raise Exception

    if mode == 'GC_n':
        mask = GC_mask
    elif mode in ["RC_point","RC_point_mask"]:
        mask = point_width_mask
    elif mode in ["RC_block","RC_block_mask"]:
        mask = block_width_mask
    elif mode in ["RC_mix","RC_mix_mask"]:
        mask = point_width_mask + block_width_mask
    elif mode in ["mix","mix_mask"]:
        mask = GC_mask + point_width_mask + block_width_mask
        
    mask = mask.clip(min=-eps, max=eps)
    GC_img = (ori_img.astype(np.uint32) + mask).clip(min=0, max=255)
    if image_post :
        return GC_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return GC_img.astype(np.uint32), mask.astype(np.uint32)

def add_rand_noise_rand_area(ori_img, area_scale = 10, eps=100, prob=0.5, sl = [10, 200], mode = 'GC_n', GC_size = 256, th = 50, GC_scaler=None, image_post = True):
    """
    Add scattered gradient color patch to image
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        prob: 噪声添加概率
        sl: patch length, random int [sl_low, sl_high] or user specified.
        mode (string): "GC_n","RC_point","RC_block","RC_mix","mix"
        GC_size: 生成初始渐变色块的size
        th: 高频点梯度阈值
        GC_scaler: 生成随机渐变色块的变化矩阵
        image_post: 是否处理输出图像为cv2格式
    """
    
    mask_area = get_random_area(ori_img, area_scale) # 生成随机区域mask
    # masked_img = ori_img * mask_area
    
    # 对高频点集添加连续噪声
    h, w, c = ori_img.shape
    # mask = np.zeros([h, w, c])
    GC_mask = np.zeros([h, w, c])
    point_width_mask = np.zeros([h, w, c])
    block_width_mask = np.zeros([h, w, c])
    gradxy = grad_image(ori_img, mode = 'plus')
    gradxy = gradxy.astype(np.int32) * mask_area
    
    if GC_scaler is None:
        GC_scaler = get_GC_scaler(GC_size)

    idxs_i, idxs_j = np.where(np.mean(gradxy, axis=-1)>th) # 高频区域点梯度值大于阈值的索引
    idx_m = np.concatenate((np.expand_dims(idxs_i,axis=-1), np.expand_dims(idxs_j,axis=-1)), axis = -1) # 下标索引2d矩阵(len,len)
    num_s_m = int(idx_m.shape[0] * prob) # 随机选择高频点的数量
    np.random.shuffle(idx_m)
    idx_selected = idx_m[:num_s_m,:] # 随机选择的高频点

    for idx in idx_selected:
                
        if mode == 'GC_n':
            GC = random_gradient_color_opt(GC_size, GC_scaler, 0, 100, image_post = True) # 随机渐变色, np.int32
            GC = random_transform(GC, 2) # 随机变换
            scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
            GC = cv2.resize(GC, (scales[0],scales[1])) # 随机缩放
            
            tx = max(int(idx[0] - GC.shape[0]/2), 0) # top x
            ly = max(int(idx[1] - GC.shape[1]/2), 0) # left y
            bx = min(tx + GC.shape[0], h) # bottom x
            ry = min(ly + GC.shape[1], w) # right y
            
            (GC_x1, GC_y1) = (min(GC.shape[0], h - tx), min(GC.shape[1], w - ly))
            GC_mask[tx:bx, ly:ry] += GC.astype(np.uint32)[:GC_x1, :GC_y1] # 添加噪声
        
        elif mode in ["RC_point","RC_block","RC_mix"]:

            sign = np.where(np.random.random(size=c) <= 0.5, -1, 1)
            alpha = np.random.randint(low=1, high=eps)
            scale = np.random.randint(low=sl[0], high=sl[1])
            top = max(idx[0]-scale, 0)
            bot = min(idx[0]+scale, h)
            lef = max(idx[1]-scale, 0)
            rig = min(idx[1]+scale, w)

            block_width_mask[top:bot, lef:rig] += alpha * sign
            point_width_mask[idx[0]][idx[1]] += alpha * sign
        
        elif mode == "mix":
            if np.random.rand() < 0.5: # 取随机点
                GC = random_gradient_color_opt(GC_size, GC_scaler, 0, 100, image_post = True) # 随机渐变色, np.int32
                GC = random_transform(GC, 2) # 随机变换
                scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
                GC = cv2.resize(GC, (scales[0],scales[1])) # 随机缩放
                
                tx = max(int(idx[0] - GC.shape[0]/2), 0) # top x
                ly = max(int(idx[1] - GC.shape[1]/2), 0) # left y
                bx = min(tx + GC.shape[0], h) # bottom x
                ry = min(ly + GC.shape[1], w) # right y
                
                (GC_x1, GC_y1) = (min(GC.shape[0], h - tx), min(GC.shape[1], w - ly))
                GC_mask[tx:bx, ly:ry] += GC.astype(np.uint32)[:GC_x1, :GC_y1] # 添加噪声
                
            else:
                
                sign = np.where(np.random.random(size=c) <= 0.5, -1, 1)
                alpha = np.random.randint(low=1, high=eps)
                scale = np.random.randint(low=sl[0], high=sl[1])
                top = max(idx[0]-scale, 0)
                bot = min(idx[0]+scale, h)
                lef = max(idx[1]-scale, 0)
                rig = min(idx[1]+scale, w)

                block_width_mask[top:bot, lef:rig] += alpha * sign
                point_width_mask[idx[0]][idx[1]] += alpha * sign
                
        else:
            print("unknown mode!")
            raise Exception

    if mode == 'GC_n':
        mask = GC_mask
    elif mode == "RC_point":
        mask = point_width_mask
    elif mode == "RC_block":
        mask = block_width_mask
    elif mode == "RC_mix":
        mask = point_width_mask + block_width_mask
    elif mode == "mix":
        mask = GC_mask + point_width_mask + block_width_mask
        
    mask = mask.clip(min=-eps, max=eps)
    GC_img = (ori_img.astype(np.uint32) + mask).clip(min=0, max=255)
    if image_post :
        return GC_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return GC_img.astype(np.uint32), mask.astype(np.uint32)

'''
************************************************************************************************************************************************
RAP
'''
def RAP(ori_img, alpha=1, eps=5, noise_type="point"):
    # random adversarial noise patch
    noise_type = noise_type.lower()
    try:
        assert noise_type in ["point", "block", "mix"]
    except AssertionError as error:
        raise Exception("noise type should be point or block or mix.")
    
    """
    Arguments:
        ori_img: numpy type input img.
        alpha (float): step size.
        eps (float): maximum perturbation.
        sl: patch length, random int [1, 10] or user specified.
    """
    h, w, c = ori_img.shape
    point_width_mask = np.zeros([h, w, c])
    block_width_mask = np.zeros([h, w, c])
    
    for i in range(h):
        for j in range(w):
            sign = np.where(np.random.random(size=c) <= 0.5, -1, 1)
            sl = np.random.randint(low=1, high=6)
            top = max(i-sl, 0)
            bot = min(i+sl, h)
            lef = max(j-sl, 0)
            rig = min(j+sl, w)
            block_width_mask[top:bot, lef:rig] += alpha * sign
            point_width_mask[i][j] += alpha * sign
       
    if noise_type == "point":
        mask = point_width_mask
    elif noise_type == "block":
        mask = block_width_mask
    else:
        mask = point_width_mask + block_width_mask
    mask = mask.clip(min=-eps, max=eps)
    ori_img = (ori_img + mask).clip(min=0, max=255)
    # return mask
    return ori_img, mask

def rap_process(img, type):
    alpha_dic = [2, 5]
    eps_dic = [5, 10]
    if type == 'random':
        type_dic = ["point", "block", "mix"]
        type_random = np.random.randint(low=0, high=3)
        type = type_dic[type_random]
    para_random = np.random.randint(low=0, high=2)
    img, mask = RAP(img, alpha=alpha_dic[para_random], eps=eps_dic[para_random], noise_type=type)
    return np.uint8(img), mask

'''
************************************************************************************************************************************************
优化RAP process:
处理100张112x112大小图像的用时/秒(优化前,优化后):
point: 16.68, 0.04
block: 16.64 1.06
mix: 17.69, 1.12
'''
def rap_process_opt(ori_img, eps=None, sl=6, noise_type=None, image_post = False, al=3): # 优化后的 RAP process
    '''
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        sl (int): maximum patch length
        noise_type: in ['point', 'block', 'mix']
        image_post: 是否处理输出图像为cv2格式
        al: 控制block RAP的生成亮度, al越大则mask中元素约趋近于eps
    '''
    if eps is None:
        eps_dic = [5, 10]
        para_random = np.random.randint(low=0, high=2)
        eps=eps_dic[para_random]
    if noise_type is None:
        type_dic = ["point", "block", "mix"]
        type_random = np.random.randint(low=0, high=3)
        noise_type=type_dic[type_random]
    img = ori_img.astype(np.int32)
    mask = RAP_opt(img, eps=eps, sl=sl, noise_type=noise_type, al=al)
    mask = mask.clip(min=-eps, max=eps)
    rap_img = (img + mask).clip(min=0, max=255)
    if image_post:
        return rap_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return rap_img.astype(np.int32), mask.astype(np.int32)

def RAP_opt(ori_img, eps=5, sl=1, noise_type='point', al=3): # 优化后的 RAP
    assert noise_type in ['point', 'block', 'mix']
    h, w, c = ori_img.shape
    point_mask = np.random.randint(low = -eps,high = eps+1,size = (h,w,c)) # (h,w,c)，像素噪声矩阵
    if noise_type=='point':
        return point_mask
    al_mask = np.random.randint(low = -al,high = al+1,size = (h,w,c)) # (h,w,c)，alpha噪声矩阵
    sl_mask = np.random.randint(low = 0,high = sl+1,size = (h,w)) # (h,w)，扩散矩阵
    # get sl step mask
    steps = np.max(sl_mask)
    sl_step_mask = np.zeros([steps+1,h,w]).astype(np.int32) # (sl,h,w)
    ex_mask = np.zeros([steps+1,h,w,c]).astype(np.int32) # (sl,h,w,c)
    for i in range(steps):
        step = steps-i
        sl_step_mask[step] = (sl_mask/step).astype(np.int32)
        sl_step_mask_3d = expand_3d(sl_step_mask[step],channel=c) # (h,w,c), 扩充维度
        ex_mask[step] = expansion_matrix_3d(sl_step_mask_3d * al_mask, step) # (h,w,c)，矩阵扩散
        sl_mask -= sl_step_mask[step]*step
    block_mask = np.sum(ex_mask,axis = 0) # (h,w,c)，block mask
    if noise_type=='block':
        return block_mask
    elif noise_type=='mix':
        return point_mask + block_mask
    
def point_RAP(ori_img, eps=5): # 优化后的point RAP
    point_mask = np.random.randint(low = -eps,high = eps+1,size = ori_img.shape)
    point_mask = point_mask.clip(min=-eps, max=eps)
    return point_mask

def block_RAP(ori_img, eps=5, sl=1, al=3): # 优化后的block RAP
    h, w, c = ori_img.shape
    al_mask = np.random.randint(low = -al,high = al+1,size = (h,w,c)) # (h,w,c)，alpha矩阵
    sl_mask = np.random.randint(low = 0,high = sl+1,size = (h,w)) # (h,w)，扩散矩阵
    #point_mask = point_mask.clip(min=-eps, max=eps)
    # get sl step mask
    steps = np.max(sl_mask)
    sl_step_mask = np.zeros([steps+1,h,w]).astype(np.int32) # (sl,h,w)
    ex_mask = np.zeros([steps+1,h,w,c]).astype(np.int32) # (sl,h,w,c)
    for i in range(steps):
        step = steps-i
        sl_step_mask[step] = (sl_mask/step).astype(np.int32)
        sl_step_mask_3d = expand_3d(sl_step_mask[step],channel=c) # (h,w,c)
        ex_mask[step] = expansion_matrix_3d(sl_step_mask_3d*al_mask, step) # (h,w,c)，矩阵扩散
        sl_mask -= sl_step_mask[step]*step
    block_mask = np.sum(ex_mask,axis = 0) # (h,w,c)，block mask
    block_mask = block_mask.clip(min=-eps, max=eps)
    return block_mask

def block_RAP_2D(ori_img, eps=5, sl=1, al=3): # 优化后的block RAP
    h, w, c = ori_img.shape
    al_mask = np.random.randint(low = -al,high = al+1,size = (h,w)) # (h,w)，alpha矩阵
    sl_mask = np.random.randint(low = 0,high = sl+1,size = (h,w)) # (h,w)，扩散矩阵
    # get sl step mask
    steps = np.max(sl_mask)
    sl_step_mask = np.zeros([steps+1,h,w]).astype(np.int32) # (sl,h,w)
    ex_mask = np.zeros([steps+1,h,w]).astype(np.int32) # (sl,h,w)
    for i in range(steps):
        step = steps-i
        sl_step_mask[step] = (sl_mask/step).astype(np.int32)
        sl_step_mask_2d = sl_step_mask[step]
        ex_mask[step] = expansion_matrix_2d(sl_step_mask_2d*al_mask, step) # (h,w)，矩阵扩散
        sl_mask -= sl_step_mask[step]*step
    block_mask = np.sum(ex_mask,axis = 0) # (h,w)，block mask
    block_mask = block_mask.clip(min=0, max=eps) # min=-eps -> 0
    return block_mask

def get_random_area(ori_img, scale=10):
    mask_rap = block_RAP_2D(ori_img, eps=1, sl=scale, al=1)
    mask_rap3 = np.expand_dims(mask_rap, axis=-1)
    mask_rap3 = np.repeat(mask_rap3, 3, axis = -1)
    return mask_rap3

def mix_RAP(ori_img, sl=1, eps=5): # 优化后的mix RAP
    mix_mask = point_RAP(ori_img, eps=eps) + block_RAP(ori_img, sl=sl, eps=eps)
    mix_mask = mix_mask.clip(min=-eps, max=eps)
    return mix_mask

def expand_3d(X,channel=3):
    # 将2d mat 扩充成channel通道3d mat
    assert len(X.shape) == 2
    X_tmp = np.expand_dims(X, axis = -1)
    X_3d = np.repeat(X_tmp, channel,axis = -1)
    return X_3d

def expansion_matrix_2d(X, steps=1):
    # 每个点向四周扩展step, X:2d mat
    # 平移溢出为0
    arr_tmp = X.copy()
    # 向右下角padding
    arr_tmp = np.concatenate((arr_tmp,np.zeros([arr_tmp.shape[0], steps*2])),axis = 1)
    arr_tmp = np.concatenate((arr_tmp,np.zeros([steps*2, arr_tmp.shape[1]])),axis = 0)
    #for tl in range(1,steps+1):
    arr_tmp_down = np.zeros(arr_tmp.shape)
    arr_tmp_right = np.zeros(arr_tmp.shape)
    for tl in range(1, steps*2 + 1):
        down_tl = [i for i in range(arr_tmp.shape[0]-tl)]
        arr_down = np.concatenate((np.zeros((tl,arr_tmp.shape[1])),arr_tmp[down_tl,:]),axis = 0)
        arr_tmp_down += arr_down
    arr_out_down = arr_tmp + arr_tmp_down
    for tl in range(1, steps*2 + 1):
        down_tl = [i for i in range(arr_tmp.shape[1]-tl)]
        arr_right = np.concatenate((np.zeros((arr_out_down.shape[0],tl)),arr_out_down[:,down_tl]),axis = 1)
        arr_tmp_right += arr_right
    arr_out_right = arr_out_down + arr_tmp_right
    #out_mat = arr_tmp + arr_tmp_down + arr_tmp_right
    out_mat = arr_out_right[steps : arr_out_right.shape[0] - steps, steps : arr_out_right.shape[1]-steps]
    return out_mat

def expansion_matrix_3d(X, steps=1):
    # 每个点向四周扩展step, X:3d mat (h,w,c)
    # 平移溢出为0
    #h, w, c = X.shape
    c = X.shape[2]
    arr_tmp = X.copy()
    # 向右下角padding
    arr_tmp = np.concatenate((arr_tmp,np.zeros([arr_tmp.shape[0], steps*2, c])),axis = 1)
    arr_tmp = np.concatenate((arr_tmp,np.zeros([steps*2, arr_tmp.shape[1], c])),axis = 0) # (h+steps*2,w+steps*2,c)
    #for tl in range(1,steps+1):
    arr_tmp_down = np.zeros(arr_tmp.shape) # (h+steps*2,w+steps*2,c)
    arr_tmp_right = np.zeros(arr_tmp.shape) # (h+steps*2,w+steps*2,c)
    for i in range(steps*2): # 向下扩散
        step = i + 1
        down_tl = [k for k in range(arr_tmp.shape[0]-step)]
        arr_down = np.concatenate((np.zeros((step, arr_tmp.shape[1], c)),arr_tmp[down_tl,:,:]),axis = 0) # (h+steps*2,w+steps*2,c)
        arr_tmp_down += arr_down
    arr_out_down = arr_tmp + arr_tmp_down
    for i in range(steps*2): # 向右扩散
        step = i + 1
        down_tl = [k for k in range(arr_tmp.shape[1]-step)]
        arr_right = np.concatenate((np.zeros((arr_tmp.shape[0], step, c)),arr_out_down[:,down_tl,:]),axis = 1) # (h+steps*2,w+steps*2,c)
        arr_tmp_right += arr_right
    arr_out_right = arr_out_down + arr_tmp_right
    #out_mat = arr_tmp + arr_tmp_down + arr_tmp_right
    out_mat = arr_out_right[steps : arr_out_right.shape[0] - steps, steps : arr_out_right.shape[1]-steps, :] # (h,w,c)
    return out_mat

'''
************************************************************************************************************************************************
distribution process:
'''
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

def load_dis(atk_method = 'FGSM', dataset_name = 'CIFAR10', eps = None):
    
    # if dataset_name == 'CIFAR10':
    save_root = 'E:/Files/code/Adv_Defense/Detection/Self-Perturbation/data/dist'
    
    if eps is None:
        eps = 8 if dataset_name == 'CIFAR10' else 4
    params = {'eps':eps, 'alpha':2, 'steps':10, 'gamma':0}

    sub_root = dataset_name + '_' + atk_method + '_eps' + str(params['eps']) + '_noise_dist.pt'
    save_root = os.path.join(save_root, sub_root)
    dis_dict = torch.load(save_root)
    
    print("Loading distribution dict from {}".format(save_root))
    
    return dis_dict

def dis_varient(dis_dict = None, loc_low = -3, loc_high = 3, cov_high = 0.005, cal_dist = False, device = 'cpu'):
    
    if dis_dict is None:
        dis_dict = load_dis(atk_method = 'FGSM', dataset_name = 'CIFAR10', eps = 8)
    
    loc = dis_dict['loc']
    loc_varient = torch.randn(loc.shape)
    loc_scale = loc_low + (loc_high - loc_low) * torch.rand(1)
    new_loc = loc + loc_varient * loc_scale
    
    cov = dis_dict['covariance_matrix'].squeeze(0)
    cov_varient = torch.randn(cov.shape)
    scale = cov_high * torch.rand(1)
    cov_varient = torch.mm(cov_varient, cov_varient.t())
    new_cov = cov + cov_varient * scale
    
    new_dis = torch.distributions.MultivariateNormal(loc=new_loc, covariance_matrix=new_cov) # distribution
    
    if not cal_dist:
        return new_dis
    else:
        # cos_sim = nn.CosineSimilarity(dim=1)
        pdist = nn.PairwiseDistance(p=2)
        loc_dist = torch.cdist(loc, new_loc, p=2)
        cov_dist = pdist(cov, new_cov).mean(0)
        dist = {'loc_dist': loc_dist, 'cov_dist': cov_dist}
        return new_dis, dist
    
def sample(dis, batch_size=1, eps = 8, clip = 1):
    fea = dis.sample([batch_size, ])
    # if fea.shape[1] == 1:
    fea = fea.squeeze(1).squeeze(1)
    
    sign = torch.randint_like(fea, -1, 2)
    fea *= sign
    
    if np.random.rand() < clip:
        fea = fea.clip(min=-eps, max=eps)
    
    shape_noise = int(np.sqrt(fea.shape[-1] / 3))
    
    mask = fea.reshape((shape_noise, shape_noise, 3))
    
    return np.array(mask).astype(np.uint8)

def dis_process(ori_img, dis_dict, eps=None, eps_low=1, eps_high = 10, image_post = False): # 优化后的 DIS process
    '''
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        sl (int): maximum patch length
        noise_type: in ['point', 'block', 'mix']
        image_post: 是否处理输出图像为cv2格式
        al: 控制block RAP的生成亮度, al越大则mask中元素约趋近于eps
    '''
    if eps is None or eps < 0:
        eps = np.random.randint(low=eps_low, high=eps_high + 1)
    img = ori_img.astype(np.int32)

    dis = dis_varient(dis_dict = dis_dict, loc_low = -3, loc_high = 3, cov_high = 0.005, cal_dist = False, device = 'cpu')
    mask = sample(dis, 1, eps, clip=1)
    mask  = cv2.resize(mask, (ori_img.shape[0], ori_img.shape[1])) # (112, 112) for faces
    dis_img = (img + mask.astype(np.int32)).clip(min=0, max=255)
    if image_post:
        return dis_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return dis_img.astype(np.int32), mask.astype(np.int32)

def sample_multi(dis, batch_size=4, eps = 8, clip = 1):
    fea = dis.sample([batch_size, ])
    # if fea.shape[1] == 1:
    fea = fea.squeeze(1).squeeze(1)
    
    sign = torch.randint_like(fea, -1, 2)
    fea *= sign
    
    if np.random.rand() < clip:
        fea = fea.clip(min=-eps, max=eps)
    
    shape_noise = int(np.sqrt(fea.shape[-1] / 3))
    
    mask = fea.reshape((batch_size, shape_noise, shape_noise, 3)) # (b, h,w,c)
    
    return mask

def dis_process_padding(ori_img, dis_dict, eps=None, eps_low=1, eps_high = 10, image_post = False, padding = True, var_loc = 3, var_cov = 0.005): # 优化后的 DIS process
    '''
    Arguments:
        ori_img: numpy type input img.
        eps (float): maximum perturbation.
        sl (int): maximum patch length
        noise_type: in ['point', 'block', 'mix']
        image_post: 是否处理输出图像为cv2格式
        al: 控制block RAP的生成亮度, al越大则mask中元素约趋近于eps
    '''
    if eps is None or eps < 0:
        eps = np.random.randint(low=eps_low, high=eps_high + 1)
    img = ori_img.astype(np.int32)

    dis = dis_varient(dis_dict = dis_dict, loc_low = -var_loc, loc_high = var_loc, cov_high = var_cov, cal_dist = False, device = 'cpu')
    if not padding:
        mask = sample(dis, 1, eps, clip=1)
        mask  = cv2.resize(mask, (ori_img.shape[0], ori_img.shape[1])) # (112, 112) for faces
    else:
        num = int(ori_img.shape[0] / 32) # mask shape 32
        mask_batch = sample_multi(dis, num*num, eps, clip=1)
        maskss = []
        for i in range(num):
            masks = [mask_ij for mask_ij in mask_batch[i*num:(i+1)*num]]
            maskss.append(torch.cat(masks, 1))
        mask = torch.cat(maskss, 0)
        mask = np.array(mask).astype(np.int32)

    dis_img = (img + mask.astype(np.int32)).clip(min=0, max=255)
    if image_post:
        return dis_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return dis_img.astype(np.int32), mask.astype(np.int32)

'''
************************************************************************************************************************************************
'''

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm+1e-8)
    return output

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cal_fam(model, inputs):
    model.zero_grad()
    inputs = inputs.detach().clone()
    inputs.requires_grad_()
    output = model(inputs)

    target = output[:, 1]-output[:, 0]
    target.backward(torch.ones(target.shape).cuda())
    fam = torch.abs(inputs.grad)
    fam = torch.max(fam, dim=1, keepdim=True)[0]
    return fam


def cal_normfam(model, inputs):
    fam = cal_fam(model, inputs)
    _, x, y = fam[0].shape
    fam = torch.nn.functional.interpolate(fam, (int(y/2), int(x/2)), mode='bilinear', align_corners=False)
    fam = torch.nn.functional.interpolate(fam, (y, x), mode='bilinear', align_corners=False)
    for i in range(len(fam)):
        fam[i] -= torch.min(fam[i])
        fam[i] /= torch.max(fam[i])
    return fam


def calRes(y_true_all, y_pred_all):
    y_true_all, y_pred_all = np.array(
        y_true_all.cpu()), np.array(y_pred_all.cpu())

    fprs, tprs, ths = roc_curve(
        y_true_all, y_pred_all, pos_label=1, drop_intermediate=False)

    acc = accuracy_score(y_true_all, np.where(y_pred_all >= 0.5, 1, 0))*100.

    ind = 0
    for fpr in fprs:
        if fpr > 1e-2:
            break
        ind += 1
    TPR_2 = tprs[ind-1]

    ind = 0
    for fpr in fprs:
        if fpr > 1e-3:
            break
        ind += 1
    TPR_3 = tprs[ind-1]

    ind = 0
    for fpr in fprs:
        if fpr > 1e-4:
            break
        ind += 1
    TPR_4 = tprs[ind-1]

    ap = average_precision_score(y_true_all, y_pred_all)
    return ap, acc, auc(fprs, tprs), TPR_2, TPR_3, TPR_4


def shuffle_tensor(input, label):
    l = input.size(0)
    order = torch.randperm(l)
    input = input[order]
    label = label[order]
    return input, label

def get_img_list(data_root):
    # get image list
    file_list = os.listdir(data_root)
    img_list = []
    cls_list = []
    for file in file_list:
        file_dir = os.path.join(data_root, file)
        if os.path.isfile(file_dir):
            if file_dir.endswith('jpg') or file_dir.endswith('png') or file_dir.endswith('JPEG'):
                img_list.append(file_dir)
        elif os.path.isdir(file_dir):
            for sub_file in os.listdir(file_dir):
                sub_file_dir = os.path.join(file_dir, sub_file)
                if sub_file_dir.endswith('jpg') or sub_file_dir.endswith('png') or sub_file_dir.endswith('JPEG'):
                    img_list.append(sub_file_dir)

    return img_list

def rap_process_eps(img, type, eps):
    alpha_dic = [4,8,12]
    if type == 'random':
        type_dic = ["point", "block", "mix"]
        type_random = np.random.randint(low=0, high=3)
        type = type_dic[type_random]
    para_random = np.random.randint(low=0, high=3)
    img, mask = RAP(img, alpha=alpha_dic[para_random], eps=eps, noise_type=type)
    min_value = - np.min(mask) # 应当为eps
    mask = mask + min_value
    return np.uint8(img), mask
    
def run_single(save_img=False):
    
    eps = 10
    num_per = 70
    
    # 抽取图像
    data_root = 'E:/Files/code/Dataset_Face/lfw-sample/single_Hugo_Chavez'
    ori_image_name = 'Hugo_Chavez_0001.jpg'
    ori_image_dir = os.path.join(data_root, 'Origin', ori_image_name)
    ori_img = cv2.imread(ori_image_dir)

    typelist = ["point", "block", "mix"]
    for type in typelist:
        print('RAP processing', type)
        output_dir = 'E:/Files/code/Dataset_Face/lfw-sample/single_Hugo_Chavez/Noise/RAP-' + type
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) # 创建数据保存路径
        for i in tqdm(range(num_per)):
            adv_img, mask = rap_process_eps(ori_img, type, eps)
            if save_img:
                # save image
                # _ = save_adv_image(adv_img, augment+'-eps'+str(eps)+'-'+img_name, output_dir)
                # _ = save_adv_image(adv_img, 'RAP-'+type+str(i)+'-'+ori_image_name, output_dir)
                _ = save_adv_image(mask, 'RAP-noise-'+type+str(i)+'-'+ori_image_name, output_dir)
      
    print('Augmentation Compete')

def run_tmp(save_img=False):
    
    eps = 15
    # num_per = 70
    
    # 抽取图像
    data_root = 'E:/Files/code/Dataset/cifar10/test'
    img_list = get_img_list(data_root)
    total = len(img_list)
    print('Nums of images', total)

    for ori_image_dir in img_list:
        # ori_image_dir = os.path.join(data_root, ori_name)
        ori_name = ori_image_dir.split('/')[-1]
        ori_img = cv2.imread(ori_image_dir)

        typelist = ["point", "block", "mix"]
        for type in typelist:
            print('Self-Perturbation processing', type)
            output_dir = 'E:/Files/code/Dataset_Face/lfw-sample/output1'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) # 创建数据保存路径
            adv_img, mask = rap_process_eps(ori_img, type, eps)
            if save_img:
                # save image
                # _ = save_adv_image(adv_img, augment+'-eps'+str(eps)+'-'+img_name, output_dir)
                _ = save_adv_image(adv_img, 'SP-'+type+'-'+ori_name, output_dir)
                # _ = save_adv_image(np.uint8(mask), 'SP-noise-'+type+'-'+ori_name, output_dir)
                # mask5 = mask*5
                # _ = save_adv_image(np.uint8(mask5), 'SP-noise5-'+type+'-'+ori_name, output_dir)
                # mask10 = mask*10
                # _ = save_adv_image(np.uint8(mask10), 'SP-noise10-'+type+'-'+ori_name, output_dir)
                # mask15 = mask*15
                # _ = save_adv_image(np.uint8(mask15), 'SP-noise15-'+type+'-'+ori_name, output_dir)
                # mask20 = mask*20
                # _ = save_adv_image(np.uint8(mask20), 'SP-noise20-'+type+'-'+ori_name, output_dir)
    
    print('Augmentation Compete')

def run(base_data_root, output_dir, augment, eps, save_img, type):
    # augment_f = get_augmentation(augment)
    # 抽取图像
    img_list = get_img_list(base_data_root)
    total = len(img_list)
    print('Generating pseudo adv images')
    print('Nums of images', total)

    for i in tqdm(range(total)):
        # print('turn ', i)
        # img_name= img_list[i]# 获取图片名
        # # print(images_names)
        # img_dir = os.path.join(base_data_root, img_name)# 读取原图片
        # ori_img = cv2.imread(img_dir)

        ori_image_dir = img_list[i]
        ori_img = cv2.imread(ori_image_dir)
        img_name = str(i) + '.png' 

        # augmentation
        adv_img, _ = rap_process(ori_img, type)
        if save_img:
            # save image
            # _ = save_adv_image(adv_img, augment+'-eps'+str(eps)+'-'+img_name, output_dir)
            _ = save_adv_image(adv_img, augment+'-'+img_name, output_dir)
        
    print('Augmentation Compete')

def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_img", type=bool, default=True,
                        help="save images or not")
    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        help="base dataset name")
    parser.add_argument("--augment_list", type=str, default=['RAP-random'],
                        help="face model ")
    return parser.parse_args()

def main1():
    args = input_args()
    dataset_name = args.dataset
    save_img = args.save_img
    augment_list = args.augment_list
    eps = 5
    typelist = ["point", "block", "mix", "random"]
    # type = typelist[1]
    type = "random"
        
    if dataset_name == 'celebahq':
        base_data_root = 'E:/Files/code/Dataset_Face/CelebA-HQ/train-5000'
        augment_dir = 'E:/Files/code/Dataset_Face/CelebA-HQ-Augmentation'
    elif dataset_name == 'CIFAR10':
        base_data_root = 'E:/Files/code/Dataset/cifar10/train'
        augment_dir = 'E:/Files/code/Dataset/cifar10-Augmentation'

    time0 = time.time()

    for augment in augment_list:
        # output_dir = os.path.join(augment_dir, augment)
        output_dir = os.path.join(augment_dir, 'RAP-'+type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) # 创建数据保存路径
        time1 = time.time()
        run(base_data_root, output_dir, augment, eps, save_img, type)
        time2 = time.time()
        print('Time: ', time2 - time1)

    time2 = time.time()
    print('Total time: ', time2 - time0)

def main():
    args = input_args()
    save_img = args.save_img
    augment_list = args.augment_list
    # eps = 5
    # typelist = ["point", "block", "mix", "random"]
    type = "random"
        
    dataset_name = 'CIFAR10'
    base_data_root = 'E:/Files/code/Dataset/cifar10/train'
    augment_dir = 'E:/Files/code/Dataset/cifar10-Augmentation'

    augment = 'RAP-random'

    
    # cls_list = []
    for cls in os.listdir(base_data_root):
        # get image list
        cls_dir = os.path.join(base_data_root, cls)
        
        if os.path.isdir(cls_dir):
            print('Generating pseudo adv images for ' + cls)
            # cls_list.append(cls)
            img_list = []
            img_names = []
            for file in os.listdir(cls_dir):
                file_dir = os.path.join(cls_dir, file)
                if file_dir.endswith('jpg') or file_dir.endswith('png') or file_dir.endswith('JPEG'):
                    img_names.append(file)
                    img_list.append(file_dir)

            # output_dir = os.path.join(augment_dir, 'RAP-'+type+'-eps'+str(eps), cls)
            output_dir = os.path.join(augment_dir, 'RAP-'+type, cls)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir) # 创建数据保存路径

            total = len(img_list)
            print('Nums of images', total)

            for i in tqdm(range(total)):

                ori_image_dir = img_list[i]
                ori_img = cv2.imread(ori_image_dir)
                img_name = img_names[i]

                # augmentation
                adv_img, _ = rap_process(ori_img, type)
                if save_img:
                    _ = save_adv_image(adv_img, img_name, output_dir)
                
            print('Augmentation Compete')

        else:
            continue
    


if __name__=='__main__':
    main()
    # run_single(save_img=True)
    # run_tmp(save_img=True)