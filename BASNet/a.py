import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
from cv2 import imread
import numpy as np
from PIL import Image
from torchvision.transforms.functional import normalize, resize
from model import BASNet


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(image_name)


if __name__ == '__main__':
	# basnet 的输入为1*3*256*256 是什么结构？？
	# --------- 1. get image path and name ---------
	image_dir = '1.JPEG'
	model_dir = './saved_models/basnet.pth'
	img_ori = imread(image_dir)
	img = img_ori.transpose((2,0,1))
	print(img.shape)
	print("...load BASNet...")
	net = BASNet(3,1)
	net.load_state_dict(torch.load(model_dir))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	#缺少数据预处理与精度切换
	img = img/np.max(img)
	img = torch.tensor(img)
	img = normalize(resize(img, (256, 256)) , [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	#print(img)
	img = img.unsqueeze(0)
	inputs_test = img.type(torch.FloatTensor)
	print(img.shape)
		
	d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)
	pred = d1[:,0,:,:]
	pred = normPRED(pred)
	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()
	im = Image.fromarray(predict_np*255).convert('RGB')
	imo = im.resize((img_ori.shape[1],img_ori.shape[0]),resample=Image.BILINEAR)
	path = "./res_sailency/1.png"
	imo.save(path,quality=100)
	pb_np = np.array(imo)
	print(pb_np.shape)
	del d1,d2,d3,d4,d5,d6,d7,d8	
