import os
#import torch
import numpy as np
#from torch.autograd import Variable
#from torch import nn
#import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageFilter,ImageDraw
#from torchvision import transforms as tfs
#from datetime import datetime
#import matplotlib.pyplot as plt
#from tqdm import tqdm
#import random
 

image_path = '/remote-home/pxy/data/FUSAR-MAP/OPT/test'
label_path = '/remote-home/pxy/data/FUSAR-MAP/LAB/test'
image_cropped_save_path = '/remote-home/pxy/data/FUSAR-MAP/OPT/test_cropped'
label_crooped_save_path = '/remote-home/pxy/data/FUSAR-MAP/LAB/test_cropped'

if not os.path.exists(image_cropped_save_path):
    os.makedirs(image_cropped_save_path)
if not os.path.exists(label_crooped_save_path):
    os.makedirs(label_crooped_save_path)

crop_width = 256
crop_height = 256
step = 0

image_list = os.listdir(image_path)
image_list.sort()
for img_name in image_list:
    mask_name = img_name.replace('Optical','Label')
    img_path = os.path.join(image_path,img_name)
    mask_path = os.path.join(label_path,mask_name)
    img = Image.open(img_path)
    #mask = Image.open(mask_path)
    for i in range(img.size[0]//crop_width):
        for j in range(img.size[1]//crop_height):
            img_cropped = img.crop((i*crop_width,j*crop_height,(i+1)*crop_width,(j+1)*crop_height))
            img_save_name = img_name.split('.tif')[0]+'_'+str(i)+'_'+str(j)+'.tif'
            img_save_path = os.path.join(image_cropped_save_path,img_save_name)
            img_cropped.save(img_save_path)

            # mask_cropped = mask.crop((i*crop_width,j*crop_height,(i+1)*crop_width,(j+1)*crop_height))
            # mask_save_name = mask_name.split('.tif')[0]+'_'+str(i)+'_'+str(j)+'.tif'
            # mask_save_path = os.path.join(label_crooped_save_path,mask_save_name)
            # mask_cropped.save(mask_save_path)



