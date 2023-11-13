import os
#import torch
import numpy as np
import shutil
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
#from PIL import Image
Image.MAX_IMAGE_PIXELS = None

 

image_path = '/remote-home/pxy/data/FUSARMAP2/img'
label_path = '/remote-home/pxy/data/FUSARMAP2/mask'
image_cropped_save_path = '/remote-home/pxy/data/FUSARMAP2/img_cropped'
label_cropped_save_path = '/remote-home/pxy/data/FUSARMAP2/mask_cropped'

shutil.rmtree(image_cropped_save_path)
shutil.rmtree(label_cropped_save_path)

if not os.path.exists(image_cropped_save_path):
    os.makedirs(image_cropped_save_path)
if not os.path.exists(label_cropped_save_path):
    os.makedirs(label_cropped_save_path)

crop_width = 1024
crop_height = 1024
step = 0

image_list = os.listdir(image_path)
image_list.sort()
for img_name in image_list:
    print(img_name)
    mask_name = img_name.replace('.tif','_mask.tif')
    img_path = os.path.join(image_path,img_name)
    mask_path = os.path.join(label_path,mask_name)
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    #mask = Image.open(mask_path)
    for i in range(img.size[0]//crop_width):
        for j in range(img.size[1]//crop_height):
            img_cropped = img.crop((i*crop_width,j*crop_height,(i+1)*crop_width,(j+1)*crop_height))
            mask_cropped = mask.crop((i*crop_width,j*crop_height,(i+1)*crop_width,(j+1)*crop_height))
            img_cropped_np = np.array(img_cropped)
            mask_cropped_np = np.array(mask_cropped)
            #print(np.shape(mask_cropped_np))
            #max_np = max(img_cropped_np)
            if np.max(img_cropped_np) <=0:
                continue
            if np.min(mask_cropped_np.sum(axis=2))<2:
                print('filter black label')
                continue
            img_save_name = img_name.split('.tif')[0]+'_'+str(i)+'_'+str(j)+'.tif'
            img_save_path = os.path.join(image_cropped_save_path,img_save_name)
            img_cropped.save(img_save_path)
           
            mask_save_name = mask_name.split('.tif')[0]+'_'+str(i)+'_'+str(j)+'.tif'
            mask_save_path = os.path.join(label_cropped_save_path,mask_save_name)
            mask_cropped.save(mask_save_path)
            #print('save_cropped:',img_save_path)

            # mask_cropped = mask.crop((i*crop_width,j*crop_height,(i+1)*crop_width,(j+1)*crop_height))
            # mask_save_name = mask_name.split('.tif')[0]+'_'+str(i)+'_'+str(j)+'.tif'
            # mask_save_path = os.path.join(label_crooped_save_path,mask_save_name)
            # mask_cropped.save(mask_save_path)



