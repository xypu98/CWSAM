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

 

#image_path = '/remote-home/pxy/data/FUSARMAP2/img'
label_path = '/remote-home/pxy/data/FUSARMAP2/mask_cropped'
#image_cropped_save_path = '/remote-home/pxy/data/FUSARMAP2/img_cropped'
label_cropped_save_path = '/remote-home/pxy/data/FUSARMAP2/mask_cropped_5cls'
shutil.rmtree(label_cropped_save_path)

if not os.path.exists(label_cropped_save_path):
    os.makedirs(label_cropped_save_path)

original_class_color = [[0, 0, 255],[0, 139, 0],[0, 255, 0],[139, 0, 0],[255, 0, 0],[205, 173, 0],[83, 134, 139],[0, 139, 139],[139, 105, 20],[189, 183, 107],[178, 34, 34]]

#self.classes =['building','vegetation','water','road']     
new_class = [ [1,0,0], [0,1,0], [0,0,1], [1,1,0]] #background,building,vegetation,water,road

convert_class_dict ={
    tuple([0, 0, 255]):[0, 0, 255],
    tuple([0, 139, 0]):[0, 255, 0],
    tuple([0, 255, 0]):[0, 255, 0],
    tuple([139, 0, 0]):[0,0,0],
    tuple([255, 0, 0]):[255, 0, 0],
    tuple([205, 173, 0]):[255, 0, 0],
    tuple([83, 134, 139]):[255, 255, 0],
    tuple([0, 139, 139]):[0, 255, 0],
    tuple([139, 105, 20]):[0, 255, 0],
    tuple([189, 183, 107]):[255, 255, 0],
    tuple([178, 34, 34]):[0,0,0]
    }

image_list = os.listdir(label_path)
image_list.sort()
for img_name in image_list:
    print(img_name)
    #mask_name = img_name.replace('.tif','_mask.tif')
    img_path = os.path.join(label_path,img_name)
    img = np.array(Image.open(img_path))

    for i in range(1024):
        for j in range(1024):
            #print(img[i,j])
            img[i,j]=np.array(convert_class_dict[tuple(img[i,j])])

    img_save_path = os.path.join(label_cropped_save_path,img_name)
    Image.fromarray(img).save(img_save_path)
    #         img_cropped.save(img_save_path)
           




    # for i in range(img.size[0]//crop_width):
    #     for j in range(img.size[1]//crop_height):
    #         img_cropped = img.crop((i*crop_width,j*crop_height,(i+1)*crop_width,(j+1)*crop_height))
    #         mask_cropped = mask.crop((i*crop_width,j*crop_height,(i+1)*crop_width,(j+1)*crop_height))
    #         img_cropped_np = np.array(img_cropped)
    #         mask_cropped_np = np.array(mask_cropped)
    #         #print(np.shape(mask_cropped_np))
    #         #max_np = max(img_cropped_np)
    #         if np.max(img_cropped_np) <=0:
    #             continue
    #         if np.min(mask_cropped_np.sum(axis=2))<2:
    #             print('filter black label')
    #             continue
    #         img_save_name = img_name.split('.tif')[0]+'_'+str(i)+'_'+str(j)+'.tif'
    #         img_save_path = os.path.join(image_cropped_save_path,img_save_name)
    #         img_cropped.save(img_save_path)
           
    #         mask_save_name = mask_name.split('.tif')[0]+'_'+str(i)+'_'+str(j)+'.tif'
    #         mask_save_path = os.path.join(label_cropped_save_path,mask_save_name)
    #         mask_cropped.save(mask_save_path)
            #print('save_cropped:',img_save_path)

            # mask_cropped = mask.crop((i*crop_width,j*crop_height,(i+1)*crop_width,(j+1)*crop_height))
            # mask_save_name = mask_name.split('.tif')[0]+'_'+str(i)+'_'+str(j)+'.tif'
            # mask_save_path = os.path.join(label_crooped_save_path,mask_save_name)
            # mask_cropped.save(mask_save_path)