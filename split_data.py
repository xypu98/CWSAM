import os
import time
import shutil
import numpy as np

train_path = './load/CAMO/Images/Train/'
test_path = './load/CAMO/Images/Test/'

GT_path = './load/CAMO/GT/'

Train_gt_path = './load/CAMO/Train_gt/'
Test_gt_path = './load/CAMO/Test_gt/'

train_list = os.listdir(train_path)
test_list = os.listdir(test_path)

for img in train_list:
    gt_path =os.path.join(GT_path,img).replace('.jpg','.png')
    train_gt_img = os.path.join(Train_gt_path,img)
    shutil.copy(gt_path,train_gt_img)

for img in test_list:
    gt_path =os.path.join(GT_path,img).replace('.jpg','.png')
    test_gt_img = os.path.join(Test_gt_path,img)
    shutil.copy(gt_path,test_gt_img)