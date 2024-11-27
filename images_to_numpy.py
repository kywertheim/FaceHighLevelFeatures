import cv2, os
import sys
sys.path.insert(0, '..')
sys.path.insert(0,'/home/gulraiz/Documents/PHD_Data/code/')
sys.path.insert(0,'/home/gulraiz/Documents/PHD_Data/code/PIPNet/')
import numpy as np
from PIL import Image
import logging
import copy
import importlib
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from functions import * 
from mobilenetv3 import mobilenetv3_large
from Get_Backbone_Features import *
from networks import *


import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

import glob

i=0
for file in glob.glob("/home/gulraiz/Documents/PHD_Data/code/Data/part/cropped/*.jpg"):
    #print(file)
    image_f = cv2.imread(file)
    image_f=get_features(image_f)
    print(file)
    head,tail = os.path.split(file) 
    tail=os.path.splitext(tail)[0]
    # #print(head)
    #print(head+'/'+tail+'.npy')
    with open(head+'/'+tail+'.npy', 'wb') as f:
        np.save(f, image_f[0].data.numpy())
        
    # with open(head+'/'+tail+'.npy', 'rb') as f:

    a = np.load(head+'/'+tail+'.npy')
    print(a)
    print(i)
    i+=1
    
     
# class GenderDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, resize=True, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.resize = resize
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

#         image_f = cv2.imread(img_path)


#         label = self.img_labels.iloc[idx, 1]
#         image_f=get_features(image_f)
#             #image = transform(image)
        

#         return image_f, label, img_path


