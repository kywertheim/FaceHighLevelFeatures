from typing import Any

import pandas as pd
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
import numpy as np
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
# Specify the file path
def inference(model, image_f):
    output = model(image_f)
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(output)

    class_value = np.argmax(pred_probab.cpu().detach().numpy())

    return class_value, output
file_path = '/home/gulraiz/Documents/PHD_Data/code/Data/part/age_cropped.csv'
gender_path = '/home/gulraiz/Documents/PHD_Data/code/Data/part/gender_cropped.csv'

# Specify the column index to read
column_index = 2  # for example, to read the third column (indexing starts from 0)

# Read the CSV file into a pandas DataFrame with space as delimiter
data = pd.read_csv(file_path, delimiter=',')
datagender: Any = pd.read_csv(gender_path, delimiter=',')
# Extract the specified column
Age_Vales = data.iloc[:, 1]
Gender_Vales = datagender.iloc[:, 1]
images_name= data.iloc[:, 0]
index=0
print("device is the :")
print(device)
#Gender training and testing
model =AgeNetCls(resnet18_A, cfg.num_nb)
model.load_state_dict(torch.load("/home/gulraiz/Documents/PHD_Data/code/age_model_cls.pth",map_location=torch.device('cpu')))
model=model.to(device)
model.eval()


model2 =GenderNet(resnet18_G, cfg.num_nb)
model2.load_state_dict(torch.load("/home/gulraiz/Documents/PHD_Data/code/gender_modelFinal.pth",map_location=torch.device('cpu')))
model2=model2.to(device)
model2.eval()
gender_labels={"m":0,"f":1}



age_labels={"3":[0],"(0, 2)":[0],"(4, 6)":[0],"13":[1],"(8, 12)":[0,1],"(8, 13)":[0,1],"(15, 20)":[1],"22":[2],'35':[2,3],"(25, 32)":[2,3],"(38, 48)":[3,4],"36":[3],"(38, 43)":[3,4],"45":[4],"58":[4,5],"55":[4,5],"(48, 53)":[4,5],"(60, 100)":[6,7,8,9,10]}
age_count=0
gender_count=0
total_count=len(data)

for image in images_name:
    image=images_name[index]

    age=Age_Vales[index]
    gender=datagender[datagender["name"] == image]["gender"].values[0]


    image_path="/home/gulraiz/Documents/PHD_Data/code/Data/part/cropped/"+image
    cropped_frame = cv2.imread(image_path)


    frame_features, lms_pred_merge = get_features(cropped_frame, detect=False)

    frame_features = frame_features.unsqueeze(0)
    class_val_age, prob = inference(model, frame_features)
    class_val_gender, prob = inference(model2, frame_features)

    if class_val_age==age:
        age_count+=1
    if class_val_gender == gender:
        gender_count+=1
    else:
        print("Gender wrong")
    index = index + 1
    print(str(index)+' '+str(age_count/index)+" "+str(gender_count/index))

print("Age:"+str(age_count))
print("Gender:" + str(gender_count))

print("Age ACC:"+str(age_count/total_count))
print("Gender ACC:" + str(gender_count/total_count))

# Print the values in the specified column
# Age:2447
# Gender:3608
# Age ACC:0.6125156445556946
# Gender ACC:0.9031289111389237
