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
file_path = '/home/gulraiz/Documents/PHD_Data/code/Data/faces/fold_0_data.csv'

# Specify the column index to read
column_index = 2  # for example, to read the third column (indexing starts from 0)

# Read the CSV file into a pandas DataFrame with space as delimiter
data = pd.read_csv(file_path, delimiter='\t')

# Extract the specified column
Age_Vales = data.iloc[:, 3]
Gender_Vales = data.iloc[:, 4]
images_folder=data.iloc[:, 0]
images_name= data.iloc[:, 1]
faceids=data.iloc[:, 2]
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
    if Age_Vales[index] == 'None' or Gender_Vales[index] =='None' or (Gender_Vales[index]!='m' and Gender_Vales[index]!='f'):
        index+=1
        total_count-=1
        continue
    age=age_labels[Age_Vales[index]]
    gender=gender_labels[Gender_Vales[index]]
    faceid=faceids[index]
    image_path="/home/gulraiz/Documents/PHD_Data/code/Data/faces/"+images_folder[index] + "/coarse_tilt_aligned_face."+str(faceid)+"."+image

    print(gender)
    '/home/gulraiz/Documents/PHD_Data/code/Data/faces/30601258@N03/coarse_tilt_aligned_face.1.10424815813_e94629b1ec_o.jpg'
    frame = cv2.imread(image_path)
    rects = get_cropped(frame)
    index=index+1
    for r in rects:
        det_ymin, det_ymax, det_xmin, det_xmax = r
        cropped_frame = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
        frame_features, lms_pred_merge = get_features(cropped_frame, detect=False)

        frame_features = frame_features.unsqueeze(0)
        class_val_age, prob = inference(model, frame_features)
        class_val_gender, prob = inference(model2, frame_features)

        if class_val_age in age:
            age_count+=1
        if class_val_gender == gender:
            gender_count+=1
        break
print("Age:"+str(age_count))
print("Gender:" + str(gender_count))

print("Age ACC:"+str(age_count/total_count))
print("Gender ACC:" + str(gender_count/total_count))

# Print the values in the specified column
# Age:2447
# Gender:3608
# Age ACC:0.6125156445556946
# Gender ACC:0.9031289111389237
print(column_values)