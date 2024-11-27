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
class Gender_Age_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, resize=True, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.resize = resize
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        head,tail = os.path.split(img_path) 
        tail=os.path.splitext(tail)[0]
        image_f=np.load(head+'/'+tail+'.npy')
        image_f=torch.from_numpy(image_f)

        label = self.img_labels.iloc[idx, 1]
        #image_f=get_features(image_f)
        
        
        
        
        

        return image_f, label, img_path


def train(net,optimizer,criterion,trainloader,save_path,epochs=100,save_after=50):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, path = data
            inputs = inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.6f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), save_path)
    



def inference(model,image_f):
    
    output=model(image_f)
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(output)
    
    class_value=np.argmax(pred_probab.cpu().detach().numpy() )
    
    return class_value,output

def train_gender(save_after):
    training_data = Gender_Age_Dataset(
        "/home/gulraiz/Documents/PHD_Data/code/Data/part/gender_cropped.csv",
        "/home/gulraiz/Documents/PHD_Data/code/Data/part/cropped"
    )
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    
    net=GenderNet(resnet18, cfg.num_nb)
    if True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    train(net,optimizer,criterion,trainloader,"/home/gulraiz/Documents/PHD_Data/code/gender_model.pth",save_after=save_after)

def train_age_reg(epochs,save_after):
    training_data = Gender_Age_Dataset(
        "/home/ubuntu/gender/age_cropped.csv",
        "/home/ubuntu/gender/cropped"
    )
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    
    net=AgeNet(resnet18, cfg.num_nb)
    if True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)
    
    criterion = nn.L1Loss()  # mean square error
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    train(net,optimizer,criterion,trainloader,"/home/gulraiz/Documents/PHD_Data/code/age_model.pth",epochs,save_after)
    
def train_age_cls(epochs,save_after):
    training_data = Gender_Age_Dataset(
        "/home/gulraiz/Documents/PHD_Data/code/Data/part/age_cropped.csv",
        "/home/gulraiz/Documents/PHD_Data/code/Data/part/cropped"
    )
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    
    net=AgeNetCls(resnet18, cfg.num_nb)
    
    if True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    train(net,optimizer,criterion,trainloader,"/home/gulraiz/Documents/PHD_Data/code/age_model_cls.pth",save_after=save_after)
    





#train_age_cls(300,50)
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


# frame = cv2.imread("/home/ubuntu/test_36.jpg")
# frame_features=get_features(frame,detect=True)
# frame_features=frame_features.unsqueeze(0)
# class_val_gender,output=inference(model2,frame_features)
# print(class_val_gender)
# print(output)


#class_val_age=inference(model,frame_features)


train_gender(50)
from torchsummary import summary
from torchviz import make_dot

summary(resnet18_G, (3, 256, 256))

print(resnet18_G)


import numpy as np
import cv2

cap = cv2.VideoCapture("/home/gulraiz/Downloads/test2.mp4")
#cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
gender_class_label=["Male", "Female"]
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
j=0
while(cap.isOpened()):

    ret, frame = cap.read()
    j += 1
    if j%10==0:

        if ret==True:

            rects= get_cropped(frame)

            for r in rects:
                det_ymin,det_ymax, det_xmin,det_xmax=r
                cropped_frame= frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
                frame_features,lms_pred_merge= get_features(cropped_frame, detect=False)


                frame_features = frame_features.unsqueeze(0)
                class_val_age, prob = inference(model, frame_features)
                class_val_gender, prob = inference(model2, frame_features)

                for i in range(cfg.num_lms):
                    x_pred = lms_pred_merge[i * 2] * cropped_frame.shape[1]
                    y_pred = lms_pred_merge[i * 2 + 1] * cropped_frame.shape[0]
                    cv2.circle(cropped_frame, (int(x_pred), int(y_pred)), 1, (0, 0, 255), 2)

                cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (36, 255, 12), 1)
                cv2.putText(frame, "Age:" + str((class_val_age*10)+5), (det_xmin, det_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(frame, "Gender:" + str(gender_class_label[class_val_gender]), (det_xmin+150, det_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)




            # print("detected")
            if len(rects) == 0:
                print("no face detected")


            print(j)
            if(j>320 and j<2700):
                frame=frame[80:frame.shape[0],:]
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
        else:
            break

# Release everything if job is finished 2.2 1.1, 0.9, 0.8 , 0.75, 0.72, 0.54,0.55, 0.48, 0.45, 0.39, 0.31, 0.28, 0.26, 0.24, 0.23, 0.20, 0.21, 0.189, 0.186, 0.183, 0.126, 0.113, 0.129, 0.117, 0.112,
#0.110, 0.09, 0.08, 0.07, 0.05, 0.02, 0.06, 0.04, 0.03, 0.03, 0.04, 0.04, 0.02, 0.01, 0.03, 0.03, 0.01, 0.02, 0.03, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.008, 0.006, 0.005, 0.005, 0.004, 0.005, 0.007,

# 0.69, 0.27, 0.21, 0.14, 0.13, 0.10, 0.07, 0.03, 0.02, 0.01, 0.01, 0.01, 0.002, 0.002, 0.0009,
cap.release()
out.release()
cv2.destroyAllWindows()
