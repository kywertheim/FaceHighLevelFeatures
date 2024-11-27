import cv2, os
import sys
sys.path.insert(0,'/home/gulraiz/Documents/PHD_Data/code/PIPNet/lib/FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import *


experiment_name = "experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py".split('/')[-1][:-3]
data_name = "experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py".split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
print(config_path)
my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name
save_dir = os.path.join('/home/gulraiz/Documents/PHD_Data/code/PIPNet/snapshots', cfg.data_name, cfg.experiment_name)
meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('/home/gulraiz/Documents/PHD_Data/code/PIPNet/data', cfg.data_name, 'meanface.txt'), cfg.num_nb)
resnet18_A = models.resnet18(pretrained=cfg.pretrained)
resnet18_G = models.resnet18(pretrained=cfg.pretrained)
resnet18 = models.resnet18(pretrained=cfg.pretrained)
net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
net2=ResNet18BackBone(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
if True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("the device is")
print(device)
net = net.to(device)
net2 = net2.to(device)
weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
detector = FaceBoxesDetector('FaceBoxes', '/home/gulraiz/Documents/PHD_Data/code/PIPNet/lib/FaceBoxesV2/weights/FaceBoxesV2.pth', cfg.use_gpu, device)
my_thresh = 0.9
det_box_scale = 1.2

net.eval()
def get_cropped(frame):
    detections, _ = detector.detect(frame, my_thresh, 1)
    rects=[]
    for i in range (0,len(detections)):
        det_xmin = detections[i][2]
        det_ymin = detections[i][3]
        det_width = detections[i][4]
        det_height = detections[i][5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1
        det_xmin -= int(det_width * (det_box_scale - 1) / 2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (det_box_scale - 1) / 2)
        det_xmax += int(det_width * (det_box_scale - 1) / 2)
        det_ymax += int(det_height * (det_box_scale - 1) / 2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        # det_xmax = min(det_xmax, 256-1)
        # det_ymax = min(det_ymax, 256-1)
        # det_width = det_xmax - det_xmin + 1
        # det_height = det_ymax - det_ymin + 1

        rectangle=[det_ymin,det_ymax,det_xmin,det_xmax]
        rects.append(rectangle)

        det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
    return rects
def get_features(frame, net_stride=cfg.net_stride, num_nb=cfg.num_nb,detect=False):

    if detect:
        rects=get_cropped(frame)
        det_ymin,det_ymax, det_xmin,det_xmax=rects[0]
        frame= frame[det_ymin:det_ymax, det_xmin:det_xmax, :]

        #print("detected")
        if len(frame)==0:
        	return []
        cv2.imshow("face",frame)
        cv2.waitKey(1)
    

    #det_crop = cv2.resize(frame, (cfg.input_size, cfg.input_size))
    inputs = Image.fromarray(frame[:,:,::-1].astype('uint8'), 'RGB')
    inputs = preprocess(inputs).unsqueeze(0)
    inputs = inputs.to(device)
    x_intermediate_Features= forward_pip_intermediate(net2, inputs, preprocess, cfg.input_size, net_stride, num_nb)
    x_intermediate_Features=x_intermediate_Features.squeeze(0)
    lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess,
                                                                                             cfg.input_size, net_stride,
                                                                                             num_nb)
    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
    tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
    tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
    lms_pred = lms_pred.cpu().numpy()
    lms_pred_merge = lms_pred_merge.cpu().numpy()





    return x_intermediate_Features,lms_pred_merge



# import glob
# import os
# files=glob.glob("/home/gulraiz/Documents/PHD_Data/code/Data/part/part1/*")
# import csv
#
#
# for f in files:
#     frame=cv2.imread(f)
#     cropped=get_cropped(frame)
#     if len(cropped)>0:
#
#         head, tail = os.path.split(f)
#         print(tail)
#         cv2.imwrite("/home/gulraiz/Documents/PHD_Data/code/Data/part/cropped/"+tail,cropped)
#
#


# frame=cv2.imread('/home/gulraiz/Documents/PHD_Data/code/Data/part/part1/1_1_1_20170109194553059.jpg')
# # # # # #
# f=get_features(frame)
# print(f)
