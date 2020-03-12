import torch
import torch.nn as nn
import cv2
import os
from darknet import DarkNet
from util import *
import numpy as np
import argparse
import time
from torch.autograd import Variable
import pickle
import pandas
import random

def cmd_line():
    cmd = argparse.ArgumentParser(description="YOLO Detector")

    cmd.add_argument('--video', dest='video', default='data/vid', type=str)
    cmd.add_argument('--bs', dest='bs', default=1)
    cmd.add_argument('--nms', dest='nms', default=.4)
    cmd.add_argument('--conf', dest='conf', default=.5)
    cmd.add_argument('--cfg', dest='cfg', default='src/detector/cfg/yolov3.cfg', type=str) #default='cfg/yolov3.cfg', type=str)
    cmd.add_argument('--reso', dest='reso', default="416", type=str)
    cmd.add_argument('--weights', dest='weights', default='src/detector/cfg/yolov3.weights', type=str) #default='cfg/yolov3.weights', type=str)
    
    return cmd.parse_args()

def draw_boxes(x, results):

    br_corner = tuple(x[3:5].int())
    tl_corner = tuple(x[1:3].int())
    cl = int(x[-1])
    image = results[int(x[0])]
    label = '{}'.format(classes[cl])
    colour = random.choice(box_colours)
    cv2.rectangle(image, tl_corner, br_corner, colour, 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0]
    br_corner = text_size[0] + tl_corner[0] + 3, text_size[1] + tl_corner[1] + 4
    cv2.rectangle(image, tl_corner, br_corner, colour, -1)
    cv2.putText(image, label, (tl_corner[0], text_size[1] + tl_corner[1] + 4), cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 1)
    
    return image

try:
    classes = load_classes('../../data/coco.names')
except FileNotFoundError:
    classes = load_classes('data/coco.names')
except:
    print("classes not found")
    exit() #add error handling in draw box func

no_classes = len(classes)

params = cmd_line()
nms = float(params.nms)
batch_size = int(params.bs)
conf = float(params.conf)
gpu = torch.cuda.is_available()

model = DarkNet(params.cfg) #setup
model.load_weights(params.weights)
print("network loaded")
if gpu:
    model.cuda()

model.network_info['height'] = params.reso 
input_dim = int(model.network_info['height'])
if ((input_dim % 32 != 0) or (input_dim <= 32)):
    raise Exception('invalid image size')

model.eval() #eval mode
read_s = time.time() #need to note time

load_s = time.time()

det_loop_s = time.time()
output_init = False

scale_factor = torch.min(416/image_dims, 1)[0].view(-1, 1) #make coords of bbox conform to image on padded area
out[:,[1,3]] -= (input_dim - scale_factor * image_dims[:,0].view(-1, 1))/2
out[:,[2,4]] -= (input_dim - scale_factor * image_dims[:,1].view(-1, 1))/2
out[:,1:5] /= scale_factor #undo resize image scaling

for coord in range (out.shape[0]):
    out[coord, [1,3]] = torch.clamp(out[coord, [1,3]], 0, image_dims[coord,0])
    out[coord, [2,4]] = torch.clamp(out[coord, [2,4]], 0, image_dims[coord,1])

class_load = time.time()

try:
    box_colours = pickle.load(open('colours.p', 'rb'))
except FileNotFoundError:
    box_colours = pickle.load(open('src/detector/colours.p', 'rb'))
except:
    print("colours not found")

draw_time = time.time()

list(map(lambda x : draw_boxes(x, loaded_imgs), out))

list(map(cv2.imwrite, det_names, loaded_imgs))
end = time.time()