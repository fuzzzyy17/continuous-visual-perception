import torch
import torch.nn as nn
import cv2
import os
from darknet import Darknet
from util import *
import numpy as np
import argparse
import time

no_classes = 80

def cmd_line():
    cmd = argparse.ArgumentParser(description="YOLO Detector")

    cmd.add_argument('--images', dest='imgs', default='imgs', type=str)
    cmd.add_argument('--det', dest='det', default='det', type=str)
    cmd.add_argument('--bs', dest='bs', default=1)
    cmd.add_argument('--nms', dest='nms', default=.4)
    cmd.add_argument('--conf', dest='conf', default=.5)
    cmd.add_argument('--cfg', dest='cfg', default='src/detector/cfg/yolov3.cfg', type=str)
    cmd.add_argument('--reso', dest='reso', default="416", type=str)
    cmd.add_argument('--weights', dest='weights', default='src/detector/cfg/yolov3.weights', type=str)
    
    return cmd.parse_args()

classes = load_classes('data/coco.names')

params = cmd_line()
imgs = params.images
nms = float(params.nms)
batch_size = int(params.bs)
conf = float(params.conf)
gpu = torch.cuda.is_available()

model = Darknet(params.cfg) #setup
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

try: #get image locations in list
    image_paths = [os.path.join(os.path.realpath('.'), imgs, img) for img in os.listdir(imgs)]
except NotADirectoryError:
    image_paths = []
    image_paths.append(os.path.join(os.path.realpath('.'), imgs))
except FileNotFoundError:
    print('no file/dir with name: ', imgs)
    exit()

if not os.path.exists(params.det): #create if doesn't exist
    os.makedirs(params.det)

load_s = time.time()
loaded_imgs = [cv2.imread(im) for im in image_paths] #load as numpy arr
# keep list of original image dims
image_batches = list(map(convert_image_to_input, loaded_imgs, [input_dim for im in range(len(image_paths))])) #pytorch vars for imgs
image_dims = [(im.shape[1], im.shape[0]) for im in loaded_imgs]
image_dims = torch.FloatTensor(image_dims).repeat(1,2)
if gpu:
    image_dims = image_dims.cuda()

left = 0
if (len(image_dims) % batch_size):
    left = 1
if batch_size != 1:
    no_batches = len(image_paths)// batch_size + left
    image_batches = [torch.cat((image_batches[i*batch_size: min((i+1)*batch_size, len(image_batches))])) for i in range(no_batches)]
    