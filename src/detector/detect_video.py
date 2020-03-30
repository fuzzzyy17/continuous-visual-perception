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

    cmd.add_argument('--video', dest='video', default='C:/Users/faiza/downloads/fruit-and-vegetable-detection.mp4', type=str) #default='data/vid'
    cmd.add_argument('--bs', dest='bs', default=1)
    cmd.add_argument('--nms', dest='nms', default=.4)
    cmd.add_argument('--conf', dest='conf', default=.5)
    #cmd.add_argument('--cfg', dest='cfg', default='src/detector/cfg/yolov3.cfg', type=str)
    cmd.add_argument('--cfg', dest='cfg', default='cfg/yolov3.cfg', type=str)
    cmd.add_argument('--reso', dest='reso', default="416", type=str)
    #cmd.add_argument('--weights', dest='weights', default='src/detector/cfg/yolov3.weights', type=str) 
    cmd.add_argument('--weights', dest='weights', default='cfg/yolov3.weights', type=str)
    
    return cmd.parse_args()

def draw_boxes(x, results):

    br_corner = tuple(x[3:5].int())
    tl_corner = tuple(x[1:3].int())
    cl = int(x[-1])
    image = results
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

# try:
#     videofile = '../../data/vid'
#     stream = cv2.VideoCapture(videofile)
# except FileNotFoundError:
#     videofile = 'data/vid'
#     stream = cv2.VideoCapture(videofile)
# except:
#     print("video not found, using webcam")
#     stream = cv2.VideoCapture(0)

videofile = 'C:/Users/faiza/downloads/fruit-and-vegetable-detection.mp4'
stream = cv2.VideoCapture(videofile)

assert stream.isOpened(), 'no video input available'

frames = 0

try:
    box_colours = pickle.load(open('colours.p', 'rb'))
except FileNotFoundError:
    box_colours = pickle.load(open('src/detector/colours.p', 'rb'))
except:
    print('colours not found')
    exit()

start = 0
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
start = time.time() 

while stream.isOpened():
    valid, frame = stream.read()

    if valid != False:
        image = convert_image_to_input(frame, input_dim)
        #cv2.imshow("a", frame)
        img_dim = frame.shape[1], frame.shape[0]
        img_dim = torch.FloatTensor(img_dim).repeat(1,2)

        try:
            image = image.cuda()
            img_dim = img_dim.cuda()
        except:
            continue

        with torch.no_grad():
            out = model(Variable(image), gpu)
        out = results(out, no_classes, conf, nms)

        if type(out) == int:
            frames +=1 
            print("vid FPS: {:4.3f}".format(frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            term = cv2.waitKey(1)
            if term & 0xFF == ord('q'):
                break
            continue

        img_dim = img_dim.repeat(out.size(0), 1)
        scaling = torch.min(416/img_dim, 1)[0].view(-1, 1)
        out[:,[1,3]] -= (input_dim - scaling * img_dim[:,0].view(-1, 1))/2
        out[:,[2,4]] -= (input_dim - scaling * img_dim[:,1].view(-1, 1))/2
        out[:,1:5] /= scaling

        for c in range(out.shape[0]):
            out[c, [1,3]] = torch.clamp(out[c, [1,3]], 0, img_dim[c,0])
            out[c, [2,4]] = torch.clamp(out[c, [2,4]], 0, img_dim[c,1])
        
        list(map(lambda x : draw_boxes(x, frame), out))

        cv2.imshow("frame", frame)
        term = cv2.waitKey(1)
        if term & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS: {:3.1f}".format(frames / (time.time() - start)))
    
    else:
        break