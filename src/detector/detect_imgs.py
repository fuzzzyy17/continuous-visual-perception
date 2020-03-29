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

    cmd.add_argument('--images', dest='imgs', default='../../data/imgs', type=str)
    cmd.add_argument('--det', dest='det', default='../../data/det', type=str)
    cmd.add_argument('--bs', dest='bs', default=1)
    cmd.add_argument('--nms', dest='nms', default=.4)
    cmd.add_argument('--conf', dest='conf', default=.5)
    cmd.add_argument('--cfg', dest='cfg', default='cfg/yolov3.cfg', type=str)
    #cmd.add_argument('--cfg', dest='cfg', default='src/detector/cfg/yolov3.cfg', type=str)
    cmd.add_argument('--reso', dest='reso', default="416", type=str)
    cmd.add_argument('--weights', dest='weights', default='cfg/yolov3.weights', type=str)
    #cmd.add_argument('--weights', dest='weights', default='src/detector/cfg/yolov3.weights', type=str)
    
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

no_classes = len(classes)

params = cmd_line()
imgs = params.imgs
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

det_loop_s = time.time()
output_init = False

for i, batch in enumerate(image_batches):
    load_start = time.time()
    if gpu:
        batch = batch.cuda()
    with torch.no_grad():
        pred = model(Variable(batch), gpu)
    pred = results(pred, no_classes, conf, nms)
    load_end = time.time()

    if type(pred) == int:
        for img_no, img in enumerate(image_paths[i*batch_size : min((i+1)*batch_size, len(image_paths))]):
            img_id = img_no + batch_size*i
            print("time taken to detect objects in: {:15s} : {:5.2f} s".format(img.split("/")[-1], (load_end - load_start)/batch_size))
            print("{:15s} {:s} \n".format("objects detected: ", " "))
        continue

    pred[:,0] = pred[:,0] + i*batch_size
    if not output_init:
        out = pred
        output_init = True
    else:
        out = torch.cat((out, pred))
    
    for img_no, img in enumerate(image_paths[batch_size*i : min((i+1)*batch_size, len(image_paths))]):
        img_id = img_no+ batch_size*img_no
        objects = [classes[int(x[-1])] for x in out if int(x[0]) == img_id]
        print("time taken to detect objects in: {:15s} : {:5.2f} s".format(img.split("/")[-1], (load_end - load_start)/batch_size))
        print("{:15s} {:s} \n".format("objects detected: ", " ".join(objects)))

    if gpu:
        torch.cuda.synchronize() #prevent async. calling

try:
    out
except NameError:
    print("no detections")
    exit()

image_dims = torch.index_select(image_dims, 0, out[:,0].long())

scale_factor = torch.min(input_dim/image_dims, 1)[0].view(-1, 1) #make coords of bbox conform to image on padded area
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
    print('colours not found')
    exit()

draw_time = time.time()

list(map(lambda x : draw_boxes(x, loaded_imgs), out))
det_names = pandas.Series(image_paths).apply(lambda x: "{}/detection_{}".format(params.det, x.split('/')[-1]))

list(map(cv2.imwrite, det_names, loaded_imgs))
end = time.time()

print("{:20s}: {:3.3f}".format("Address read time", load_s - read_s))
print("{:20s}: {:3.3f}".format("Batch loading time", det_loop_s - load_s))
print("{:40s}".format("Detected " + str(len(image_paths)) + " images"))
print("{:20s}: {:3.3f}".format("", end - draw_time))

torch.cuda.empty_cache()