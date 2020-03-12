import os
import sys
sys.path.insert(1, '/src/detector/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import util
import numpy as np
import cv2

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class DarkNet(nn.Module):
    def __init__(self, cfg):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfg)
        self.network_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        outputs = {} #outputs cached for route
        modules = self.blocks[1:] #skip first since net block
        first_detection = 0

        for i, module in enumerate(modules):
            module_type = (module['type'])

            if module_type == 'upsample' or module_type == 'convolutional':
                x = self.module_list[i](x)
            
            elif module_type == 'shortcut':
                origin = int(module['from'])
                x = outputs[i-1] + outputs[i+origin]

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] -i

                if len(layers) == 1:
                    x = outputs[i+(layers[0])]

                else: #2 feature maps
                    if (layers[1]) > 0:
                        layers[1] = layers[1] -i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1) #concat along depth(channel dim)

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.network_info['height']) #no input dims
                no_classes = int(module['classes'])
                x = x.data
                x = util.transform_pred(x, input_dim, no_classes, anchors, CUDA)        

                if first_detection == 0:
                    detections = x
                    first_detection = 1
                else:
                    detections = torch.cat((detections, x), 1)
            
            outputs[i] = x
        return detections
    
    def load_weights(self, weight_file):
        f = open(weight_file, 'rb')
        header = np.fromfile(f, dtype=np.int32, count=5) #first 5 vals are headers
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(f, dtype=np.float32)

        pointer = 0

        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]    
                
                if batch_normalize:
                    btc = model[1]
                    no_btc_biases = btc.bias.numel() #no weights of batch normalising layer                   
                    
                    btc_biases = torch.from_numpy(weights[pointer:pointer + no_btc_biases]) #load weights
                    pointer += no_btc_biases
                    btc_weights = torch.from_numpy(weights[pointer:pointer + no_btc_biases])
                    pointer += no_btc_biases
                    btc_mean = torch.from_numpy(weights[pointer:pointer + no_btc_biases]) 
                    pointer += no_btc_biases
                    btc_var = torch.from_numpy(weights[pointer:pointer + no_btc_biases]) 
                    pointer += no_btc_biases

                    btc_mean = btc_mean.view_as(btc.running_mean) #reshape weights -> dims of models weights
                    btc_biases = btc_biases.view_as(btc.bias.data)
                    btc_var = btc_var.view_as(btc.running_var)
                    btc_weights = btc_weights.view_as(btc.weight.data)
                    
                    btc.running_mean.data.copy_(btc_mean) #data -> model
                    btc.bias.data.copy_(btc_biases)
                    btc.running_var.data.copy_(btc_var)
                    btc.weight.data.copy_(btc_weights)

                else:
                    no_biases = conv.bias.numel() #no. biases
                    conv_biases = torch.from_numpy(weights[pointer:pointer + no_biases])
                    pointer += no_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                no_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[pointer:pointer + no_weights])
                pointer += no_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

def parse_cfg(cfgfile):

    file = open(cfgfile, 'r')
    lines = file.read().split('\n') # lines as list
    lines = [x for x in lines if len(x) > 0] # remove empty lines
    lines = [x for x in lines if x[0] != '#'] # remove comments
    lines = [x.rstrip().lstrip() for x in lines] # remove whitespace

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, val = line.split('=')
            block[key.rstrip()] = val.lstrip()
    blocks.append(block)
    return blocks

def create_modules(blocks): #create module per block 
    module_list = nn.ModuleList()
    network_info = blocks[0] #info on input/preprocessing
    output_filters = []
    prev_filters = 3

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential() #exec module objs sequentially

        if (x['type'] == 'convolutional'):
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            kernel_size = int(x['size'])
            filters = int(x['filters'])
            stride = int(x['stride'])
            padding = int(x['pad'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv) # add conv layer

            if batch_normalize: # add batch norm layer
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            if activation == 'leaky': #check activation func
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)

        elif (x['type'] == 'route'): # if route layer
            x['layers'] = x['layers'].split(',')

            start = int(x['layers'][0])
            try: #check for second layer
                end = int(x['layers'][1])
            except:
                end = 0
            
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)
            if end < 0: # if concating maps
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif (x['type'] == 'upsample'): #use bilinear2dup if upsamp layer
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{0}'.format(index), upsample)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{0}'.format(index), detection)
        
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (network_info, module_list)

# def get_test_input():
#     img = cv2.imread('src/detector/test.png')
#     img = cv2.resize(img, (608,608)) #resize to input dime
#     img = img[:,:,::-1].transpose((2,0,1)) #bgr to rgb, hwc to chw
#     img = img[np.newaxis,:,:,:]/255 #add channel at 0 for batch, normalise
#     img = torch.from_numpy(img).float()
#     img = Variable(img)    
#     return img

# blocks = parse_cfg('src/detector/cfg/yolov3.cfg')
# print(create_modules(blocks))

try:
    model = DarkNet('cfg/yolov3.cfg')
except FileNotFoundError:
    model = DarkNet('src/detector/cfg/yolov3.cfg')
try:
    model.load_weights('cfg/yolov3.weights')
except FileNotFoundError:
    model.load_weights('src/detector/cfg/yolov3.weights')
# input = get_test_input()
# pred = model(input, torch.cuda.is_available())
# print(pred)