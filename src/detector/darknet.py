from __future__ import division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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

        for i, module in enumerate(modules):
            module_type = (module['type'])

            if module_type == ('upsample' or 'convolutional'):
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

blocks = parse_cfg('src/detector/cfg/yolov3.cfg')
print(create_modules(blocks))