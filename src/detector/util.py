from __future__ import division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np

#detection feat map --> 2D tensor
#rows of tensor corr. to attrs of bounding box
def transform_pred(pred, input_dim, no_classes, anchors, CUDA):
    
    stride = input_dim // pred.size(2)#
    no_anchors = len(anchors)
    bounding_box_atrs = no_classes + 5
    batch_size = pred.size(0)
    grid_size = input_dim // stride

    pred = pred.view(batch_size, bounding_box_atrs * no_anchors, grid_size**2)
    pred = pred.transpose(1,2).contiguous()
    pred = pred.view(batch_size, ((grid_size**2) * no_anchors), bounding_box_atrs)

    # for i in anchors: #anchors dims currently describe input img attrs from net block
    #     anchors.append((i[0]/stride, i[1]/stride)) #larger than detection map by factor of stride therefore divide

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    pred[:,:,0] = torch.sigmoid(pred[:,:,0]) #sigmoid centre x
    pred[:,:,1] = torch.sigmoid(pred[:,:,1]) #centre y
    pred[:,:,4] = torch.sigmoid(pred[:,:,4]) #obj score

    try:
        pred = pred.cuda()
    except:
        pass

    grid = np.arange(grid_size)
    grid_x, grid_y = np.meshgrid(grid, grid) #centre offsets
    x_offset = torch.FloatTensor(grid_x).view(-1,1)
    y_offset = torch.FloatTensor(grid_y).view(-1,1)

    try: #if CUDA == True:
        y_offset = y_offset.cuda()
        x_offset = x_offset.cuda()
    except:
        pass
    
    xy_offset = torch.cat((x_offset, y_offset), 1).repeat(1, no_anchors).view(-1,2).unsqueeze(0)
    pred[:,:,:2] = pred[:,:,:2] + xy_offset

    anchors = torch.FloatTensor(anchors) #apply anchors to dims of bounding box
    try:
        anchors = anchors.cuda()
    except:
        pass

    anchors = anchors.repeat(grid_size**2, 1).unsqueeze(0)
    pred[:,:,2:4] = torch.exp(pred[:,:,2:4])*anchors
    
    pred[:,:,5: 5 + no_classes] = torch.sigmoid((pred[:,:,5: 5 + no_classes])) #class scores
    pred[:,:,:4] *= stride #resize detection map to size of input img
    
    return pred