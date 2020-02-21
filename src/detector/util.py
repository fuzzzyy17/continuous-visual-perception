from __future__ import division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#detection feat map --> 2D tensor
def transform_pred(pred, input_dim, no_classes, anchors, CUDA):
    
    stride = input_dim // pred.size(2)#
    no_anchors = len(anchors)
    bounding_box_atrs = no_classes + 5
    batch_size = pred.size(0)
    grid_size = input_dim // stride

    pred = pred.view(batch_size, bounding_box_atrs * no_anchors, grid_size**2)
    pred = pred.transpose(1,2).contiguous()
    pred = pred.view(batch_size, ((grid_size**2) * no_anchors), bounding_box_atrs)