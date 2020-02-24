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

def get_img_classes(tensor):
    numpy_tensor = tensor.cpu().numpy()
    unique_numpy = np.unique(numpy_tensor)
    unique_tensor = torch.from_numpy(unique_numpy)
    tensor_ = tensor.new(unique_tensor.shape)
    tensor_.copy_(unique_tensor)
    return tensor_

def results(pred, no_classes, conf, nms_thresh=.4):

    pre_conf = (pred[:,:,4] > conf).float().unsqueeze(2) #check if bbox obj score above thresh
    pred = pred * pre_conf #if below, =0

    corner_box_coords = pred.new(pred.shape) #move coords from middle to top left corner
    corner_box_coords[:,:,3] = (pred[:,:,1] - pred[:,:,3]/2)
    corner_box_coords[:,:,2] = (pred[:,:,0] - pred[:,:,2]/2)
    corner_box_coords[:,:,1] = (pred[:,:,1] - pred[:,:,3]/2)
    corner_box_coords[:,:,0] = (pred[:,:,0] - pred[:,:,2]/2)
    
    pred[:,:,:4] = corner_box_coords[:,:,:4]

    batch_size = pred.size(0)
    out_init = False #output not init. 

    for i in range(batch_size): #loop over 1st dim of pred since nms/conf done 1 img at a time
        image_pred = pred[i]
        max_conf, max_conf_score = torch.max(image_pred[:,5:5 + no_classes], 1) #get index of 5 highest score/val classes
        max_conf_score = max_conf_score.float().unsqueeze(1)
        max_conf = max_conf.float().unsqueeze(1)
        top = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(top, 1)
        
        thresh_indxs = (torch.nonzero(image_pred[:,4]))#remove boxes w/ obj conf < thresh
        if image_pred.shape[0] == 0:
            continue
        try:
            image_pred = image_pred[thresh_indxs.squeeze(),:].view(-1,7)
        except:
            continue

        img_classes = get_img_classes(image_pred[:,-1])
        for class_ in img_classes: #non-max supprs.
            class_detections = image_pred*(image_pred[:,-1]==class_).float().unsqueeze(1) #find objs of spec. class 
            class_detections_indxs = torch.nonzero(class_detections[:,-2]).squeeze()
            img_class_prediction = image_pred[class_detections_indxs].view(-1,7)

            sort_conf = torch.sort(img_class_prediction[:,4], descending=True)[1] #sort by max obj conf
            img_class_prediction = img_class_prediction[sort_conf]
            index = img_class_prediction.size(0) #no. detections
