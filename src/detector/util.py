from __future__ import division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np

def load_classes(class_file):
    f = open(class_file, 'r')
    classes = f.read().split("\n")[:-1]
    return classes

def convert_image_to_input(image, input_dim):
    image = cv2.resize(image, (input_dim, input_dim))
    image = image[:,:,::-1].transpose((2,0,1)).copy()
    image = torch.from_numpy(image).float().div(255).unsqueeze(0)
    return image

def resize_image(image, input_dim): #resize to keep aspect ratio consistent
    image_h, image_w = image.shape[0], image.shape[1] 
    input_h, input_w = input_dim
    new_h = int(image_h * min(input_w/image_w, input_h/image_h))
    new_w = int(image_w * min(input_w/image_w, input_h/image_h))
    new_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    padded = np.full((input_dim[1], input_dim[0], 3), 128) #padding
    padded[(input_h - new_h)//2 : (input_h-new_h)//2 + new_h, (input_w - new_w)//2 + new_w,:] = padded
    return padded

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

            for iou in range(index): #get ious of boxes that exist in next loop
                try:
                    ious = calc_bound_box_iou(img_class_prediction[iou].unsqueeze(0), img_class_prediction[iou + 1:])
                    # second input: tensor w/ multiple rows of bounding boxes
                except (ValueError, IndexError): #bboxes removed as loop
                    break

                #elim detects w/ IOU > thresh
                iou_test = (ious<nms_thresh).float().unsqueeze(1)
                img_class_prediction[i+1:] = img_class_prediction[i+1:] * iou_test
                non_0_entries = torch.nonzero(img_class_prediction[:, 4]).squeeze()
                img_class_prediction = img_class_prediction[non_0_entries].view(-1,7)
            
            batch_idx = img_class_prediction.new(img_class_prediction.size(0), 1).fill_(i) # repeat for as many detects for cls in img
            seq = batch_idx, img_class_prediction

            if not out_init:
                list_out = torch.cat(seq, 1)
                out_init = True
            else:
                out = (seq, 1)
                list_out = torch.cat((list_out, out))

    try:
        return list_out
    else:
        return 0

def calc_bound_box_iou(box1, box2): #box1: bbox row; box2: tensor w/ multiple rows of bounding boxes

    b1x1, b1y1, b1x2, b1y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2x1, b2y1, b2x2, b2y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    intersection_x1 = torch.max(b1x1, b2x1)
    intersection_y1 = torch.max(b1y1, b2y1)
    intersection_x2 = torch.max(b1x2, b2x2)
    intersection_y2 = torch.max(b1y2, b2y2)

    intersection = torch.clamp(intersection_x2 - intersection_x1 + 1, min = 0) * torch.clamp(intersection_y2, intersection_y1 + 1, min = 0)
    b1_area = (b1x2 - b1x1 + 1) * (b1y2 - b1y1 + 1)
    b2_area = (b2x2 - b2x1 + 1) * (b2y2 - b2y1 + 1)

    iou = intersection / (b1_area + b2_area - intersection)
    return iou #tensor w/ ious of box1 bounding box concat. w/ ea. bbox from box2