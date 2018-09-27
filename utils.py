from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(preds, inpdim, anchors, num_classes, CUDA=True):
    batch_size = preds.size(0)
    stride = inpdim // preds.size(2)
    grid_size = preds.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    preds = preds.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    preds = preds.transpose(1, 2).contiguous()
    preds = preds.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    ### anchors original sizes conform to the original dim of the input
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    ### sigmoid transformation for center(x, y), and prob(object)
    preds[:,:,0] = torch.sigmoid(preds[:,:,0])
    preds[:,:,1] = torch.sigmoid(preds[:,:,1])
    preds[:,:,4] = torch.sigmoid(preds[:,:,4])

    ### add grid coodinates
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).\
                 repeat(1, num_anchors).\
                 view(-1, 2).\
                 unsqueeze(0)

    preds[:,:,:2] += x_y_offset

    ### add log space transformations of anchors
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    preds[:,:,2:4] = torch.exp(preds[:,:,2:4])*anchors

    ### sigmoid transformation to class scores
    preds[:,:,5:5+num_classes] = torch.sigmoid(preds[:,:,5:5+num_classes])

    ### resize the feature map
    preds[:,:,:4] *= stride

    return preds

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    b1x1, b1y1, b1x2, b1y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2x1, b2y1, b2x2, b2y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_rect_x1 = torch.max(b1x1, b2x1)
    inter_rect_y1 = torch.max(b1y1, b2y1)
    inter_rect_x2 = torch.min(b1x2, b2x2)
    inter_rect_y2 = torch.min(b1y2, b2y2)

    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1, min=0)*\
                 torch.clamp(inter_rect_y2-inter_rect_y1+1, min=0)

    b1_area = (b1x2-b1x1+1)*(b1y2-b1y1+1)
    b2_area = (b2x2-b2x1+1)*(b2y2-b2y1+1)

    iou = inter_area / (b1_area+b2_area-inter_area)

    return iou

def write_results(pred, confidence, num_classes, nms_conf=0.4):
    ### Object confidence thresholding
    conf_mask = (pred[:,:,4] > confidence).float().unsqueeze(2)
    pred = pred * conf_mask

    ### non-maximum suppression
    ### First convert (x, y, w, h) to (xlu, ylu, xrl, yrl),
    ### which are the four corner coordinates relative to
    ### the center coordinate
    box_corners = pred.new(pred.shape)
    box_corners[:,:,0] = (pred[:,:,0] - pred[:,:,2]/2)
    box_corners[:,:,1] = (pred[:,:,1] - pred[:,:,3]/2)
    box_corners[:,:,2] = (pred[:,:,0] + pred[:,:,2]/2)
    box_corners[:,:,3] = (pred[:,:,1] + pred[:,:,3]/2)
    pred[:,:,:4] = box_corners[:,:,:4]

    batch_size = pred.size(0)
    write = False
    for index in range(batch_size):
        image_pred = pred[index]

        ### convert class scores to predicted class by maximization
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        ### Remove zero rows from the confidence thresholding step
        non_zero_index = torch.nonzero(image_pred[:,4])
        try:
            image_pred_ = image_pred[non_zero_index.squeeze(),:].view(-1, 7)
        except:
            continue
        if image_pred_.shape[0] == 0:
            continue
        img_classes = unique(image_pred_[:,-1])

        ### Perform NMS
        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1]==cls).float().unsqueeze(1)
            cls_mask_index = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[cls_mask_index].view(-1, 7)

            conf_sort_index = torch.sort(
                image_pred_class[:,4], descending=True
            )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            nbr_bboxes = image_pred_class.size(0)

            ### Try-Catch block:
            ### nbr_bboxes is changing in the loop
            ### ValueError: empty row
            ### IndexError: no more rows
            for i in range(nbr_bboxes):
                try:
                    ious = bbox_iou(
                        image_pred_class[i].unsqueeze(0),
                        image_pred_class[i+1:]
                    )
                except ValueError:
                    break
                except IndexError:
                    break

                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask
                non_zero_index = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_index].view(-1, 7)

            ### Processing the result
            batch_index = image_pred_class.new(
                image_pred_class.size(0),1
            ).fill_(index)
            seq = (batch_index, image_pred_class)

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0

def load_classes(namesfile):
    fp = open(namesfile, 'r')
    names = fp.read().split('\n')[:-1]
    return names

def letterbox_img(img, inpdim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inpdim
    new_w = int(img_w*min(w/img_w, h/img_h))
    new_h = int(img_h*min(w/img_w, h/img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )

    canvas = np.full((inpdim[1], inpdim[0], 3), 128)
    canvas[((h-new_h)//2):((h-new_h)//2+new_h), \
           ((w-new_w)//2):((w-new_w)//2+new_w),:] = resized_image
    return canvas

def prep_image(img, inpdim):
    img = letterbox_img(img, (inpdim, inpdim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
