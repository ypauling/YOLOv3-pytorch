from __future__ import division

import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from utils import *

import argparse
import os
import os.path as osp

from darknet import Darknet
import pickle
import pandas as pd
import random

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 video detector')

    parser.add_argument(
        '--video', dest='video',
        help='Video/Directory containing videos for detection',
        default='video', type=str
    )
    parser.add_argument(
        '--det', dest='det',
        help='Directory to put detections in',
        default='det', type=str
    )
    parser.add_argument('--bs', dest='bs', help='Batch size', default=1)
    parser.add_argument(
        '--confidence', dest='confidence',
        help='Object confidence to filter predictions',
        default=0.5
    )
    parser.add_argument(
        '--nms_threshold', dest='nms_threshold',
        help='NMS threshold',
        default=0.4
    )
    parser.add_argument(
        '--cfg', dest='cfgfile',
        help='Configuration file',
        default='cfg/yolov3.cfg', type=str
    )
    parser.add_argument(
        '--weights', dest='weightsfile',
        help='Weights file',
        default='yolov3.weights', type=str
    )
    parser.add_argument(
        '--reso', dest='resolution',
        help='Input resolution to the network',
        default=416, type=str
    )
    return parser.parse_args()

args = arg_parse()
videofile = args.video
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_threshold = float(args.nms_threshold)
cfgfile = args.cfgfile
weightsfile = args.weightsfile

CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes('data/coco.names')
colors = pickle.load(open('pallete', 'rb'))

def write_bbox(x, frame):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = '{0}'.format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(frame, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0]+t_size[0]+3, c1[1]+t_size[1]+4
    cv2.rectangle(frame, c1, c2, color, -1)
    cv2.putText(
        frame, label, (c1[0], c1[1]+t_size[1]+4),
        cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1
    )
    return frame

print('Loading network...')
model = Darknet(cfgfile)
model.load_weights(weightsfile)
print('Module loading finished...')

model.net_info['height'] = int(args.resolution)
inpdim = model.net_info['height']
assert inpdim % 32 == 0
assert inpdim > 32

if CUDA:
    model.cuda()
model.eval()

read_video = time.time()
cap = cv2.VideoCapture(videofile)
assert cap.isOpened(), 'Cannot capture source'

frames = 0
start = time.time()
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        img = prep_image(frame, inpdim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).unsqueeze(0)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        img.requires_grad = False
        output = model(img, CUDA)
        output = write_results(output, confidence, num_classes, nms_threshold)

        if type(output) == int:
            frames += 1
            print('FPS of the video is {:5.4f}'.format(
                frames/(time.time()-start)
            ))
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key & 0xff == ord('q'):
                break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inpdim/im_dim, 1)[0].view(-1,1)
        output[:,[1,3]] -= (inpdim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inpdim - scaling_factor*im_dim[:,1].view(-1,1))/2
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[:,[1,3]] = torch.clamp(output[:,[1,3]], 0.0, im_dim[i,0])
            output[:,[2,4]] = torch.clamp(output[:,[2,4]], 0.0, im_dim[i,1])

        list(map(lambda x: write_bbox(x, frame), output))
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break
        frames += 1
        print('FPS of the video is {:5.4f}'.format(frames/(time.time()-start)))

    else:
        break

cap.release()
cv2.destroyAllWindows()
