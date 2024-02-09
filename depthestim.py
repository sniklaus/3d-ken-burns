#!/usr/bin/env python

import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import torch
import torchvision
import urllib
import zipfile

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################

args_strIn = './images/doublestrike.jpg'
args_strOut = './depthestim.npy'

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
	'in=',
	'out=',
])[0]:
	if strOption == '--in' and strArg != '': args_strIn = strArg # path to the input image
	if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
# end

##########################################################

if __name__ == '__main__':
	npyImage = cv2.imread(filename=args_strIn, flags=cv2.IMREAD_COLOR)

	fltFocal = max(npyImage.shape[1], npyImage.shape[0]) / 2.0
	fltBaseline = 40.0

	tenImage = torch.FloatTensor(numpy.ascontiguousarray(npyImage.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
	tenDisparity = disparity_estimation(tenImage)
	tenDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tenImage, size=(tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tenDisparity)
	tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) * (max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
	tenDepth = (fltFocal * fltBaseline) / (tenDisparity + 0.0000001)

	npyDisparity = tenDisparity[0, 0, :, :].cpu().numpy()
	npyDepth = tenDepth[0, 0, :, :].cpu().numpy()

	cv2.imwrite(filename=args_strOut.replace('.npy', '.png'), img=(npyDisparity / fltBaseline * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

	numpy.save(args_strOut, npyDepth)
# end