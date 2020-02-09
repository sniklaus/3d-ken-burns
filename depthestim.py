#!/usr/bin/env python

import torch
import torchvision

import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
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
import urllib
import zipfile

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objectCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './depthestim.npy'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

if __name__ == '__main__':
	numpyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	dblFocal = max(numpyImage.shape[0], numpyImage.shape[1]) / 2.0
	dblBaseline = 40.0

	tensorImage = torch.FloatTensor(numpyImage.transpose(2, 0, 1)).unsqueeze(0).cuda() / 255.0
	tensorDisparity = disparity_estimation(tensorImage)
	tensorDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tensorImage, size=(tensorDisparity.size(2) * 4, tensorDisparity.size(3) * 4), mode='bilinear', align_corners=False), tensorDisparity)
	tensorDisparity = torch.nn.functional.interpolate(input=tensorDisparity, size=(tensorImage.size(2), tensorImage.size(3)), mode='bilinear', align_corners=False) * (max(tensorImage.size(2), tensorImage.size(3)) / 256.0)
	tensorDepth = (dblFocal * dblBaseline) / (tensorDisparity + 0.0000001)

	numpyDisparity = tensorDisparity[0, 0, :, :].cpu().numpy()
	numpyDepth = tensorDepth[0, 0, :, :].cpu().numpy()

	cv2.imwrite(filename=arguments_strOut.replace('.npy', '.png'), img=(numpyDisparity / dblBaseline * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

	numpy.save(arguments_strOut, numpyDepth)
# end