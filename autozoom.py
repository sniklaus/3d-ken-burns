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
args_strOut = './autozoom.mp4'
args_strDepth = None
args_fltDuration = 3.0
args_fltFps = 25.0
args_boolBoomerang = True

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
	'in=',
	'out=',
	'depth=',
	'duration=',
	'fps=',
	'boomerang=',
])[0]:
	if strOption == '--in' and strArg != '': args_strIn = strArg # path to the input image
	if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
	if strOption == '--depth' and strArg != '': args_strDepth = strArg # optional path to a depth map in numpy format
	if strOption == '--duration' and strArgument != '': args_fltDuration = float(strArgument) # duration the output in seconds
	if strOption == '--fps' and strArgument != '': args_fltFps = float(strArgument) # frame rate of the output
	if strOption == '--boomerang' and strArgument != '': args_boolBoomerang = bool(strArgument) # boomerang or just zoom
# end

##########################################################

if __name__ == '__main__':
	npyImage = cv2.imread(filename=args_strIn, flags=cv2.IMREAD_COLOR)

	intWidth = npyImage.shape[1]
	intHeight = npyImage.shape[0]

	fltRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(1024 * fltRatio), 1024)
	intHeight = min(int(1024 / fltRatio), 1024)

	npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

	process_load(npyImage, {} if args_strDepth is None else {'npyDepth': numpy.load(args_strDepth)})

	objFrom = {
		'fltCenterU': intWidth / 2.0,
		'fltCenterV': intHeight / 2.0,
		'intCropWidth': int(math.floor(0.97 * intWidth)),
		'intCropHeight': int(math.floor(0.97 * intHeight))
	}

	objTo = process_autozoom({
		'fltShift': 100.0,
		'fltZoom': 1.25,
		'objFrom': objFrom
	})

	npyResult = process_kenburns({
		'fltSteps': numpy.linspace(0.0, 1.0, round(args_fltDuration * args_fltFps)).tolist(),
		'objFrom': objFrom,
		'objTo': objTo,
		'boolInpaint': True
	})
	
	npyResult = [ npyFrame[:, :, ::-1] for npyFrame in npyResult ]

	if args_boolBoomerang == True:
		npyResult = npyResult + list(reversed(npyResult))[1:]
	# end

	moviepy.ImageSequenceClip(sequence=npyResult, fps=args_fltFps).write_videofile(args_strOut)
# end