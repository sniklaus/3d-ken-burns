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

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

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

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './autozoom.mp4'
boomerang_clip = True
zoom_duration = 3.0
frame_rate = 25

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
	if strOption == '--boomerang' and strArgument != '': boomerang_clip = int(strArgument) # 0 or 1 if 1 the clip will zoom in then out, if 0 it will just zoom in
	if strOption == '--duration' and strArgument != '': zoom_duration = float(strArgument) # duration of the zoom of the clip in seconds
	if strOption == '--fps' and strArgument != '': frame_rate = int(strArgument) # frame rate of the clip
# end

##########################################################

if __name__ == '__main__':
	npyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	zoom_steps = int(zoom_duration * frame_rate)
	intWidth = npyImage.shape[1]
	intHeight = npyImage.shape[0]

	fltRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(1024 * fltRatio), 1024)
	intHeight = min(int(1024 / fltRatio), 1024)

	npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

	process_load(npyImage, {})

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
		'fltSteps': numpy.linspace(0.0, 1.0, zoom_steps).tolist(),
		'objFrom': objFrom,
		'objTo': objTo,
		'boolInpaint': True
	})
	
	if boomerang_clip:
		frames = [ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:] ]
	else:
		frames = [ npyFrame[:, :, ::-1] for npyFrame in npyResult ]
	moviepy.editor.ImageSequenceClip(sequence = frames, fps=frame_rate).write_videofile(arguments_strOut)
# end