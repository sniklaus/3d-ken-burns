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

def process_dollyzoom(objSettings):
	npyOutputs = []

	if 'boolInpaint' not in objSettings or objSettings['boolInpaint'] == True:
		objCommon['tenInpaImage'] = objCommon['tenRawImage'].view(1, 3, -1)
		objCommon['tenInpaDisparity'] = objCommon['tenRawDisparity'].view(1, 1, -1)
		objCommon['tenInpaDepth'] = objCommon['tenRawDepth'].view(1, 1, -1)
		objCommon['tenInpaPoints'] = objCommon['tenRawPoints'].view(1, 3, -1)

		for fltStep in [ 1.0 ]:
			fltFrom = 1.0 - fltStep
			fltTo = 1.0 - fltFrom

			fltCropWidth = (fltFrom * objSettings['objFrom']['intCropWidth']) + (fltTo * objSettings['objTo']['intCropWidth'])
			fltCropHeight = (fltFrom * objSettings['objFrom']['intCropHeight']) + (fltTo * objSettings['objTo']['intCropHeight'])

			fltDepthFrom = objCommon['tenRawDepth'].min().item()
			fltDepthTo = fltDepthFrom * (objSettings['objFrom']['intCropWidth'] / fltCropWidth)

			tenShift = process_shift({
				'tenPoints': objCommon['tenInpaPoints'],
				'fltShiftU': 0.0,
				'fltShiftV': 0.0,
				'fltDepthFrom': fltDepthFrom,
				'fltDepthTo': fltDepthTo
			})[1]

			process_inpaint(tenShift)
		# end
	# end

	for fltStep in objSettings['fltSteps']:
		fltFrom = 1.0 - fltStep
		fltTo = 1.0 - fltFrom

		fltCropWidth = (fltFrom * objSettings['objFrom']['intCropWidth']) + (fltTo * objSettings['objTo']['intCropWidth'])
		fltCropHeight = (fltFrom * objSettings['objFrom']['intCropHeight']) + (fltTo * objSettings['objTo']['intCropHeight'])

		fltDepthFrom = objCommon['tenRawDepth'].min().item()
		fltDepthTo = fltDepthFrom * (objSettings['objFrom']['intCropWidth'] / fltCropWidth)

		tenPoints = process_shift({
			'tenPoints': objCommon['tenInpaPoints'],
			'fltShiftU': 0.0,
			'fltShiftV': 0.0,
			'fltDepthFrom': fltDepthFrom,
			'fltDepthTo': fltDepthTo
		})[0]

		tenRender, tenExisting = render_pointcloud(tenPoints, torch.cat([ objCommon['tenInpaImage'], objCommon['tenInpaDepth'] ], 1).view(1, 4, -1), objCommon['intWidth'], objCommon['intHeight'], objCommon['fltFocal'], objCommon['fltBaseline'])

		tenRender = fill_disocclusion(tenRender, tenRender[:, 3:4, :, :] * (tenExisting > 0.001).float())

		npyOutput = (tenRender[0, 0:3, :, :].numpy(force=True).transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).astype(numpy.uint8)
		npyOutput = cv2.getRectSubPix(image=npyOutput, patchSize=(int(round(fltCropWidth)), int(round(fltCropHeight))), center=(objCommon['intWidth'] / 2.0, objCommon['intHeight'] / 2.0))
		npyOutput = cv2.resize(src=npyOutput, dsize=(objCommon['intWidth'], objCommon['intHeight']), fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR)

		npyOutputs.append(npyOutput)
	# end

	return npyOutputs
# end

##########################################################

args_strIn = './images/doublestrike.jpg'
args_strOut = './dollyzoom.mp4'
args_strDepth = None
args_fltZoom = 1.2

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
	'in=',
	'out=',
	'depth=',
	'zoom=',
])[0]:
	if strOption == '--in' and strArg != '': args_strIn = strArg # path to the input image
	if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
	if strOption == '--depth' and strArg != '': args_strDepth = strArg # optional path to a depth map in numpy format
	if strOption == '--zoom' and strArg != '': args_fltZoom = float(strArg) # zoom factor for the dolly zoom effect
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

	objTo = {
		'fltCenterU': intWidth / 2.0,
		'fltCenterV': intHeight / 2.0,
		'intCropWidth': int(round(objFrom['intCropWidth'] / args_fltZoom)),
		'intCropHeight': int(round(objFrom['intCropHeight'] / args_fltZoom))
	}

	npyResult = process_dollyzoom({
		'fltSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
		'objFrom': objFrom,
		'objTo': objTo,
		'boolInpaint': True
	})

	moviepy.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:-1] ], fps=25).write_videofile(args_strOut)
# end
