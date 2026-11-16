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

print('this script first downloads the labeled nyu depth data, which may take a while')
print('it performs the evaluation using the train / test split from Nathan Silberman')
print('it crops the input by 16 pixels, like https://github.com/princeton-vl/relative_depth')
print('differs from the paper, see https://github.com/sniklaus/3d-ken-burns/issues/34')

##########################################################

fltAbsrel = []
fltLogten = []
fltSqrel = []
fltRmse = []
fltThr1 = []
fltThr2 = []
fltThr3 = []

##########################################################

torch.hub.download_url_to_file('http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat', './benchmark-nyu-splits.mat')

intTests = [ intTest - 1 for intTest in scipy.io.loadmat('./benchmark-nyu-splits.mat')['testNdxs'].flatten().tolist() ]

assert(len(intTests) == 654)

##########################################################

torch.hub.download_url_to_file('http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat', './benchmark-nyu-data.mat')

objData = h5py.File('./benchmark-nyu-data.mat', 'r')

npyImages = numpy.array(objData['images'], numpy.uint8)
npyDepths = numpy.array(objData['depths'], numpy.float32)[:, None, :, :]

objData.close()

assert(npyImages.shape[0] == 1449 and npyImages.shape[2] == 640 and npyImages.shape[3] == 480)
assert(npyDepths.shape[0] == 1449 and npyDepths.shape[2] == 640 and npyDepths.shape[3] == 480)

for intTest in intTests:
	print(intTest)

	npyImage = npyImages[intTest, :, 16:-16, 16:-16].transpose(2, 1, 0)[:, :, ::-1]
	npyReference = npyDepths[intTest, :, 16:-16, 16:-16].transpose(2, 1, 0)[:, :, 0]

	tenImage = torch.FloatTensor(numpy.ascontiguousarray(npyImage.copy().transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
	tenDisparity = disparity_estimation(tenImage)
	tenDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tenImage, size=(tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tenDisparity)
	tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) * (max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
	tenDepth = 1.0 / tenDisparity

	npyEstimate = tenDepth[0, 0, :, :].numpy(force=True)
	npyLstsqa = numpy.stack([npyEstimate.flatten(), numpy.full(npyEstimate.flatten().shape, 1.0, numpy.float32)], 1)
	npyLstsqb = npyReference.flatten()
	npyScalebias = numpy.linalg.lstsq(npyLstsqa, npyLstsqb, None)[0]
	npyEstimate = (npyEstimate * npyScalebias[0]) + npyScalebias[1]

	fltAbsrel.append(((npyEstimate - npyReference).__abs__() / npyReference).mean().item())
	fltLogten.append((numpy.log10(npyEstimate) - numpy.log10(npyReference)).__abs__().mean().item())
	fltSqrel.append(((npyEstimate - npyReference).__pow__(2.0) / npyReference).mean().item())
	fltRmse.append((npyEstimate - npyReference).__pow__(2.0).mean().__pow__(0.5).item())
	fltThr1.append((numpy.maximum((npyEstimate / npyReference), (npyReference / npyEstimate)) < 1.25 ** 1).mean().item())
	fltThr2.append((numpy.maximum((npyEstimate / npyReference), (npyReference / npyEstimate)) < 1.25 ** 2).mean().item())
	fltThr3.append((numpy.maximum((npyEstimate / npyReference), (npyReference / npyEstimate)) < 1.25 ** 3).mean().item())
# end

##########################################################

print('abs_rel = ', sum(fltAbsrel) / len(fltAbsrel))
print('log10   = ', sum(fltLogten) / len(fltLogten))
print('sq_rel  = ', sum(fltSqrel) / len(fltSqrel))
print('rms     = ', sum(fltRmse) / len(fltRmse))
print('thr1    = ', sum(fltThr1) / len(fltThr1))
print('thr2    = ', sum(fltThr2) / len(fltThr2))
print('thr3    = ', sum(fltThr3) / len(fltThr3))