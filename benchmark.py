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

print('large parts of this benchmark were adapted from Tobias Koch')
print('this implementation first downloads the official evaluation scripts')
print('the depth boundary error is currently different from the paper')
print('this is due to the official evaluation scripts being outdated')

##########################################################

abs_rel = [ numpy.nan ] * 1000
sq_rel = [ numpy.nan ] * 1000
rms  = [ numpy.nan ] * 1000
log10 = [ numpy.nan ] * 1000
thr1 = [ numpy.nan ] * 1000
thr2 = [ numpy.nan ] * 1000
thr3 = [ numpy.nan ] * 1000
dde_0 = [ numpy.nan ] * 1000
dde_m = [ numpy.nan ] * 1000
dde_p = [ numpy.nan ] * 1000
dbe_acc = [ numpy.nan ] * 1000
dbe_com = [ numpy.nan ] * 1000
pe_fla = []
pe_ori = []

##########################################################

if os.path.isfile('./benchmark-ibims-scripts.zip') == False:
	urllib.request.urlretrieve('ftp://m1455541:m1455541@dataserv.ub.tum.de/evaluation_scripts.zip', './benchmark-ibims-scripts.zip')
# end

objectZip = zipfile.ZipFile('./benchmark-ibims-scripts.zip', 'r')

strScript = objectZip.read('evaluation_scripts/evaluate_ibims_error_metrics.py').decode('utf-8')
strScript = strScript.replace('# exclude masked invalid and missing measurements', 'idx = gt!=0')
strScript = strScript.replace('gt=gt[gt!=0]', 'gt=gt[idx]')
strScript = strScript.replace('pred=pred[pred!=0]', 'pred=pred[idx]')

exec(strScript)

objectZip.close()

##########################################################

if os.path.isfile('./benchmark-ibims-data.zip') == False:
	urllib.request.urlretrieve('ftp://m1455541:m1455541@dataserv.ub.tum.de/ibims1_core_mat.zip', './benchmark-ibims-data.zip')
# end

objectZip = zipfile.ZipFile('./benchmark-ibims-data.zip', 'r')

for intMat, strMat in enumerate([ strFile for strFile in objectZip.namelist() if strFile.endswith('.mat') ]):
	print(intMat, strMat)

	objectMat = scipy.io.loadmat(io.BytesIO(objectZip.read(strMat)))['data']

	tensorImage = torch.FloatTensor(numpy.ascontiguousarray(objectMat['rgb'][0][0][:, :, ::-1]).transpose(2, 0, 1)).unsqueeze(0).cuda() / 255.0
	tensorDisparity = disparity_estimation(tensorImage)
	tensorDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tensorImage, size=(tensorDisparity.shape[2] * 4, tensorDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tensorDisparity)
	tensorDisparity = torch.nn.functional.interpolate(input=tensorDisparity, size=(tensorImage.shape[2], tensorImage.shape[3]), mode='bilinear', align_corners=False) * (max(tensorImage.shape[2], tensorImage.shape[3]) / 256.0)
	tensorDepth = 1.0 / tensorDisparity

	valid = objectMat['mask_transp'][0][0] * objectMat['mask_invalid'][0][0] * (objectMat['depth'][0][0] != 0.0)

	pred = tensorDepth[0, 0, :, :].cpu().numpy()
	numpyLstsqa = numpy.stack([pred[valid == 1.0].flatten(), numpy.full([int((valid == 1.0).sum().item())], 1.0, numpy.float32)], 1)
	numpyLstsqb = objectMat['depth'][0][0][valid == 1.0].flatten()
	numpyScalebias = numpy.linalg.lstsq(numpyLstsqa, numpyLstsqb, None)[0]
	pred = (pred * numpyScalebias[0]) + numpyScalebias[1]

	abs_rel[intMat], sq_rel[intMat], rms[intMat], log10[intMat], thr1[intMat], thr2[intMat], thr3[intMat] = compute_global_errors((objectMat['depth'][0][0] * valid).flatten(), (pred * valid).flatten())
	dde_0[intMat], dde_m[intMat], dde_p[intMat] = compute_directed_depth_error((objectMat['depth'][0][0] * valid).flatten(), (pred * valid).flatten(), 3.0)
	dbe_acc[intMat], dbe_com[intMat] = compute_depth_boundary_error(objectMat['edges'][0][0], pred)

	if objectMat['mask_wall_paras'][0][0].size > 0:
		pe_fla_wall, pe_ori_wall = compute_planarity_error(objectMat['depth'][0][0] * valid, pred * valid, objectMat['mask_wall_paras'][0][0], objectMat['mask_wall'][0][0] * valid, objectMat['calib'][0][0])
		pe_fla.extend(pe_fla_wall.tolist())
		pe_ori.extend(pe_ori_wall.tolist())
	# end

	if objectMat['mask_table_paras'][0][0].size > 0:
		pe_fla_table, pe_ori_table = compute_planarity_error(objectMat['depth'][0][0] * valid, pred * valid, objectMat['mask_table_paras'][0][0], objectMat['mask_table'][0][0] * valid, objectMat['calib'][0][0])
		pe_fla.extend(pe_fla_table.tolist())
		pe_ori.extend(pe_ori_table.tolist())
	# end

	if objectMat['mask_floor_paras'][0][0].size > 0:
		pe_fla_floor, pe_ori_floor = compute_planarity_error(objectMat['depth'][0][0] * valid, pred * valid, objectMat['mask_floor_paras'][0][0], objectMat['mask_floor'][0][0] * valid, objectMat['calib'][0][0])
		pe_fla.extend(pe_fla_floor.tolist())
		pe_ori.extend(pe_ori_floor.tolist())
	# end
# end

objectZip.close()

##########################################################

print('abs_rel = ', numpy.nanmean(abs_rel))
print('sq_rel  = ', numpy.nanmean(sq_rel))
print('rms     = ', numpy.nanmean(rms))
print('log10   = ', numpy.nanmean(log10))
print('thr1    = ', numpy.nanmean(thr1))
print('thr2    = ', numpy.nanmean(thr2))
print('thr3    = ', numpy.nanmean(thr3))
print('dde_0   = ', numpy.nanmean(dde_0))
print('dde_m   = ', numpy.nanmean(dde_m))
print('dde_p   = ', numpy.nanmean(dde_p))
print('dbe_acc = ', numpy.nanmean(dbe_acc))
print('dbe_com = ', numpy.nanmean(dbe_com))
print('pe_fla  = ', numpy.nanmean(pe_fla))
print('pe_ori  = ', numpy.nanmean(pe_ori))