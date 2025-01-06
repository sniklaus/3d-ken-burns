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

torch.hub.download_url_to_file('ftp://m1455541:m1455541@dataserv.ub.tum.de/evaluation_scripts.zip', './benchmark-ibims-scripts.zip')

with zipfile.ZipFile('./benchmark-ibims-scripts.zip', 'r') as objZip:
	strScript = objZip.read('evaluation_scripts/evaluate_ibims_error_metrics.py').decode('utf-8')
	strScript = strScript.replace('# exclude masked invalid and missing measurements', 'idx = gt!=0')
	strScript = strScript.replace('gt=gt[gt!=0]', 'gt=gt[idx]')
	strScript = strScript.replace('pred=pred[pred!=0]', 'pred=pred[idx]')

	exec(strScript)
# end

##########################################################

torch.hub.download_url_to_file('ftp://m1455541:m1455541@dataserv.ub.tum.de/ibims1_core_mat.zip', './benchmark-ibims-data.zip')

with zipfile.ZipFile('./benchmark-ibims-data.zip', 'r') as objZip:
	for intMat, strMat in enumerate([ strFile for strFile in objZip.namelist() if strFile.endswith('.mat') ]):
		print(intMat, strMat)

		objMat = scipy.io.loadmat(io.BytesIO(objZip.read(strMat)))['data']

		tenImage = torch.FloatTensor(numpy.ascontiguousarray(objMat['rgb'][0][0][:, :, ::-1].transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
		tenDisparity = disparity_estimation(tenImage)
		tenDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tenImage, size=(tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tenDisparity)
		tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) * (max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
		tenDepth = 1.0 / tenDisparity

		valid = objMat['mask_transp'][0][0] * objMat['mask_invalid'][0][0] * (objMat['depth'][0][0] != 0.0)

		pred = tenDepth[0, 0, :, :].numpy(force=True)
		npyLstsqa = numpy.stack([pred[valid == 1.0].flatten(), numpy.full([int((valid == 1.0).sum().item())], 1.0, numpy.float32)], 1)
		npyLstsqb = objMat['depth'][0][0][valid == 1.0].flatten()
		npyScalebias = numpy.linalg.lstsq(npyLstsqa, npyLstsqb, None)[0]
		pred = (pred * npyScalebias[0]) + npyScalebias[1]

		abs_rel[intMat], sq_rel[intMat], rms[intMat], log10[intMat], thr1[intMat], thr2[intMat], thr3[intMat] = compute_global_errors((objMat['depth'][0][0] * valid).flatten(), (pred * valid).flatten())
		dde_0[intMat], dde_m[intMat], dde_p[intMat] = compute_directed_depth_error((objMat['depth'][0][0] * valid).flatten(), (pred * valid).flatten(), 3.0)
		dbe_acc[intMat], dbe_com[intMat] = compute_depth_boundary_error(objMat['edges'][0][0], pred)

		if objMat['mask_wall_paras'][0][0].size > 0:
			pe_fla_wall, pe_ori_wall = compute_planarity_error(objMat['depth'][0][0] * valid, pred * valid, objMat['mask_wall_paras'][0][0], objMat['mask_wall'][0][0] * valid, objMat['calib'][0][0])
			pe_fla += pe_fla_wall.tolist()
			pe_ori += pe_ori_wall.tolist()
		# end

		if objMat['mask_table_paras'][0][0].size > 0:
			pe_fla_table, pe_ori_table = compute_planarity_error(objMat['depth'][0][0] * valid, pred * valid, objMat['mask_table_paras'][0][0], objMat['mask_table'][0][0] * valid, objMat['calib'][0][0])
			pe_fla += pe_fla_table.tolist()
			pe_ori += pe_ori_table.tolist()
		# end

		if objMat['mask_floor_paras'][0][0].size > 0:
			pe_fla_floor, pe_ori_floor = compute_planarity_error(objMat['depth'][0][0] * valid, pred * valid, objMat['mask_floor_paras'][0][0], objMat['mask_floor'][0][0] * valid, objMat['calib'][0][0])
			pe_fla += pe_fla_floor.tolist()
			pe_ori += pe_ori_floor.tolist()
		# end
	# end
# end

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