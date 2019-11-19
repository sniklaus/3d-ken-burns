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

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 120) # requires at least pytorch version 1.2.0

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

objectPlayback = {
	'strImage': None,
	'numpyImage': None,
	'strMode': 'automatic',
	'intTime': 0,
	'dblTime': numpy.linspace(0.0, 1.0, 75).tolist() + list(reversed(numpy.linspace(0.0, 1.0, 75).tolist())),
	'strCache': {},
	'objectFrom': {
		'dblCenterU': 512.0,
		'dblCenterV': 384.0,
		'intCropWidth': 1024,
		'intCropHeight': 768
	},
	'objectTo': {
		'dblCenterU': 512.0,
		'dblCenterV': 384.0,
		'intCropWidth': 1024,
		'intCropHeight': 768
	}
}

objectFlask = flask.Flask(import_name=__name__, static_url_path='', static_folder=os.path.abspath('./'))

@objectFlask.route(rule='/', methods=[ 'GET' ])
def index():
	return objectFlask.send_static_file('interface.html')
# end

@objectFlask.route(rule='/load_image', methods=[ 'POST' ])
def load_image():
	objectPlayback['strImage'] = flask.request.form['strFile']
	objectPlayback['numpyImage'] = numpy.ascontiguousarray(cv2.imdecode(buf=numpy.fromstring(base64.b64decode(flask.request.form['strData'].split(';base64,')[1]), numpy.uint8), flags=-1)[:, :, 0:3])
	objectPlayback['strCache'] = {}

	process_load(objectPlayback['numpyImage'], {})

	for dblX, dblY in [ (100.0, 0.0), (-100.0, 0.0), (0.0, 100.0), (0.0, -100.0) ]:
		process_inpaint(torch.FloatTensor([ dblX, dblY, 0.0 ]).view(1, 3, 1).cuda())
	# end

	return ''
# end

@objectFlask.route(rule='/autozoom', methods=[ 'POST' ])
def autozoom():
	objectPlayback['objectFrom'] = {
		'dblCenterU': 512.0,
		'dblCenterV': 384.0,
		'intCropWidth': 1000,
		'intCropHeight': 750
	}

	objectPlayback['objectTo'] = process_autozoom({
		'dblShift': 100.0,
		'dblZoom': 1.25,
		'objectFrom': objectPlayback['objectFrom']
	})

	return flask.jsonify({
		'objectFrom': objectPlayback['objectFrom'],
		'objectTo': objectPlayback['objectTo']
	})
# end

@objectFlask.route(rule='/update_mode', methods=[ 'POST' ])
def update_mode():
	objectPlayback['strMode'] = flask.request.form['strMode']

	return ''
# end

@objectFlask.route(rule='/update_from', methods=[ 'POST' ])
def update_from():
	objectPlayback['intTime'] = objectPlayback['dblTime'].index(0.0)
	objectPlayback['strCache'] = {}
	objectPlayback['objectFrom']['dblCenterU'] = float(flask.request.form['dblCenterU'])
	objectPlayback['objectFrom']['dblCenterV'] = float(flask.request.form['dblCenterV'])
	objectPlayback['objectFrom']['intCropWidth'] = int(flask.request.form['intCropWidth'])
	objectPlayback['objectFrom']['intCropHeight'] = int(flask.request.form['intCropHeight'])

	return ''
# end

@objectFlask.route(rule='/update_to', methods=[ 'POST' ])
def update_to():
	objectPlayback['intTime'] = objectPlayback['dblTime'].index(1.0)
	objectPlayback['strCache'] = {}
	objectPlayback['objectTo']['dblCenterU'] = float(flask.request.form['dblCenterU'])
	objectPlayback['objectTo']['dblCenterV'] = float(flask.request.form['dblCenterV'])
	objectPlayback['objectTo']['intCropWidth'] = int(flask.request.form['intCropWidth'])
	objectPlayback['objectTo']['intCropHeight'] = int(flask.request.form['intCropHeight'])

	return ''
# end

@objectFlask.route(rule='/get_live', methods=[ 'GET' ])
def get_live():
	def generator():
		dblFramelimiter = 0.0

		while True:
			for intYield in range(100): gevent.sleep(0.0)

			gevent.sleep(max(0.0, (1.0 / 25.0) - (time.time() - dblFramelimiter))); dblFramelimiter = time.time()

			if objectPlayback['strImage'] is None:
				yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode(ext='.jpg', img=numpy.ones([ 768, 1024, 3 ], numpy.uint8) * 29, params=[ cv2.IMWRITE_JPEG_QUALITY, 80 ])[1].tobytes() + b'\r\n'; continue
			# end

			if objectPlayback['intTime'] > len(objectPlayback['dblTime']) - 1:
				objectPlayback['intTime'] = 0
			# end

			intTime = objectPlayback['intTime']
			dblTime = objectPlayback['dblTime'][intTime]

			if objectPlayback['strMode'] == 'automatic':
				objectPlayback['intTime'] += 1
			# end

			if str(dblTime) not in objectPlayback['strCache']:
				numpyKenburns = process_kenburns({
					'dblSteps': [ dblTime ],
					'objectFrom': objectPlayback['objectFrom'],
					'objectTo': objectPlayback['objectTo'],
					'boolInpaint': False
				})[0]

				objectPlayback['strCache'][str(dblTime)] = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode(ext='.jpg', img=numpyKenburns, params=[ cv2.IMWRITE_JPEG_QUALITY, 80 ])[1].tobytes() + b'\r\n'
			# end

			yield objectPlayback['strCache'][str(dblTime)]
		# end
	# end

	return flask.Response(response=generator(), mimetype='multipart/x-mixed-replace; boundary=frame')
# end

@objectFlask.route(rule='/get_result', methods=[ 'GET' ])
def get_result():
	strTempdir = tempfile.gettempdir() + '/kenburns-' + str(os.getpid()) + '-' + str.join('', [ random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for intCount in range(20) ])

	os.makedirs(strTempdir + '/')

	numpyKenburns = process_kenburns({
		'dblSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
		'objectFrom': objectPlayback['objectFrom'],
		'objectTo': objectPlayback['objectTo'],
		'boolInpaint': True
	})

	moviepy.editor.ImageSequenceClip(sequence=[ numpyFrame[:, :, ::-1] for numpyFrame in numpyKenburns + list(reversed(numpyKenburns))[1:] ], fps=25).write_videofile(strTempdir + '/kenburns.mp4')

	objectKenburns = io.BytesIO(open(strTempdir + '/kenburns.mp4', 'rb').read())

	shutil.rmtree(strTempdir + '/')

	return flask.send_file(filename_or_fp=objectKenburns, mimetype='video/mp4', as_attachment=True, attachment_filename='kenburns.mp4', cache_timeout=-1)
# end

gevent.pywsgi.WSGIServer(listener=('0.0.0.0', 8080), application=objectFlask).serve_forever()