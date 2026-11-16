#!/usr/bin/env python

import cupy
import cv2
import math
import moviepy
import numpy
import os
import re
import tempfile
import torch
import torchvision

from cog import BasePredictor, Input, Path

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

class Predictor(BasePredictor):
    def setup(self):
        pass
    # end

    def run(self, image: Path = Input(description='input image')) -> Path:
        npyImage = cv2.imread(filename=str(image), flags=cv2.IMREAD_COLOR)

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
            'intCropHeight': int(math.floor(0.97 * intHeight)),
        }

        objTo = process_autozoom({
            'fltShift': 100.0,
            'fltZoom': 1.25,
            'objFrom': objFrom
        })

        npyResult = process_kenburns({
            'fltSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
            'objFrom': objFrom,
            'objTo': objTo,
            'boolInpaint': True
        })

        objOutput = Path(tempfile.mkdtemp()) / 'output.mp4'

        moviepy.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:-1] ], fps=25).write_videofile(str(objOutput))

        return objOutput
    # end
# end
