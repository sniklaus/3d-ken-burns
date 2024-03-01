def process_load(npyImage, objSettings):
	objCommon['fltFocal'] = 1024 / 2.0
	objCommon['fltBaseline'] = 40.0
	objCommon['intWidth'] = npyImage.shape[1]
	objCommon['intHeight'] = npyImage.shape[0]

	tenImage = torch.FloatTensor(numpy.ascontiguousarray(npyImage.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()

	if 'npyDepth' not in objSettings:
		tenDisparity = disparity_estimation(tenImage)
		tenDisparity = disparity_adjustment(tenImage, tenDisparity)
		tenDisparity = disparity_refinement(tenImage, tenDisparity)
		tenDisparity = tenDisparity / tenDisparity.max() * objCommon['fltBaseline']
		tenDepth = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (tenDisparity + 0.0000001)

	elif 'npyDepth' in objSettings:
		tenDepth = torch.FloatTensor(numpy.ascontiguousarray(numpy.atleast_3d(objSettings['npyDepth']).astype(numpy.float32).transpose(2, 0, 1)[None, :, :, :])).cuda()
		tenDisparity = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (tenDepth + 0.0000001)

	# end

	tenValid = (spatial_filter(tenDisparity / tenDisparity.max(), 'laplacian').abs() < 0.03).float()
	tenPoints = depth_to_points(tenDepth * tenValid, objCommon['fltFocal'])
	tenUnaltered = depth_to_points(tenDepth, objCommon['fltFocal'])

	objCommon['fltDispmin'] = tenDisparity.min().item()
	objCommon['fltDispmax'] = tenDisparity.max().item()
	objCommon['objDepthrange'] = cv2.minMaxLoc(src=tenDepth[0, 0, 128:-128, 128:-128].detach().cpu().numpy(), mask=None)
	objCommon['tenRawImage'] = tenImage
	objCommon['tenRawDisparity'] = tenDisparity
	objCommon['tenRawDepth'] = tenDepth
	objCommon['tenRawPoints'] = tenPoints.view(1, 3, -1)
	objCommon['tenRawUnaltered'] = tenUnaltered.view(1, 3, -1)

	objCommon['tenInpaImage'] = objCommon['tenRawImage'].view(1, 3, -1)
	objCommon['tenInpaDisparity'] = objCommon['tenRawDisparity'].view(1, 1, -1)
	objCommon['tenInpaDepth'] = objCommon['tenRawDepth'].view(1, 1, -1)
	objCommon['tenInpaPoints'] = objCommon['tenRawPoints'].view(1, 3, -1)
# end

def process_inpaint(tenShift):
	objInpainted = pointcloud_inpainting(objCommon['tenRawImage'], objCommon['tenRawDisparity'], tenShift)

	objInpainted['tenDepth'] = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (objInpainted['tenDisparity'] + 0.0000001)
	objInpainted['tenValid'] = (spatial_filter(objInpainted['tenDisparity'] / objInpainted['tenDisparity'].max(), 'laplacian').abs() < 0.03).float()
	objInpainted['tenPoints'] = depth_to_points(objInpainted['tenDepth'] * objInpainted['tenValid'], objCommon['fltFocal'])
	objInpainted['tenPoints'] = objInpainted['tenPoints'].view(1, 3, -1)
	objInpainted['tenPoints'] = objInpainted['tenPoints'] - tenShift

	tenMask = (objInpainted['tenExisting'] == 0.0).view(1, 1, -1)

	objCommon['tenInpaImage'] = torch.cat([ objCommon['tenInpaImage'], objInpainted['tenImage'].view(1, 3, -1)[tenMask.repeat(1, 3, 1)].view(1, 3, -1) ], 2)
	objCommon['tenInpaDisparity'] = torch.cat([ objCommon['tenInpaDisparity'], objInpainted['tenDisparity'].view(1, 1, -1)[tenMask.repeat(1, 1, 1)].view(1, 1, -1) ], 2)
	objCommon['tenInpaDepth'] = torch.cat([ objCommon['tenInpaDepth'], objInpainted['tenDepth'].view(1, 1, -1)[tenMask.repeat(1, 1, 1)].view(1, 1, -1) ], 2)
	objCommon['tenInpaPoints'] = torch.cat([ objCommon['tenInpaPoints'], objInpainted['tenPoints'].view(1, 3, -1)[tenMask.repeat(1, 3, 1)].view(1, 3, -1) ], 2)
# end

def process_shift(objSettings):
	fltClosestDepth = objCommon['objDepthrange'][0] + (objSettings['fltDepthTo'] - objSettings['fltDepthFrom'])
	fltClosestFromU = objCommon['objDepthrange'][2][0]
	fltClosestFromV = objCommon['objDepthrange'][2][1]
	fltClosestToU = fltClosestFromU + objSettings['fltShiftU']
	fltClosestToV = fltClosestFromV + objSettings['fltShiftV']
	fltClosestFromX = ((fltClosestFromU - (objCommon['intWidth'] / 2.0)) * fltClosestDepth) / objCommon['fltFocal']
	fltClosestFromY = ((fltClosestFromV - (objCommon['intHeight'] / 2.0)) * fltClosestDepth) / objCommon['fltFocal']
	fltClosestToX = ((fltClosestToU - (objCommon['intWidth'] / 2.0)) * fltClosestDepth) / objCommon['fltFocal']
	fltClosestToY = ((fltClosestToV - (objCommon['intHeight'] / 2.0)) * fltClosestDepth) / objCommon['fltFocal']

	fltShiftX = fltClosestFromX - fltClosestToX
	fltShiftY = fltClosestFromY - fltClosestToY
	fltShiftZ = objSettings['fltDepthTo'] - objSettings['fltDepthFrom']

	tenShift = torch.tensor(data=[[[fltShiftX], [fltShiftY], [fltShiftZ]]], dtype=torch.float32, device=torch.device('cuda'))

	tenPoints = objSettings['tenPoints'].clone()

	tenPoints[:, 0:1, :] *= tenPoints[:, 2:3, :] / (objSettings['tenPoints'][:, 2:3, :] + 0.0000001)
	tenPoints[:, 1:2, :] *= tenPoints[:, 2:3, :] / (objSettings['tenPoints'][:, 2:3, :] + 0.0000001)

	tenPoints += tenShift

	return tenPoints, tenShift
# end

def process_autozoom(objSettings):
	npyShiftU = numpy.linspace(-objSettings['fltShift'], objSettings['fltShift'], 16)[None, :].repeat(16, 0)
	npyShiftV = numpy.linspace(-objSettings['fltShift'], objSettings['fltShift'], 16)[:, None].repeat(16, 1)
	fltCropWidth = objSettings['objFrom']['intCropWidth'] / objSettings['fltZoom']
	fltCropHeight = objSettings['objFrom']['intCropHeight'] / objSettings['fltZoom']

	fltDepthFrom = objCommon['objDepthrange'][0]
	fltDepthTo = objCommon['objDepthrange'][0] * (fltCropWidth / objSettings['objFrom']['intCropWidth'])

	fltBest = 0.0
	fltBestU = None
	fltBestV = None

	for intU in range(16):
		for intV in range(16):
			fltShiftU = npyShiftU[intU, intV].item()
			fltShiftV = npyShiftV[intU, intV].item()

			if objSettings['objFrom']['fltCenterU'] + fltShiftU < fltCropWidth / 2.0:
				continue

			elif objSettings['objFrom']['fltCenterU'] + fltShiftU > objCommon['intWidth'] - (fltCropWidth / 2.0):
				continue

			elif objSettings['objFrom']['fltCenterV'] + fltShiftV < fltCropHeight / 2.0:
				continue

			elif objSettings['objFrom']['fltCenterV'] + fltShiftV > objCommon['intHeight'] - (fltCropHeight / 2.0):
				continue

			# end

			tenPoints = process_shift({
				'tenPoints': objCommon['tenRawPoints'],
				'fltShiftU': fltShiftU,
				'fltShiftV': fltShiftV,
				'fltDepthFrom': fltDepthFrom,
				'fltDepthTo': fltDepthTo
			})[0]

			tenRender, tenExisting = render_pointcloud(tenPoints, objCommon['tenRawImage'].view(1, 3, -1), objCommon['intWidth'], objCommon['intHeight'], objCommon['fltFocal'], objCommon['fltBaseline'])

			if fltBest < (tenExisting > 0.0).float().sum().item():
				fltBest = (tenExisting > 0.0).float().sum().item()
				fltBestU = fltShiftU
				fltBestV = fltShiftV
			# end
		# end
	# end

	return {
		'fltCenterU': objSettings['objFrom']['fltCenterU'] + fltBestU,
		'fltCenterV': objSettings['objFrom']['fltCenterV'] + fltBestV,
		'intCropWidth': int(round(objSettings['objFrom']['intCropWidth'] / objSettings['fltZoom'])),
		'intCropHeight': int(round(objSettings['objFrom']['intCropHeight'] / objSettings['fltZoom']))
	}
# end

def process_kenburns(objSettings):
	npyOutputs = []

	if 'boolInpaint' not in objSettings or objSettings['boolInpaint'] == True:
		objCommon['tenInpaImage'] = objCommon['tenRawImage'].view(1, 3, -1)
		objCommon['tenInpaDisparity'] = objCommon['tenRawDisparity'].view(1, 1, -1)
		objCommon['tenInpaDepth'] = objCommon['tenRawDepth'].view(1, 1, -1)
		objCommon['tenInpaPoints'] = objCommon['tenRawPoints'].view(1, 3, -1)

		for fltStep in [ 1.0 ]:
			fltFrom = 1.0 - fltStep
			fltTo = 1.0 - fltFrom

			fltShiftU = ((fltFrom * objSettings['objFrom']['fltCenterU']) + (fltTo * objSettings['objTo']['fltCenterU'])) - (objCommon['intWidth'] / 2.0)
			fltShiftV = ((fltFrom * objSettings['objFrom']['fltCenterV']) + (fltTo * objSettings['objTo']['fltCenterV'])) - (objCommon['intHeight'] / 2.0)
			fltCropWidth = (fltFrom * objSettings['objFrom']['intCropWidth']) + (fltTo * objSettings['objTo']['intCropWidth'])
			fltCropHeight = (fltFrom * objSettings['objFrom']['intCropHeight']) + (fltTo * objSettings['objTo']['intCropHeight'])

			fltDepthFrom = objCommon['objDepthrange'][0]
			fltDepthTo = objCommon['objDepthrange'][0] * (fltCropWidth / max(objSettings['objFrom']['intCropWidth'], objSettings['objTo']['intCropWidth']))

			tenShift = process_shift({
				'tenPoints': objCommon['tenInpaPoints'],
				'fltShiftU': fltShiftU,
				'fltShiftV': fltShiftV,
				'fltDepthFrom': fltDepthFrom,
				'fltDepthTo': fltDepthTo
			})[1]

			process_inpaint(1.1 * tenShift)
		# end
	# end

	for fltStep in objSettings['fltSteps']:
		fltFrom = 1.0 - fltStep
		fltTo = 1.0 - fltFrom

		fltShiftU = ((fltFrom * objSettings['objFrom']['fltCenterU']) + (fltTo * objSettings['objTo']['fltCenterU'])) - (objCommon['intWidth'] / 2.0)
		fltShiftV = ((fltFrom * objSettings['objFrom']['fltCenterV']) + (fltTo * objSettings['objTo']['fltCenterV'])) - (objCommon['intHeight'] / 2.0)
		fltCropWidth = (fltFrom * objSettings['objFrom']['intCropWidth']) + (fltTo * objSettings['objTo']['intCropWidth'])
		fltCropHeight = (fltFrom * objSettings['objFrom']['intCropHeight']) + (fltTo * objSettings['objTo']['intCropHeight'])

		fltDepthFrom = objCommon['objDepthrange'][0]
		fltDepthTo = objCommon['objDepthrange'][0] * (fltCropWidth / max(objSettings['objFrom']['intCropWidth'], objSettings['objTo']['intCropWidth']))

		tenPoints = process_shift({
			'tenPoints': objCommon['tenInpaPoints'],
			'fltShiftU': fltShiftU,
			'fltShiftV': fltShiftV,
			'fltDepthFrom': fltDepthFrom,
			'fltDepthTo': fltDepthTo
		})[0]

		tenRender, tenExisting = render_pointcloud(tenPoints, torch.cat([ objCommon['tenInpaImage'], objCommon['tenInpaDepth'] ], 1).view(1, 4, -1), objCommon['intWidth'], objCommon['intHeight'], objCommon['fltFocal'], objCommon['fltBaseline'])

		tenRender = fill_disocclusion(tenRender, tenRender[:, 3:4, :, :] * (tenExisting > 0.0).float())

		npyOutput = (tenRender[0, 0:3, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).astype(numpy.uint8)
		npyOutput = cv2.getRectSubPix(image=npyOutput, patchSize=(max(objSettings['objFrom']['intCropWidth'], objSettings['objTo']['intCropWidth']), max(objSettings['objFrom']['intCropHeight'], objSettings['objTo']['intCropHeight'])), center=(objCommon['intWidth'] / 2.0, objCommon['intHeight'] / 2.0))
		npyOutput = cv2.resize(src=npyOutput, dsize=(objCommon['intWidth'], objCommon['intHeight']), fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR)

		npyOutputs.append(npyOutput)
	# end

	return npyOutputs
# end

##########################################################

def preprocess_kernel(strKernel, objVariables):
	with open('./common.cuda', 'r') as objFile:
		strKernel = objFile.read() + strKernel
	# end

	for strVariable in objVariables:
		objValue = objVariables[strVariable]

		if type(objValue) == int:
			strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

		elif type(objValue) == float:
			strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

		elif type(objValue) == str:
			strKernel = strKernel.replace('{{' + strVariable + '}}', objValue)

		# end
	# end

	while True:
		objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objMatch is None:
			break
		# end

		intArg = int(objMatch.group(2))

		strTensor = objMatch.group(4)
		intSizes = objVariables[strTensor].size()

		strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(intSizes[intArg]) == False else intSizes[intArg].item()))
	# end

	while True:
		objMatch = re.search('(STRIDE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objMatch is None:
			break
		# end

		intArg = int(objMatch.group(2))

		strTensor = objMatch.group(4)
		intStrides = objVariables[strTensor].stride()

		strKernel = strKernel.replace(objMatch.group(), str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()))
	# end

	while True:
		objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objMatch is None:
			break
		# end

		intArgs = int(objMatch.group(2))
		strArgs = objMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
	# end

	while True:
		objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objMatch is None:
			break
		# end

		intArgs = int(objMatch.group(2))
		strArgs = objMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy.memoize(for_each_device=True)
def launch_kernel(strFunction, strKernel):
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = cupy.cuda.get_cuda_path()
    # end

    return cupy.RawKernel(strKernel, strFunction, tuple(['-I ' + os.environ['CUDA_HOME'], '-I ' + os.environ['CUDA_HOME'] + '/include']))
# end

def depth_to_points(tenDepth, fltFocal):
	tenHorizontal = torch.linspace(start=(-0.5 * tenDepth.shape[3]) + 0.5, end=(0.5 * tenDepth.shape[3]) - 0.5, steps=tenDepth.shape[3], dtype=tenDepth.dtype, device=tenDepth.device).view(1, 1, 1, -1).repeat(tenDepth.shape[0], 1, tenDepth.shape[2], 1)
	tenHorizontal = tenHorizontal * (1.0 / fltFocal)

	tenVertical = torch.linspace(start=(-0.5 * tenDepth.shape[2]) + 0.5, end=(0.5 * tenDepth.shape[2]) - 0.5, steps=tenDepth.shape[2], dtype=tenDepth.dtype, device=tenDepth.device).view(1, 1, -1, 1).repeat(tenDepth.shape[0], 1, 1, tenDepth.shape[3])
	tenVertical = tenVertical * (1.0 / fltFocal)

	return torch.cat([ tenDepth * tenHorizontal, tenDepth * tenVertical, tenDepth ], 1)
# end

def spatial_filter(tenInput, strType):
	tenOutput = None

	if strType == 'laplacian':
		tenLaplacian = tenInput.new_zeros(tenInput.shape[1], tenInput.shape[1], 3, 3)

		for intKernel in range(tenInput.shape[1]):
			tenLaplacian[intKernel, intKernel, 0, 1] = -1.0
			tenLaplacian[intKernel, intKernel, 0, 2] = -1.0
			tenLaplacian[intKernel, intKernel, 1, 1] = 4.0
			tenLaplacian[intKernel, intKernel, 1, 0] = -1.0
			tenLaplacian[intKernel, intKernel, 2, 0] = -1.0
		# end

		tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 1, 1, 1, 1 ], mode='replicate')
		tenOutput = torch.nn.functional.conv2d(input=tenOutput, weight=tenLaplacian)

	elif strType == 'median-3':
		tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 1, 1, 1, 1 ], mode='reflect')
		tenOutput = tenOutput.unfold(2, 3, 1).unfold(3, 3, 1)
		tenOutput = tenOutput.contiguous().view(tenOutput.shape[0], tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 3 * 3)
		tenOutput = tenOutput.median(-1, False)[0]

	elif strType == 'median-5':
		tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 2, 2, 2, 2 ], mode='reflect')
		tenOutput = tenOutput.unfold(2, 5, 1).unfold(3, 5, 1)
		tenOutput = tenOutput.contiguous().view(tenOutput.shape[0], tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 5 * 5)
		tenOutput = tenOutput.median(-1, False)[0]

	# end

	return tenOutput
# end

def render_pointcloud(tenInput, tenData, intWidth, intHeight, fltFocal, fltBaseline):
	tenData = torch.cat([ tenData, tenData.new_ones([ tenData.shape[0], 1, tenData.shape[2] ]) ], 1)

	tenZee = tenInput.new_zeros([ tenData.shape[0], 1, intHeight, intWidth ]).fill_(1000000.0)
	tenOutput = tenInput.new_zeros([ tenData.shape[0], tenData.shape[1], intHeight, intWidth ])

	n = tenInput.shape[0] * tenInput.shape[2]
	launch_kernel('kernel_pointrender_updateZee', preprocess_kernel('''
		extern "C" __global__ void kernel_pointrender_updateZee(
			const int n,
			const float* input,
			const float* data,
			const float* zee
		) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
			const int intSample = ( intIndex / SIZE_2(input) ) % SIZE_0(input);
			const int intPoint  = ( intIndex                 ) % SIZE_2(input);

			assert(SIZE_1(input) == 3);
			assert(SIZE_1(zee) == 1);

			float3 fltPlanePoint = make_float3(0.0, 0.0, {{fltFocal}});
			float3 fltPlaneNormal = make_float3(0.0, 0.0, 1.0);

			float3 fltLinePoint = make_float3(VALUE_3(input, intSample, 0, intPoint), VALUE_3(input, intSample, 1, intPoint), VALUE_3(input, intSample, 2, intPoint));
			float3 fltLineVector = make_float3(0.0, 0.0, 0.0) - fltLinePoint;

			if (fltLinePoint.z < 0.001) {
				return;
			}

			float fltNumerator = dot(fltPlanePoint - fltLinePoint, fltPlaneNormal);
			float fltDenominator = dot(fltLineVector, fltPlaneNormal);
			float fltDistance = fltNumerator / fltDenominator;

			if (fabs(fltDenominator) < 0.001) {
				return;
			}

			float3 fltIntersection = fltLinePoint + (fltDistance * fltLineVector); // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

			float fltOutputX = fltIntersection.x + (0.5 * SIZE_3(zee)) - 0.5;
			float fltOutputY = fltIntersection.y + (0.5 * SIZE_2(zee)) - 0.5;

			float fltError = 1000000.0 - (({{fltFocal}} * {{fltBaseline}}) / (fltLinePoint.z + 0.0000001));

			int intNorthwestX = (int) (floor(fltOutputX));
			int intNorthwestY = (int) (floor(fltOutputY));
			int intNortheastX = intNorthwestX + 1;
			int intNortheastY = intNorthwestY;
			int intSouthwestX = intNorthwestX;
			int intSouthwestY = intNorthwestY + 1;
			int intSoutheastX = intNorthwestX + 1;
			int intSoutheastY = intNorthwestY + 1;

			float fltNorthwest = (intSoutheastX - fltOutputX)    * (intSoutheastY - fltOutputY);
			float fltNortheast = (fltOutputX    - intSouthwestX) * (intSouthwestY - fltOutputY);
			float fltSouthwest = (intNortheastX - fltOutputX)    * (fltOutputY    - intNortheastY);
			float fltSoutheast = (fltOutputX    - intNorthwestX) * (fltOutputY    - intNorthwestY);

			if ((fltNorthwest >= fltNortheast) && (fltNorthwest >= fltSouthwest) && (fltNorthwest >= fltSoutheast)) {
				if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(zee)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(zee))) {
					atomicMin(&zee[OFFSET_4(zee, intSample, 0, intNorthwestY, intNorthwestX)], fltError);
				}

			} else if ((fltNortheast >= fltNorthwest) && (fltNortheast >= fltSouthwest) && (fltNortheast >= fltSoutheast)) {
				if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(zee)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(zee))) {
					atomicMin(&zee[OFFSET_4(zee, intSample, 0, intNortheastY, intNortheastX)], fltError);
				}

			} else if ((fltSouthwest >= fltNorthwest) && (fltSouthwest >= fltNortheast) && (fltSouthwest >= fltSoutheast)) {
				if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(zee)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(zee))) {
					atomicMin(&zee[OFFSET_4(zee, intSample, 0, intSouthwestY, intSouthwestX)], fltError);
				}

			} else if ((fltSoutheast >= fltNorthwest) && (fltSoutheast >= fltNortheast) && (fltSoutheast >= fltSouthwest)) {
				if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(zee)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(zee))) {
					atomicMin(&zee[OFFSET_4(zee, intSample, 0, intSoutheastY, intSoutheastX)], fltError);
				}

			}
		} }
	''', {
		'intWidth': intWidth,
		'intHeight': intHeight,
		'fltFocal': fltFocal,
		'fltBaseline': fltBaseline,
		'input': tenInput,
		'data': tenData,
		'zee': tenZee
	}))(
		grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
		block=tuple([ 512, 1, 1 ]),
		args=[ cupy.int32(n), tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr() ]
	)

	n = tenZee.nelement()
	launch_kernel('kernel_pointrender_updateDegrid', preprocess_kernel('''
		extern "C" __global__ void kernel_pointrender_updateDegrid(
			const int n,
			const float* input,
			const float* data,
			float* zee
		) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
			const int intN = ( intIndex / SIZE_3(zee) / SIZE_2(zee) / SIZE_1(zee) ) % SIZE_0(zee);
			const int intC = ( intIndex / SIZE_3(zee) / SIZE_2(zee)               ) % SIZE_1(zee);
			const int intY = ( intIndex / SIZE_3(zee)                             ) % SIZE_2(zee);
			const int intX = ( intIndex                                           ) % SIZE_3(zee);

			assert(SIZE_1(input) == 3);
			assert(SIZE_1(zee) == 1);

			int intCount = 0;
			float fltSum = 0.0;

			int intOpposingX[] = {  1,  0,  1,  1 };
			int intOpposingY[] = {  0,  1,  1, -1 };

			for (int intOpposing = 0; intOpposing < 4; intOpposing += 1) {
				int intOneX = intX + intOpposingX[intOpposing];
				int intOneY = intY + intOpposingY[intOpposing];
				int intTwoX = intX - intOpposingX[intOpposing];
				int intTwoY = intY - intOpposingY[intOpposing];

				if ((intOneX < 0) | (intOneX >= SIZE_3(zee)) | (intOneY < 0) | (intOneY >= SIZE_2(zee))) {
					continue;

				} else if ((intTwoX < 0) | (intTwoX >= SIZE_3(zee)) | (intTwoY < 0) | (intTwoY >= SIZE_2(zee))) {
					continue;

				}

				if (VALUE_4(zee, intN, intC, intY, intX) >= VALUE_4(zee, intN, intC, intOneY, intOneX) + 1.0) {
					if (VALUE_4(zee, intN, intC, intY, intX) >= VALUE_4(zee, intN, intC, intTwoY, intTwoX) + 1.0) {
						intCount += 2;
						fltSum += VALUE_4(zee, intN, intC, intOneY, intOneX);
						fltSum += VALUE_4(zee, intN, intC, intTwoY, intTwoX);
					}
				}
			}

			if (intCount > 0) {
				zee[OFFSET_4(zee, intN, intC, intY, intX)] = min(VALUE_4(zee, intN, intC, intY, intX), fltSum / intCount);
			}
		} }
	''', {
		'intWidth': intWidth,
		'intHeight': intHeight,
		'fltFocal': fltFocal,
		'fltBaseline': fltBaseline,
		'input': tenInput,
		'data': tenData,
		'zee': tenZee
	}))(
		grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
		block=tuple([ 512, 1, 1 ]),
		args=[ cupy.int32(n), tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr() ]
	)

	n = tenInput.shape[0] * tenInput.shape[2]
	launch_kernel('kernel_pointrender_updateOutput', preprocess_kernel('''
		extern "C" __global__ void kernel_pointrender_updateOutput(
			const int n,
			const float* input,
			const float* data,
			const float* zee,
			float* output
		) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
			const int intSample = ( intIndex / SIZE_2(input) ) % SIZE_0(input);
			const int intPoint  = ( intIndex                 ) % SIZE_2(input);

			assert(SIZE_1(input) == 3);
			assert(SIZE_1(zee) == 1);

			float3 fltPlanePoint = make_float3(0.0, 0.0, {{fltFocal}});
			float3 fltPlaneNormal = make_float3(0.0, 0.0, 1.0);

			float3 fltLinePoint = make_float3(VALUE_3(input, intSample, 0, intPoint), VALUE_3(input, intSample, 1, intPoint), VALUE_3(input, intSample, 2, intPoint));
			float3 fltLineVector = make_float3(0.0, 0.0, 0.0) - fltLinePoint;

			if (fltLinePoint.z < 0.001) {
				return;
			}

			float fltNumerator = dot(fltPlanePoint - fltLinePoint, fltPlaneNormal);
			float fltDenominator = dot(fltLineVector, fltPlaneNormal);
			float fltDistance = fltNumerator / fltDenominator;

			if (fabs(fltDenominator) < 0.001) {
				return;
			}

			float3 fltIntersection = fltLinePoint + (fltDistance * fltLineVector); // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

			float fltOutputX = fltIntersection.x + (0.5 * SIZE_3(output)) - 0.5;
			float fltOutputY = fltIntersection.y + (0.5 * SIZE_2(output)) - 0.5;

			float fltError = 1000000.0 - (({{fltFocal}} * {{fltBaseline}}) / (fltLinePoint.z + 0.0000001));

			int intNorthwestX = (int) (floor(fltOutputX));
			int intNorthwestY = (int) (floor(fltOutputY));
			int intNortheastX = intNorthwestX + 1;
			int intNortheastY = intNorthwestY;
			int intSouthwestX = intNorthwestX;
			int intSouthwestY = intNorthwestY + 1;
			int intSoutheastX = intNorthwestX + 1;
			int intSoutheastY = intNorthwestY + 1;

			float fltNorthwest = (intSoutheastX - fltOutputX)    * (intSoutheastY - fltOutputY);
			float fltNortheast = (fltOutputX    - intSouthwestX) * (intSouthwestY - fltOutputY);
			float fltSouthwest = (intNortheastX - fltOutputX)    * (fltOutputY    - intNortheastY);
			float fltSoutheast = (fltOutputX    - intNorthwestX) * (fltOutputY    - intNorthwestY);

			if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(output)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(output))) {
				if (fltError <= VALUE_4(zee, intSample, 0, intNorthwestY, intNorthwestX) + 1.0) {
					for (int intData = 0; intData < SIZE_1(data); intData += 1) {
						atomicAdd(&output[OFFSET_4(output, intSample, intData, intNorthwestY, intNorthwestX)], VALUE_3(data, intSample, intData, intPoint) * fltNorthwest);
					}
				}
			}

			if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(output)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(output))) {
				if (fltError <= VALUE_4(zee, intSample, 0, intNortheastY, intNortheastX) + 1.0) {
					for (int intData = 0; intData < SIZE_1(data); intData += 1) {
						atomicAdd(&output[OFFSET_4(output, intSample, intData, intNortheastY, intNortheastX)], VALUE_3(data, intSample, intData, intPoint) * fltNortheast);
					}
				}
			}

			if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(output)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(output))) {
				if (fltError <= VALUE_4(zee, intSample, 0, intSouthwestY, intSouthwestX) + 1.0) {
					for (int intData = 0; intData < SIZE_1(data); intData += 1) {
						atomicAdd(&output[OFFSET_4(output, intSample, intData, intSouthwestY, intSouthwestX)], VALUE_3(data, intSample, intData, intPoint) * fltSouthwest);
					}
				}
			}

			if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(output)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(output))) {
				if (fltError <= VALUE_4(zee, intSample, 0, intSoutheastY, intSoutheastX) + 1.0) {
					for (int intData = 0; intData < SIZE_1(data); intData += 1) {
						atomicAdd(&output[OFFSET_4(output, intSample, intData, intSoutheastY, intSoutheastX)], VALUE_3(data, intSample, intData, intPoint) * fltSoutheast);
					}
				}
			}
		} }
	''', {
		'intWidth': intWidth,
		'intHeight': intHeight,
		'fltFocal': fltFocal,
		'fltBaseline': fltBaseline,
		'input': tenInput,
		'data': tenData,
		'zee': tenZee,
		'output': tenOutput
	}))(
		grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
		block=tuple([ 512, 1, 1 ]),
		args=[ cupy.int32(n), tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr(), tenOutput.data_ptr() ]
	)

	return tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 0.0000001), tenOutput[:, -1:, :, :].detach().clone()
# end

def fill_disocclusion(tenInput, tenDepth):
	tenOutput = tenInput.clone()

	n = tenInput.shape[0] * tenInput.shape[2] * tenInput.shape[3]
	launch_kernel('kernel_discfill_updateOutput', preprocess_kernel('''
		extern "C" __global__ void kernel_discfill_updateOutput(
			const int n,
			const float* input,
			const float* depth,
			float* output
		) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
			const int intSample = ( intIndex / SIZE_3(input) / SIZE_2(input) ) % SIZE_0(input);
			const int intY      = ( intIndex / SIZE_3(input)                 ) % SIZE_2(input);
			const int intX      = ( intIndex                                 ) % SIZE_3(input);

			assert(SIZE_1(depth) == 1);

			if (VALUE_4(depth, intSample, 0, intY, intX) > 0.0) {
				return;
			}

			float fltShortest = 1000000.0;

			int intFillX = -1;
			int intFillY = -1;

			float fltDirectionX[] = { -1, 0, 1, 1,    -1, 1, 2,  2,    -2, -1, 1, 2, 3, 3,  3,  3 };
			float fltDirectionY[] = {  1, 1, 1, 0,     2, 2, 1, -1,     3,  3, 3, 3, 2, 1, -1, -2 };

			for (int intDirection = 0; intDirection < 16; intDirection += 1) {
				float fltNormalize = sqrt((fltDirectionX[intDirection] * fltDirectionX[intDirection]) + (fltDirectionY[intDirection] * fltDirectionY[intDirection]));

				fltDirectionX[intDirection] /= fltNormalize;
				fltDirectionY[intDirection] /= fltNormalize;
			}

			for (int intDirection = 0; intDirection < 16; intDirection += 1) {
				float fltFromX = intX; int intFromX = 0;
				float fltFromY = intY; int intFromY = 0;

				float fltToX = intX; int intToX = 0;
				float fltToY = intY; int intToY = 0;

				do {
					fltFromX -= fltDirectionX[intDirection]; intFromX = (int) (round(fltFromX));
					fltFromY -= fltDirectionY[intDirection]; intFromY = (int) (round(fltFromY));

					if ((intFromX < 0) | (intFromX >= SIZE_3(input))) { break; }
					if ((intFromY < 0) | (intFromY >= SIZE_2(input))) { break; }
					if (VALUE_4(depth, intSample, 0, intFromY, intFromX) > 0.0) { break; }
				} while (true);
				if ((intFromX < 0) | (intFromX >= SIZE_3(input))) { continue; }
				if ((intFromY < 0) | (intFromY >= SIZE_2(input))) { continue; }

				do {
					fltToX += fltDirectionX[intDirection]; intToX = (int) (round(fltToX));
					fltToY += fltDirectionY[intDirection]; intToY = (int) (round(fltToY));

					if ((intToX < 0) | (intToX >= SIZE_3(input))) { break; }
					if ((intToY < 0) | (intToY >= SIZE_2(input))) { break; }
					if (VALUE_4(depth, intSample, 0, intToY, intToX) > 0.0) { break; }
				} while (true);
				if ((intToX < 0) | (intToX >= SIZE_3(input))) { continue; }
				if ((intToY < 0) | (intToY >= SIZE_2(input))) { continue; }

				float fltDistance = sqrt(powf(intToX - intFromX, 2) + powf(intToY - intFromY, 2));

				if (fltShortest > fltDistance) {
					intFillX = intFromX;
					intFillY = intFromY;

					if (VALUE_4(depth, intSample, 0, intFromY, intFromX) < VALUE_4(depth, intSample, 0, intToY, intToX)) {
						intFillX = intToX;
						intFillY = intToY;
					}

					fltShortest = fltDistance;
				}
			}

			if (intFillX == -1) {
				return;

			} else if (intFillY == -1) {
				return;

			}

			for (int intDepth = 0; intDepth < SIZE_1(input); intDepth += 1) {
				output[OFFSET_4(output, intSample, intDepth, intY, intX)] = VALUE_4(input, intSample, intDepth, intFillY, intFillX);
			}
		} }
	''', {
		'input': tenInput,
		'depth': tenDepth,
		'output': tenOutput
	}))(
		grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
		block=tuple([ 512, 1, 1 ]),
		args=[ cupy.int32(n), tenInput.data_ptr(), tenDepth.data_ptr(), tenOutput.data_ptr() ]
	)

	return tenOutput
# end