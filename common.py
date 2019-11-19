def process_load(numpyImage, objectSettings):
	objectCommon['dblFocal'] = 1024 / 2.0
	objectCommon['dblBaseline'] = 40.0
	objectCommon['intWidth'] = numpyImage.shape[1]
	objectCommon['intHeight'] = numpyImage.shape[0]

	tensorImage = torch.FloatTensor(numpyImage.transpose(2, 0, 1)).unsqueeze(0).cuda() / 255.0
	tensorDisparity = disparity_estimation(tensorImage)
	tensorDisparity = disparity_adjustment(tensorImage, tensorDisparity)
	tensorDisparity = disparity_refinement(tensorImage, tensorDisparity)
	tensorDisparity = tensorDisparity / tensorDisparity.max() * objectCommon['dblBaseline']
	tensorDepth = (objectCommon['dblFocal'] * objectCommon['dblBaseline']) / (tensorDisparity + 0.0000001)
	tensorValid = (spatial_filter(tensorDisparity / tensorDisparity.max(), 'laplacian').abs() < 0.03).float()
	tensorPoints = depth_to_points(tensorDepth * tensorValid, objectCommon['dblFocal'])
	tensorUnaltered = depth_to_points(tensorDepth, objectCommon['dblFocal'])

	objectCommon['dblDispmin'] = tensorDisparity.min().item()
	objectCommon['dblDispmax'] = tensorDisparity.max().item()
	objectCommon['objectDepthrange'] = cv2.minMaxLoc(src=tensorDepth[0, 0, 128:-128, 128:-128].detach().cpu().numpy(), mask=None)
	objectCommon['tensorRawImage'] = tensorImage
	objectCommon['tensorRawDisparity'] = tensorDisparity
	objectCommon['tensorRawDepth'] = tensorDepth
	objectCommon['tensorRawPoints'] = tensorPoints.view(1, 3, -1)
	objectCommon['tensorRawUnaltered'] = tensorUnaltered.view(1, 3, -1)

	objectCommon['tensorInpaImage'] = objectCommon['tensorRawImage'].view(1, 3, -1)
	objectCommon['tensorInpaDisparity'] = objectCommon['tensorRawDisparity'].view(1, 1, -1)
	objectCommon['tensorInpaDepth'] = objectCommon['tensorRawDepth'].view(1, 1, -1)
	objectCommon['tensorInpaPoints'] = objectCommon['tensorRawPoints'].view(1, 3, -1)
# end

def process_inpaint(tensorShift):
	objectInpainted = pointcloud_inpainting(objectCommon['tensorRawImage'], objectCommon['tensorRawDisparity'], tensorShift)

	objectInpainted['tensorDepth'] = (objectCommon['dblFocal'] * objectCommon['dblBaseline']) / (objectInpainted['tensorDisparity'] + 0.0000001)
	objectInpainted['tensorValid'] = (spatial_filter(objectInpainted['tensorDisparity'] / objectInpainted['tensorDisparity'].max(), 'laplacian').abs() < 0.03).float()
	objectInpainted['tensorPoints'] = depth_to_points(objectInpainted['tensorDepth'] * objectInpainted['tensorValid'], objectCommon['dblFocal'])
	objectInpainted['tensorPoints'] = objectInpainted['tensorPoints'].view(1, 3, -1)
	objectInpainted['tensorPoints'] = objectInpainted['tensorPoints'] - tensorShift

	tensorMask = (objectInpainted['tensorExisting'] == 0.0).view(1, 1, -1)

	objectCommon['tensorInpaImage'] = torch.cat([ objectCommon['tensorInpaImage'], objectInpainted['tensorImage'].view(1, 3, -1)[tensorMask.expand(-1, 3, -1)].view(1, 3, -1) ], 2)
	objectCommon['tensorInpaDisparity'] = torch.cat([ objectCommon['tensorInpaDisparity'], objectInpainted['tensorDisparity'].view(1, 1, -1)[tensorMask.expand(-1, 1, -1)].view(1, 1, -1) ], 2)
	objectCommon['tensorInpaDepth'] = torch.cat([ objectCommon['tensorInpaDepth'], objectInpainted['tensorDepth'].view(1, 1, -1)[tensorMask.expand(-1, 1, -1)].view(1, 1, -1) ], 2)
	objectCommon['tensorInpaPoints'] = torch.cat([ objectCommon['tensorInpaPoints'], objectInpainted['tensorPoints'].view(1, 3, -1)[tensorMask.expand(-1, 3, -1)].view(1, 3, -1) ], 2)
# end

def process_shift(objectSettings):
	dblClosestDepth = objectCommon['objectDepthrange'][0] + (objectSettings['dblDepthTo'] - objectSettings['dblDepthFrom'])
	dblClosestFromU = objectCommon['objectDepthrange'][2][0]
	dblClosestFromV = objectCommon['objectDepthrange'][2][1]
	dblClosestToU = dblClosestFromU + objectSettings['dblShiftU']
	dblClosestToV = dblClosestFromV + objectSettings['dblShiftV']
	dblClosestFromX = ((dblClosestFromU - (objectCommon['intWidth'] / 2.0)) * dblClosestDepth) / objectCommon['dblFocal']
	dblClosestFromY = ((dblClosestFromV - (objectCommon['intHeight'] / 2.0)) * dblClosestDepth) / objectCommon['dblFocal']
	dblClosestToX = ((dblClosestToU - (objectCommon['intWidth'] / 2.0)) * dblClosestDepth) / objectCommon['dblFocal']
	dblClosestToY = ((dblClosestToV - (objectCommon['intHeight'] / 2.0)) * dblClosestDepth) / objectCommon['dblFocal']

	dblShiftX = dblClosestFromX - dblClosestToX
	dblShiftY = dblClosestFromY - dblClosestToY
	dblShiftZ = objectSettings['dblDepthTo'] - objectSettings['dblDepthFrom']

	tensorShift = torch.FloatTensor([ dblShiftX, dblShiftY, dblShiftZ ]).view(1, 3, 1).cuda()

	tensorPoints = objectSettings['tensorPoints'].clone()

	tensorPoints[:, 0:1, :] *= tensorPoints[:, 2:3, :] / (objectSettings['tensorPoints'][:, 2:3, :] + 0.0000001)
	tensorPoints[:, 1:2, :] *= tensorPoints[:, 2:3, :] / (objectSettings['tensorPoints'][:, 2:3, :] + 0.0000001)

	tensorPoints += tensorShift

	return tensorPoints, tensorShift
# end

def process_autozoom(objectSettings):
	numpyShiftU = numpy.linspace(-objectSettings['dblShift'], objectSettings['dblShift'], 16)[None, :].repeat(16, 0)
	numpyShiftV = numpy.linspace(-objectSettings['dblShift'], objectSettings['dblShift'], 16)[:, None].repeat(16, 1)
	dblCropWidth = objectSettings['objectFrom']['intCropWidth'] / objectSettings['dblZoom']
	dblCropHeight = objectSettings['objectFrom']['intCropHeight'] / objectSettings['dblZoom']

	dblDepthFrom = objectCommon['objectDepthrange'][0]
	dblDepthTo = objectCommon['objectDepthrange'][0] * (dblCropWidth / objectSettings['objectFrom']['intCropWidth'])

	dblBest = 0.0
	dblBestU = None
	dblBestV = None

	for intU in range(16):
		for intV in range(16):
			dblShiftU = numpyShiftU[intU, intV].item()
			dblShiftV = numpyShiftV[intU, intV].item()

			if objectSettings['objectFrom']['dblCenterU'] + dblShiftU < dblCropWidth / 2.0:
				continue

			elif objectSettings['objectFrom']['dblCenterU'] + dblShiftU > objectCommon['intWidth'] - (dblCropWidth / 2.0):
				continue

			elif objectSettings['objectFrom']['dblCenterV'] + dblShiftV < dblCropHeight / 2.0:
				continue

			elif objectSettings['objectFrom']['dblCenterV'] + dblShiftV > objectCommon['intHeight'] - (dblCropHeight / 2.0):
				continue

			# end

			tensorPoints = process_shift({
				'tensorPoints': objectCommon['tensorRawPoints'],
				'dblShiftU': dblShiftU,
				'dblShiftV': dblShiftV,
				'dblDepthFrom': dblDepthFrom,
				'dblDepthTo': dblDepthTo
			})[0]

			tensorRender, tensorExisting = render_pointcloud(tensorPoints, objectCommon['tensorRawImage'].view(1, 3, -1), objectCommon['intWidth'], objectCommon['intHeight'], objectCommon['dblFocal'], objectCommon['dblBaseline'])

			if dblBest < (tensorExisting > 0.0).float().sum().item():
				dblBest = (tensorExisting > 0.0).float().sum().item()
				dblBestU = dblShiftU
				dblBestV = dblShiftV
			# end
		# end
	# end

	return {
		'dblCenterU': objectSettings['objectFrom']['dblCenterU'] + dblBestU,
		'dblCenterV': objectSettings['objectFrom']['dblCenterV'] + dblBestV,
		'intCropWidth': int(round(objectSettings['objectFrom']['intCropWidth'] / objectSettings['dblZoom'])),
		'intCropHeight': int(round(objectSettings['objectFrom']['intCropHeight'] / objectSettings['dblZoom']))
	}
# end

def process_kenburns(objectSettings):
	numpyOutputs = []

	if 'boolInpaint' not in objectSettings or objectSettings['boolInpaint'] == True:
		objectCommon['tensorInpaImage'] = objectCommon['tensorRawImage'].view(1, 3, -1)
		objectCommon['tensorInpaDisparity'] = objectCommon['tensorRawDisparity'].view(1, 1, -1)
		objectCommon['tensorInpaDepth'] = objectCommon['tensorRawDepth'].view(1, 1, -1)
		objectCommon['tensorInpaPoints'] = objectCommon['tensorRawPoints'].view(1, 3, -1)

		for dblStep in [ 0.0, 1.0 ]:
			dblFrom = 1.0 - dblStep
			dblTo = 1.0 - dblFrom

			dblShiftU = ((dblFrom * objectSettings['objectFrom']['dblCenterU']) + (dblTo * objectSettings['objectTo']['dblCenterU'])) - (objectCommon['intWidth'] / 2.0)
			dblShiftV = ((dblFrom * objectSettings['objectFrom']['dblCenterV']) + (dblTo * objectSettings['objectTo']['dblCenterV'])) - (objectCommon['intHeight'] / 2.0)
			dblCropWidth = (dblFrom * objectSettings['objectFrom']['intCropWidth']) + (dblTo * objectSettings['objectTo']['intCropWidth'])
			dblCropHeight = (dblFrom * objectSettings['objectFrom']['intCropHeight']) + (dblTo * objectSettings['objectTo']['intCropHeight'])

			dblDepthFrom = objectCommon['objectDepthrange'][0]
			dblDepthTo = objectCommon['objectDepthrange'][0] * (dblCropWidth / max(objectSettings['objectFrom']['intCropWidth'], objectSettings['objectTo']['intCropWidth']))

			tensorShift = process_shift({
				'tensorPoints': objectCommon['tensorInpaPoints'],
				'dblShiftU': dblShiftU,
				'dblShiftV': dblShiftV,
				'dblDepthFrom': dblDepthFrom,
				'dblDepthTo': dblDepthTo
			})[1]

			process_inpaint(1.1 * tensorShift)
		# end
	# end

	for dblStep in objectSettings['dblSteps']:
		dblFrom = 1.0 - dblStep
		dblTo = 1.0 - dblFrom

		dblShiftU = ((dblFrom * objectSettings['objectFrom']['dblCenterU']) + (dblTo * objectSettings['objectTo']['dblCenterU'])) - (objectCommon['intWidth'] / 2.0)
		dblShiftV = ((dblFrom * objectSettings['objectFrom']['dblCenterV']) + (dblTo * objectSettings['objectTo']['dblCenterV'])) - (objectCommon['intHeight'] / 2.0)
		dblCropWidth = (dblFrom * objectSettings['objectFrom']['intCropWidth']) + (dblTo * objectSettings['objectTo']['intCropWidth'])
		dblCropHeight = (dblFrom * objectSettings['objectFrom']['intCropHeight']) + (dblTo * objectSettings['objectTo']['intCropHeight'])

		dblDepthFrom = objectCommon['objectDepthrange'][0]
		dblDepthTo = objectCommon['objectDepthrange'][0] * (dblCropWidth / max(objectSettings['objectFrom']['intCropWidth'], objectSettings['objectTo']['intCropWidth']))

		tensorPoints = process_shift({
			'tensorPoints': objectCommon['tensorInpaPoints'],
			'dblShiftU': dblShiftU,
			'dblShiftV': dblShiftV,
			'dblDepthFrom': dblDepthFrom,
			'dblDepthTo': dblDepthTo
		})[0]

		tensorRender, tensorExisting = render_pointcloud(tensorPoints, torch.cat([ objectCommon['tensorInpaImage'], objectCommon['tensorInpaDepth'] ], 1).view(1, 4, -1), objectCommon['intWidth'], objectCommon['intHeight'], objectCommon['dblFocal'], objectCommon['dblBaseline'])

		tensorRender = fill_disocclusion(tensorRender, tensorRender[:, 3:4, :, :] * (tensorExisting > 0.0).float())

		numpyOutput = (tensorRender[0, 0:3, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).astype(numpy.uint8)
		numpyOutput = cv2.getRectSubPix(image=numpyOutput, patchSize=(max(objectSettings['objectFrom']['intCropWidth'], objectSettings['objectTo']['intCropWidth']), max(objectSettings['objectFrom']['intCropHeight'], objectSettings['objectTo']['intCropHeight'])), center=(objectCommon['intWidth'] / 2.0, objectCommon['intHeight'] / 2.0))
		numpyOutput = cv2.resize(src=numpyOutput, dsize=(objectCommon['intWidth'], objectCommon['intHeight']), fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR)

		numpyOutputs.append(numpyOutput)
	# end

	return numpyOutputs
# end

##########################################################

class Stream:
	ptr = torch.cuda.current_stream().cuda_stream
# end

def preprocess_kernel(strKernel, objectVariables):
	strKernel = '''
		#include <samples/common/inc/helper_math.h>

		__device__ __forceinline__ float atomicMin(const float* buffer, float dblValue) {
			int intValue = __float_as_int(*buffer);

			while (__int_as_float(intValue) > dblValue) {
				intValue = atomicCAS((int*) (buffer), intValue, __float_as_int(dblValue));
			}

			return __int_as_float(intValue);
		}
	''' + strKernel

	for strVariable in objectVariables:
		objectValue = objectVariables[strVariable]

		if type(objectValue) == int:
			strKernel = strKernel.replace('{{' + strVariable + '}}', str(objectValue))

		elif type(objectValue) == float:
			strKernel = strKernel.replace('{{' + strVariable + '}}', str(objectValue))

		elif type(objectValue) == str:
			strKernel = strKernel.replace('{{' + strVariable + '}}', objectValue)

		# end
	# end

	while True:
		objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intSizes = objectVariables[strTensor].size()

		strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objectMatch = re.search('(STRIDE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intStrides = objectVariables[strTensor].stride()

		strKernel = strKernel.replace(objectMatch.group(), str(intStrides[intArg]))
	# end

	while True:
		objectMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), '(' + str.join('+', strIndex) + ')')
	# end

	while True:
		objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy.util.memoize(for_each_device=True)
def launch_kernel(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel, tuple([ '-I ' + os.environ['CUDA_HOME'], '-I ' + os.environ['CUDA_HOME'] + '/include' ])).get_function(strFunction)
# end

def depth_to_points(tensorDepth, dblFocal):
	tensorHorizontal = torch.linspace((-0.5 * tensorDepth.size(3)) + 0.5, (0.5 * tensorDepth.size(3)) - 0.5, tensorDepth.size(3)).view(1, 1, 1, tensorDepth.size(3)).expand(tensorDepth.size(0), -1, tensorDepth.size(2), -1)
	tensorHorizontal = tensorHorizontal * (1.0 / dblFocal)
	tensorHorizontal = tensorHorizontal.type_as(tensorDepth)

	tensorVertical = torch.linspace((-0.5 * tensorDepth.size(2)) + 0.5, (0.5 * tensorDepth.size(2)) - 0.5, tensorDepth.size(2)).view(1, 1, tensorDepth.size(2), 1).expand(tensorDepth.size(0), -1, -1, tensorDepth.size(3))
	tensorVertical = tensorVertical * (1.0 / dblFocal)
	tensorVertical = tensorVertical.type_as(tensorDepth)

	return torch.cat([ tensorDepth * tensorHorizontal, tensorDepth * tensorVertical, tensorDepth ], 1)
# end

def spatial_filter(tensorInput, strType):
	tensorOutput = None

	if strType == 'laplacian':
		tensorLaplacian = tensorInput.new_zeros(tensorInput.size(1), tensorInput.size(1), 3, 3)

		for intKernel in range(tensorInput.size(1)):
			tensorLaplacian[intKernel, intKernel, 0, 1] = -1.0
			tensorLaplacian[intKernel, intKernel, 0, 2] = -1.0
			tensorLaplacian[intKernel, intKernel, 1, 1] = 4.0
			tensorLaplacian[intKernel, intKernel, 1, 0] = -1.0
			tensorLaplacian[intKernel, intKernel, 2, 0] = -1.0
		# end

		tensorOutput = torch.nn.functional.pad(input=tensorInput, pad=[ 1, 1, 1, 1 ], mode='replicate')
		tensorOutput = torch.nn.functional.conv2d(input=tensorOutput, weight=tensorLaplacian)

	elif strType == 'median-3':
		tensorOutput = torch.nn.functional.pad(input=tensorInput, pad=[ 1, 1, 1, 1 ], mode='reflect')
		tensorOutput = tensorOutput.unfold(2, 3, 1).unfold(3, 3, 1)
		tensorOutput = tensorOutput.contiguous().view(tensorOutput.size(0), tensorOutput.size(1), tensorOutput.size(2), tensorOutput.size(3), 3 * 3)
		tensorOutput = tensorOutput.median(-1, False)[0]

	elif strType == 'median-5':
		tensorOutput = torch.nn.functional.pad(input=tensorInput, pad=[ 2, 2, 2, 2 ], mode='reflect')
		tensorOutput = tensorOutput.unfold(2, 5, 1).unfold(3, 5, 1)
		tensorOutput = tensorOutput.contiguous().view(tensorOutput.size(0), tensorOutput.size(1), tensorOutput.size(2), tensorOutput.size(3), 5 * 5)
		tensorOutput = tensorOutput.median(-1, False)[0]

	# end

	return tensorOutput
# end

def render_pointcloud(tensorInput, tensorData, intWidth, intHeight, dblFocal, dblBaseline):
	tensorData = torch.cat([ tensorData, tensorData.new_ones([ tensorData.size(0), 1, tensorData.size(2) ]) ], 1)

	tensorZee = tensorInput.new_zeros([ tensorData.size(0), 1, intHeight, intWidth ]).fill_(1000000.0)
	tensorOutput = tensorInput.new_zeros([ tensorData.size(0), tensorData.size(1), intHeight, intWidth ])

	n = tensorInput.size(0) * tensorInput.size(2)
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

			float3 dblPlanePoint = make_float3(0.0, 0.0, {{dblFocal}});
			float3 dblPlaneNormal = make_float3(0.0, 0.0, 1.0);

			float3 dblLinePoint = make_float3(VALUE_3(input, intSample, 0, intPoint), VALUE_3(input, intSample, 1, intPoint), VALUE_3(input, intSample, 2, intPoint));
			float3 dblLineVector = make_float3(0.0, 0.0, 0.0) - dblLinePoint;

			if (dblLinePoint.z < 0.001) {
				return;
			}

			float dblNumerator = dot(dblPlanePoint - dblLinePoint, dblPlaneNormal);
			float dblDenominator = dot(dblLineVector, dblPlaneNormal);
			float dblDistance = dblNumerator / dblDenominator;

			if (fabs(dblDenominator) < 0.001) {
				return;
			}

			float3 dblIntersection = dblLinePoint + (dblDistance * dblLineVector); // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

			float dblOutputX = dblIntersection.x + (0.5 * SIZE_3(zee)) - 0.5;
			float dblOutputY = dblIntersection.y + (0.5 * SIZE_2(zee)) - 0.5;

			float dblError = 1000000.0 - (({{dblFocal}} * {{dblBaseline}}) / (dblLinePoint.z + 0.0000001));

			int intNorthwestX = (int) (floor(dblOutputX));
			int intNorthwestY = (int) (floor(dblOutputY));
			int intNortheastX = intNorthwestX + 1;
			int intNortheastY = intNorthwestY;
			int intSouthwestX = intNorthwestX;
			int intSouthwestY = intNorthwestY + 1;
			int intSoutheastX = intNorthwestX + 1;
			int intSoutheastY = intNorthwestY + 1;

			float dblNorthwest = (intSoutheastX - dblOutputX)    * (intSoutheastY - dblOutputY);
			float dblNortheast = (dblOutputX    - intSouthwestX) * (intSouthwestY - dblOutputY);
			float dblSouthwest = (intNortheastX - dblOutputX)    * (dblOutputY    - intNortheastY);
			float dblSoutheast = (dblOutputX    - intNorthwestX) * (dblOutputY    - intNorthwestY);

			if ((dblNorthwest >= dblNortheast) & (dblNorthwest >= dblSouthwest) & (dblNorthwest >= dblSoutheast)) {
				if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(zee)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(zee))) {
					atomicMin(&zee[OFFSET_4(zee, intSample, 0, intNorthwestY, intNorthwestX)], dblError);
				}

			} else if ((dblNortheast >= dblNorthwest) & (dblNortheast >= dblSouthwest) & (dblNortheast >= dblSoutheast)) {
				if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(zee)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(zee))) {
					atomicMin(&zee[OFFSET_4(zee, intSample, 0, intNortheastY, intNortheastX)], dblError);
				}

			} else if ((dblSouthwest >= dblNorthwest) & (dblSouthwest >= dblNortheast) & (dblSouthwest >= dblSoutheast)) {
				if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(zee)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(zee))) {
					atomicMin(&zee[OFFSET_4(zee, intSample, 0, intSouthwestY, intSouthwestX)], dblError);
				}

			} else if ((dblSoutheast >= dblNorthwest) & (dblSoutheast >= dblNortheast) & (dblSoutheast >= dblSouthwest)) {
				if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(zee)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(zee))) {
					atomicMin(&zee[OFFSET_4(zee, intSample, 0, intSoutheastY, intSoutheastX)], dblError);
				}

			}
		} }
	''', {
		'intWidth': intWidth,
		'intHeight': intHeight,
		'dblFocal': dblFocal,
		'dblBaseline': dblBaseline,
		'input': tensorInput,
		'data': tensorData,
		'zee': tensorZee
	}))(
		grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
		block=tuple([ 512, 1, 1 ]),
		args=[ n, tensorInput.data_ptr(), tensorData.data_ptr(), tensorZee.data_ptr() ],
		stream=Stream
	)

	n = tensorZee.nelement()
	launch_kernel('kernel_pointrender_updateDegrid', preprocess_kernel('''
		extern "C" __global__ void kernel_pointrender_updateDegrid(
			const int n,
			const float* input,
			const float* data,
			float* zee
		) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
			const int intSample = ( intIndex / SIZE_3(zee) / SIZE_2(zee) / SIZE_1(zee) ) % SIZE_0(zee);
			const int intDepth  = ( intIndex / SIZE_3(zee) / SIZE_2(zee)                ) % SIZE_1(zee);
			const int intY      = ( intIndex / SIZE_3(zee)                               ) % SIZE_2(zee);
			const int intX      = ( intIndex                                              ) % SIZE_3(zee);

			int intCount = 0;
			float dblSum = 0.0;

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

				if (VALUE_4(zee, intSample, 0, intY, intX) >= VALUE_4(zee, intSample, 0, intOneY, intOneX) + 1.0) {
					if (VALUE_4(zee, intSample, 0, intY, intX) >= VALUE_4(zee, intSample, 0, intTwoY, intTwoX) + 1.0) {
						intCount += 2;
						dblSum += VALUE_4(zee, intSample, 0, intOneY, intOneX);
						dblSum += VALUE_4(zee, intSample, 0, intTwoY, intTwoX);
					}
				}
			}

			if (intCount > 0) {
				zee[OFFSET_4(zee, intSample, 0, intY, intX)] = min(VALUE_4(zee, intSample, 0, intY, intX), dblSum / intCount);
			}
		} }
	''', {
		'intWidth': intWidth,
		'intHeight': intHeight,
		'dblFocal': dblFocal,
		'dblBaseline': dblBaseline,
		'input': tensorInput,
		'data': tensorData,
		'zee': tensorZee
	}))(
		grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
		block=tuple([ 512, 1, 1 ]),
		args=[ n, tensorInput.data_ptr(), tensorData.data_ptr(), tensorZee.data_ptr() ],
		stream=Stream
	)

	n = tensorInput.size(0) * tensorInput.size(2)
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

			float3 dblPlanePoint = make_float3(0.0, 0.0, {{dblFocal}});
			float3 dblPlaneNormal = make_float3(0.0, 0.0, 1.0);

			float3 dblLinePoint = make_float3(VALUE_3(input, intSample, 0, intPoint), VALUE_3(input, intSample, 1, intPoint), VALUE_3(input, intSample, 2, intPoint));
			float3 dblLineVector = make_float3(0.0, 0.0, 0.0) - dblLinePoint;

			if (dblLinePoint.z < 0.001) {
				return;
			}

			float dblNumerator = dot(dblPlanePoint - dblLinePoint, dblPlaneNormal);
			float dblDenominator = dot(dblLineVector, dblPlaneNormal);
			float dblDistance = dblNumerator / dblDenominator;

			if (fabs(dblDenominator) < 0.001) {
				return;
			}

			float3 dblIntersection = dblLinePoint + (dblDistance * dblLineVector); // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

			float dblOutputX = dblIntersection.x + (0.5 * SIZE_3(output)) - 0.5;
			float dblOutputY = dblIntersection.y + (0.5 * SIZE_2(output)) - 0.5;

			float dblError = 1000000.0 - (({{dblFocal}} * {{dblBaseline}}) / (dblLinePoint.z + 0.0000001));

			int intNorthwestX = (int) (floor(dblOutputX));
			int intNorthwestY = (int) (floor(dblOutputY));
			int intNortheastX = intNorthwestX + 1;
			int intNortheastY = intNorthwestY;
			int intSouthwestX = intNorthwestX;
			int intSouthwestY = intNorthwestY + 1;
			int intSoutheastX = intNorthwestX + 1;
			int intSoutheastY = intNorthwestY + 1;

			float dblNorthwest = (intSoutheastX - dblOutputX)    * (intSoutheastY - dblOutputY);
			float dblNortheast = (dblOutputX    - intSouthwestX) * (intSouthwestY - dblOutputY);
			float dblSouthwest = (intNortheastX - dblOutputX)    * (dblOutputY    - intNortheastY);
			float dblSoutheast = (dblOutputX    - intNorthwestX) * (dblOutputY    - intNorthwestY);

			if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
				if (dblError <= VALUE_4(zee, intSample, 0, intNorthwestY, intNorthwestX) + 1.0) {
					for (int intData = 0; intData < SIZE_1(data); intData += 1) {
						atomicAdd(&output[OFFSET_4(output, intSample, intData, intNorthwestY, intNorthwestX)], VALUE_3(data, intSample, intData, intPoint) * dblNorthwest);
					}
				}
			}

			if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
				if (dblError <= VALUE_4(zee, intSample, 0, intNortheastY, intNortheastX) + 1.0) {
					for (int intData = 0; intData < SIZE_1(data); intData += 1) {
						atomicAdd(&output[OFFSET_4(output, intSample, intData, intNortheastY, intNortheastX)], VALUE_3(data, intSample, intData, intPoint) * dblNortheast);
					}
				}
			}

			if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
				if (dblError <= VALUE_4(zee, intSample, 0, intSouthwestY, intSouthwestX) + 1.0) {
					for (int intData = 0; intData < SIZE_1(data); intData += 1) {
						atomicAdd(&output[OFFSET_4(output, intSample, intData, intSouthwestY, intSouthwestX)], VALUE_3(data, intSample, intData, intPoint) * dblSouthwest);
					}
				}
			}

			if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
				if (dblError <= VALUE_4(zee, intSample, 0, intSoutheastY, intSoutheastX) + 1.0) {
					for (int intData = 0; intData < SIZE_1(data); intData += 1) {
						atomicAdd(&output[OFFSET_4(output, intSample, intData, intSoutheastY, intSoutheastX)], VALUE_3(data, intSample, intData, intPoint) * dblSoutheast);
					}
				}
			}
		} }
	''', {
		'intWidth': intWidth,
		'intHeight': intHeight,
		'dblFocal': dblFocal,
		'dblBaseline': dblBaseline,
		'input': tensorInput,
		'data': tensorData,
		'zee': tensorZee,
		'output': tensorOutput
	}))(
		grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
		block=tuple([ 512, 1, 1 ]),
		args=[ n, tensorInput.data_ptr(), tensorData.data_ptr(), tensorZee.data_ptr(), tensorOutput.data_ptr() ],
		stream=Stream
	)

	return tensorOutput[:, :-1, :, :] / (tensorOutput[:, -1:, :, :] + 0.0000001), tensorOutput[:, -1:, :, :].detach().clone()
# end

def fill_disocclusion(tensorInput, tensorDepth):
	tensorOutput = tensorInput.clone()

	n = tensorInput.size(0) * tensorInput.size(2) * tensorInput.size(3)
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

			float dblShortest = 1000000.0;

			int intFillX = -1;
			int intFillY = -1;

			float dblDirectionX[] = { -1, 0, 1, 1,    -1, 1, 2,  2,    -2, -1, 1, 2, 3, 3,  3,  3 };
			float dblDirectionY[] = {  1, 1, 1, 0,     2, 2, 1, -1,     3,  3, 3, 3, 2, 1, -1, -2 };

			for (int intDirection = 0; intDirection < 16; intDirection += 1) {
				float dblNormalize = sqrt((dblDirectionX[intDirection] * dblDirectionX[intDirection]) + (dblDirectionY[intDirection] * dblDirectionY[intDirection]));

				dblDirectionX[intDirection] /= dblNormalize;
				dblDirectionY[intDirection] /= dblNormalize;
			}

			for (int intDirection = 0; intDirection < 16; intDirection += 1) {
				float dblFromX = intX; int intFromX = 0;
				float dblFromY = intY; int intFromY = 0;

				float dblToX = intX; int intToX = 0;
				float dblToY = intY; int intToY = 0;

				do {
					dblFromX -= dblDirectionX[intDirection]; intFromX = (int) (round(dblFromX));
					dblFromY -= dblDirectionY[intDirection]; intFromY = (int) (round(dblFromY));

					if ((intFromX < 0) | (intFromX >= SIZE_3(input))) { break; }
					if ((intFromY < 0) | (intFromY >= SIZE_2(input))) { break; }
					if (VALUE_4(depth, intSample, 0, intFromY, intFromX) > 0.0) { break; }
				} while (true);
				if ((intFromX < 0) | (intFromX >= SIZE_3(input))) { continue; }
				if ((intFromY < 0) | (intFromY >= SIZE_2(input))) { continue; }

				do {
					dblToX += dblDirectionX[intDirection]; intToX = (int) (round(dblToX));
					dblToY += dblDirectionY[intDirection]; intToY = (int) (round(dblToY));

					if ((intToX < 0) | (intToX >= SIZE_3(input))) { break; }
					if ((intToY < 0) | (intToY >= SIZE_2(input))) { break; }
					if (VALUE_4(depth, intSample, 0, intToY, intToX) > 0.0) { break; }
				} while (true);
				if ((intToX < 0) | (intToX >= SIZE_3(input))) { continue; }
				if ((intToY < 0) | (intToY >= SIZE_2(input))) { continue; }

				float dblDistance = sqrt(powf(intToX - intFromX, 2) + powf(intToY - intFromY, 2));

				if (dblShortest > dblDistance) {
					intFillX = intFromX;
					intFillY = intFromY;

					if (VALUE_4(depth, intSample, 0, intFromY, intFromX) < VALUE_4(depth, intSample, 0, intToY, intToX)) {
						intFillX = intToX;
						intFillY = intToY;
					}

					dblShortest = dblDistance;
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
		'input': tensorInput,
		'depth': tensorDepth,
		'output': tensorOutput
	}))(
		grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
		block=tuple([ 512, 1, 1 ]),
		args=[ n, tensorInput.data_ptr(), tensorDepth.data_ptr(), tensorOutput.data_ptr() ],
		stream=Stream
	)

	return tensorOutput
# end