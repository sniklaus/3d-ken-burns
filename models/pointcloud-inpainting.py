class Basic(torch.nn.Module):
	def __init__(self, strType, intChannels):
		super(Basic, self).__init__()

		if strType == 'relu-conv-relu-conv':
			self.moduleMain = torch.nn.Sequential(
				torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		elif strType == 'conv-relu-conv':
			self.moduleMain = torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		# end

		if intChannels[0] == intChannels[2]:
			self.moduleShortcut = None

		elif intChannels[0] != intChannels[2]:
			self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

		# end
	# end

	def forward(self, tensorInput):
		if self.moduleShortcut is None:
			return self.moduleMain(tensorInput) + tensorInput

		elif self.moduleShortcut is not None:
			return self.moduleMain(tensorInput) + self.moduleShortcut(tensorInput)

		# end
	# end
# end

class Downsample(torch.nn.Module):
	def __init__(self, intChannels):
		super(Downsample, self).__init__()

		self.moduleMain = torch.nn.Sequential(
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tensorInput):
		return self.moduleMain(tensorInput)
	# end
# end

class Upsample(torch.nn.Module):
	def __init__(self, intChannels):
		super(Upsample, self).__init__()

		self.moduleMain = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tensorInput):
		return self.moduleMain(tensorInput)
	# end
# end

class Inpaint(torch.nn.Module):
	def __init__(self):
		super(Inpaint, self).__init__()

		self.moduleContext = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
			torch.nn.PReLU(num_parameters=64, init=0.25),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
			torch.nn.PReLU(num_parameters=64, init=0.25)
		)

		self.moduleInput = Basic('conv-relu-conv', [ 3 + 1 + 64 + 1, 32, 32 ])

		for intRow, intFeatures in [ (0, 32), (1, 64), (2, 128), (3, 256) ]:
			self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
		# end

		for intCol in [ 0, 1 ]:
			self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([ 32, 64, 64 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([ 64, 128, 128 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([ 128, 256, 256 ]))
		# end

		for intCol in [ 2, 3 ]:
			self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([ 256, 128, 128 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([ 128, 64, 64 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([ 64, 32, 32 ]))
		# end

		self.moduleImage = Basic('conv-relu-conv', [ 32, 32, 3 ])
		self.moduleDisparity = Basic('conv-relu-conv', [ 32, 32, 1 ])
	# end

	def forward(self, tensorImage, tensorDisparity, tensorShift):
		tensorDepth = (objectCommon['dblFocal'] * objectCommon['dblBaseline']) / (tensorDisparity + 0.0000001)
		tensorValid = (spatial_filter(tensorDisparity / tensorDisparity.max(), 'laplacian').abs() < 0.03).float()
		tensorPoints = depth_to_points(tensorDepth * tensorValid, objectCommon['dblFocal'])
		tensorPoints = tensorPoints.view(1, 3, -1)

		tensorMean = [ tensorImage.view(tensorImage.shape[0], -1).mean(1, True).view(tensorImage.shape[0], 1, 1, 1), tensorDisparity.view(tensorDisparity.shape[0], -1).mean(1, True).view(tensorDisparity.shape[0], 1, 1, 1) ]
		tensorStd = [ tensorImage.view(tensorImage.shape[0], -1).std(1, True).view(tensorImage.shape[0], 1, 1, 1), tensorDisparity.view(tensorDisparity.shape[0], -1).std(1, True).view(tensorDisparity.shape[0], 1, 1, 1) ]

		tensorImage = tensorImage.clone()
		tensorImage -= tensorMean[0]
		tensorImage /= tensorStd[0] + 0.0000001

		tensorDisparity = tensorDisparity.clone()
		tensorDisparity -= tensorMean[1]
		tensorDisparity /= tensorStd[1] + 0.0000001

		tensorContext = self.moduleContext(torch.cat([ tensorImage, tensorDisparity ], 1))

		tensorRender, tensorExisting = render_pointcloud(tensorPoints + tensorShift, torch.cat([ tensorImage, tensorDisparity, tensorContext ], 1).view(1, 68, -1), objectCommon['intWidth'], objectCommon['intHeight'], objectCommon['dblFocal'], objectCommon['dblBaseline'])

		tensorExisting = (tensorExisting > 0.0).float()
		tensorExisting = tensorExisting * spatial_filter(tensorExisting, 'median-5')
		tensorRender = tensorRender * tensorExisting.clone().detach()

		tensorColumn = [ None, None, None, None ]

		tensorColumn[0] = self.moduleInput(torch.cat([ tensorRender, tensorExisting ], 1))
		tensorColumn[1] = self._modules['0x0 - 1x0'](tensorColumn[0])
		tensorColumn[2] = self._modules['1x0 - 2x0'](tensorColumn[1])
		tensorColumn[3] = self._modules['2x0 - 3x0'](tensorColumn[2])

		intColumn = 1
		for intRow in range(len(tensorColumn)):
			tensorColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow])
			if intRow != 0:
				tensorColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow - 1])
			# end
		# end

		intColumn = 2
		for intRow in range(len(tensorColumn) -1, -1, -1):
			tensorColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow])
			if intRow != len(tensorColumn) - 1:
				tensorUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow + 1])

				if tensorUp.shape[2] != tensorColumn[intRow].shape[2]: tensorUp = torch.nn.functional.pad(input=tensorUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tensorUp.shape[3] != tensorColumn[intRow].shape[3]: tensorUp = torch.nn.functional.pad(input=tensorUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tensorColumn[intRow] += tensorUp
			# end
		# end

		intColumn = 3
		for intRow in range(len(tensorColumn) -1, -1, -1):
			tensorColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow])
			if intRow != len(tensorColumn) - 1:
				tensorUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow + 1])

				if tensorUp.shape[2] != tensorColumn[intRow].shape[2]: tensorUp = torch.nn.functional.pad(input=tensorUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tensorUp.shape[3] != tensorColumn[intRow].shape[3]: tensorUp = torch.nn.functional.pad(input=tensorUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tensorColumn[intRow] += tensorUp
			# end
		# end

		tensorImage = self.moduleImage(tensorColumn[0])
		tensorImage *= tensorStd[0] + 0.0000001
		tensorImage += tensorMean[0]

		tensorDisparity = self.moduleDisparity(tensorColumn[0])
		tensorDisparity *= tensorStd[1] + 0.0000001
		tensorDisparity += tensorMean[1]

		return {
			'tensorExisting': tensorExisting,
			'tensorImage': tensorImage.clamp(0.0, 1.0) if self.training == False else tensorImage,
			'tensorDisparity': torch.nn.functional.threshold(input=tensorDisparity, threshold=0.0, value=0.0)
		}
	# end
# end

moduleInpaint = Inpaint().cuda().eval(); moduleInpaint.load_state_dict(torch.load('./models/pointcloud-inpainting.pytorch'))

def pointcloud_inpainting(tensorImage, tensorDisparity, tensorShift):
	return moduleInpaint(tensorImage, tensorDisparity, tensorShift)
# end