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

class Semantics(torch.nn.Module):
	def __init__(self):
		super(Semantics, self).__init__()

		moduleVgg = torchvision.models.vgg19_bn(pretrained=True).features.eval()

		self.moduleVgg = torch.nn.Sequential(
			moduleVgg[0:3],
			moduleVgg[3:6],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[7:10],
			moduleVgg[10:13],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[14:17],
			moduleVgg[17:20],
			moduleVgg[20:23],
			moduleVgg[23:26],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[27:30],
			moduleVgg[30:33],
			moduleVgg[33:36],
			moduleVgg[36:39],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
		)
	# end

	def forward(self, tensorInput):
		tensorPreprocessed = tensorInput[:, [ 2, 1, 0 ], :, :]

		tensorPreprocessed[:, 0, :, :] = (tensorPreprocessed[:, 0, :, :] - 0.485) / 0.229
		tensorPreprocessed[:, 1, :, :] = (tensorPreprocessed[:, 1, :, :] - 0.456) / 0.224
		tensorPreprocessed[:, 2, :, :] = (tensorPreprocessed[:, 2, :, :] - 0.406) / 0.225

		return self.moduleVgg(tensorPreprocessed)
	# end
# end

class Disparity(torch.nn.Module):
	def __init__(self):
		super(Disparity, self).__init__()

		self.moduleImage = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
		self.moduleSemantics = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

		for intRow, intFeatures in [ (0, 32), (1, 48), (2, 64), (3, 512), (4, 512), (5, 512) ]:
			self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
		# end

		for intCol in [ 0, 1 ]:
			self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([ 32, 48, 48 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([ 48, 64, 64 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([ 64, 512, 512 ]))
			self.add_module('3x' + str(intCol) + ' - ' + '4x' + str(intCol), Downsample([ 512, 512, 512 ]))
			self.add_module('4x' + str(intCol) + ' - ' + '5x' + str(intCol), Downsample([ 512, 512, 512 ]))
		# end

		for intCol in [ 2, 3 ]:
			self.add_module('5x' + str(intCol) + ' - ' + '4x' + str(intCol), Upsample([ 512, 512, 512 ]))
			self.add_module('4x' + str(intCol) + ' - ' + '3x' + str(intCol), Upsample([ 512, 512, 512 ]))
			self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([ 512, 64, 64 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([ 64, 48, 48 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([ 48, 32, 32 ]))
		# end

		self.moduleDisparity = Basic('conv-relu-conv', [ 32, 32, 1 ])
	# end

	def forward(self, tensorImage, tensorSemantics):
		tensorColumn = [ None, None, None, None, None, None ]

		tensorColumn[0] = self.moduleImage(tensorImage)
		tensorColumn[1] = self._modules['0x0 - 1x0'](tensorColumn[0])
		tensorColumn[2] = self._modules['1x0 - 2x0'](tensorColumn[1])
		tensorColumn[3] = self._modules['2x0 - 3x0'](tensorColumn[2]) + self.moduleSemantics(tensorSemantics)
		tensorColumn[4] = self._modules['3x0 - 4x0'](tensorColumn[3])
		tensorColumn[5] = self._modules['4x0 - 5x0'](tensorColumn[4])

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

				if tensorUp.size(2) != tensorColumn[intRow].size(2): tensorUp = torch.nn.functional.pad(input=tensorUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tensorUp.size(3) != tensorColumn[intRow].size(3): tensorUp = torch.nn.functional.pad(input=tensorUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tensorColumn[intRow] += tensorUp
			# end
		# end

		intColumn = 3
		for intRow in range(len(tensorColumn) -1, -1, -1):
			tensorColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow])
			if intRow != len(tensorColumn) - 1:
				tensorUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow + 1])

				if tensorUp.size(2) != tensorColumn[intRow].size(2): tensorUp = torch.nn.functional.pad(input=tensorUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tensorUp.size(3) != tensorColumn[intRow].size(3): tensorUp = torch.nn.functional.pad(input=tensorUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tensorColumn[intRow] += tensorUp
			# end
		# end

		return torch.nn.functional.threshold(input=self.moduleDisparity(tensorColumn[0]), threshold=0.0, value=0.0)
	# end
# end

moduleSemantics = Semantics().cuda().eval()
moduleDisparity = Disparity().cuda().eval(); moduleDisparity.load_state_dict(torch.load('./models/disparity-estimation.pytorch'))

def disparity_estimation(tensorImage):
	intWidth = tensorImage.size(3)
	intHeight = tensorImage.size(2)

	dblRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(512 * dblRatio), 512)
	intHeight = min(int(512 / dblRatio), 512)

	tensorImage = torch.nn.functional.interpolate(input=tensorImage, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	return moduleDisparity(tensorImage, moduleSemantics(tensorImage))
# end