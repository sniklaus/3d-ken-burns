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

class Refine(torch.nn.Module):
	def __init__(self):
		super(Refine, self).__init__()

		self.moduleImageOne = Basic('conv-relu-conv', [ 3, 24, 24 ])
		self.moduleImageTwo = Downsample([ 24, 48, 48 ])
		self.moduleImageThr = Downsample([ 48, 96, 96 ])

		self.moduleDisparityOne = Basic('conv-relu-conv', [ 1, 96, 96 ])
		self.moduleDisparityTwo = Upsample([ 192, 96, 96 ])
		self.moduleDisparityThr = Upsample([ 144, 48, 48 ])
		self.moduleDisparityFou = Basic('conv-relu-conv', [ 72, 24, 24 ])

		self.moduleRefine = Basic('conv-relu-conv', [ 24, 24, 1 ])
	# end

	def forward(self, tensorImage, tensorDisparity):
		tensorMean = [ tensorImage.view(tensorImage.shape[0], -1).mean(1, True).view(tensorImage.shape[0], 1, 1, 1), tensorDisparity.view(tensorDisparity.shape[0], -1).mean(1, True).view(tensorDisparity.shape[0], 1, 1, 1) ]
		tensorStd = [ tensorImage.view(tensorImage.shape[0], -1).std(1, True).view(tensorImage.shape[0], 1, 1, 1), tensorDisparity.view(tensorDisparity.shape[0], -1).std(1, True).view(tensorDisparity.shape[0], 1, 1, 1) ]

		tensorImage = tensorImage.clone()
		tensorImage -= tensorMean[0]
		tensorImage /= tensorStd[0] + 0.0000001

		tensorDisparity = tensorDisparity.clone()
		tensorDisparity -= tensorMean[1]
		tensorDisparity /= tensorStd[1] + 0.0000001

		tensorImageOne = self.moduleImageOne(tensorImage)
		tensorImageTwo = self.moduleImageTwo(tensorImageOne)
		tensorImageThr = self.moduleImageThr(tensorImageTwo)

		tensorUpsample = self.moduleDisparityOne(tensorDisparity)
		if tensorUpsample.shape != tensorImageThr.shape: tensorUpsample = torch.nn.functional.interpolate(input=tensorUpsample, size=(tensorImageThr.shape[2], tensorImageThr.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tensorUpsample = self.moduleDisparityTwo(torch.cat([ tensorImageThr, tensorUpsample ], 1)); tensorImageThr = None
		if tensorUpsample.shape != tensorImageTwo.shape: tensorUpsample = torch.nn.functional.interpolate(input=tensorUpsample, size=(tensorImageTwo.shape[2], tensorImageTwo.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tensorUpsample = self.moduleDisparityThr(torch.cat([ tensorImageTwo, tensorUpsample ], 1)); tensorImageTwo = None
		if tensorUpsample.shape != tensorImageOne.shape: tensorUpsample = torch.nn.functional.interpolate(input=tensorUpsample, size=(tensorImageOne.shape[2], tensorImageOne.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tensorUpsample = self.moduleDisparityFou(torch.cat([ tensorImageOne, tensorUpsample ], 1)); tensorImageOne = None

		tensorRefine = self.moduleRefine(tensorUpsample)
		tensorRefine *= tensorStd[1] + 0.0000001
		tensorRefine += tensorMean[1]

		return torch.nn.functional.threshold(input=tensorRefine, threshold=0.0, value=0.0)
	# end
# end

moduleRefine = Refine().cuda().eval(); moduleRefine.load_state_dict(torch.load('./models/disparity-refinement.pytorch'))

def disparity_refinement(tensorImage, tensorDisparity):
	return moduleRefine(tensorImage, tensorDisparity)
# end