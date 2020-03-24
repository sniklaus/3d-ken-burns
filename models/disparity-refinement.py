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

	def forward(self, tenInput):
		if self.moduleShortcut is None:
			return self.moduleMain(tenInput) + tenInput

		elif self.moduleShortcut is not None:
			return self.moduleMain(tenInput) + self.moduleShortcut(tenInput)

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

	def forward(self, tenInput):
		return self.moduleMain(tenInput)
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

	def forward(self, tenInput):
		return self.moduleMain(tenInput)
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

	def forward(self, tenImage, tenDisparity):
		tenMean = [ tenImage.view(tenImage.shape[0], -1).mean(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]
		tenStd = [ tenImage.view(tenImage.shape[0], -1).std(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]

		tenImage = tenImage.clone()
		tenImage -= tenMean[0]
		tenImage /= tenStd[0] + 0.0000001

		tenDisparity = tenDisparity.clone()
		tenDisparity -= tenMean[1]
		tenDisparity /= tenStd[1] + 0.0000001

		tenImageOne = self.moduleImageOne(tenImage)
		tenImageTwo = self.moduleImageTwo(tenImageOne)
		tenImageThr = self.moduleImageThr(tenImageTwo)

		tenUpsample = self.moduleDisparityOne(tenDisparity)
		if tenUpsample.shape != tenImageThr.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageThr.shape[2], tenImageThr.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.moduleDisparityTwo(torch.cat([ tenImageThr, tenUpsample ], 1)); tenImageThr = None
		if tenUpsample.shape != tenImageTwo.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageTwo.shape[2], tenImageTwo.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.moduleDisparityThr(torch.cat([ tenImageTwo, tenUpsample ], 1)); tenImageTwo = None
		if tenUpsample.shape != tenImageOne.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageOne.shape[2], tenImageOne.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.moduleDisparityFou(torch.cat([ tenImageOne, tenUpsample ], 1)); tenImageOne = None

		tenRefine = self.moduleRefine(tenUpsample)
		tenRefine *= tenStd[1] + 0.0000001
		tenRefine += tenMean[1]

		return torch.nn.functional.threshold(input=tenRefine, threshold=0.0, value=0.0)
	# end
# end

moduleRefine = Refine().cuda().eval(); moduleRefine.load_state_dict(torch.load('./models/disparity-refinement.pytorch'))

def disparity_refinement(tenImage, tenDisparity):
	return moduleRefine(tenImage, tenDisparity)
# end