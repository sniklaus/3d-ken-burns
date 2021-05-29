netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()

def disparity_adjustment(tenImage, tenDisparity):
	assert(tenImage.shape[0] == 1)
	assert(tenDisparity.shape[0] == 1)

	boolUsed = {}
	tenMasks = []

	objPredictions = netMaskrcnn([ tenImage[ 0, [ 2, 0, 1 ], :, : ] ])[0]

	for intMask in range(objPredictions['masks'].shape[0]):
		if intMask in boolUsed:
			continue

		elif objPredictions['scores'][intMask].item() < 0.7:
			continue

		elif objPredictions['labels'][intMask].item() not in [ 1, 3, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 ]:
			continue

		# end

		boolUsed[intMask] = True
		tenMask = (objPredictions['masks'][(intMask + 0):(intMask + 1), :, :, :] > 0.5).float()

		if tenMask.sum().item() < 64:
			continue
		# end

		for intMerge in range(objPredictions['masks'].shape[0]):
			if intMerge in boolUsed:
				continue

			elif objPredictions['scores'][intMerge].item() < 0.7:
				continue

			elif objPredictions['labels'][intMerge].item() not in [ 2, 4, 27, 28, 31, 32, 33 ]:
				continue

			# end

			tenMerge = (objPredictions['masks'][(intMerge + 0):(intMerge + 1), :, :, :] > 0.5).float()

			if ((tenMask + tenMerge) > 1.0).sum().item() < 0.03 * tenMerge.sum().item():
				continue
			# end

			boolUsed[intMerge] = True
			tenMask = (tenMask + tenMerge).clip(0.0, 1.0)
		# end

		tenMasks.append(tenMask)
	# end

	tenAdjusted = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False)

	for tenAdjust in tenMasks:
		tenPlane = tenAdjusted * tenAdjust

		tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()
		tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()

		intLeft = (tenPlane.sum(2, True) > 0.0).flatten().nonzero()[0].item()
		intTop = (tenPlane.sum(3, True) > 0.0).flatten().nonzero()[0].item()
		intRight = (tenPlane.sum(2, True) > 0.0).flatten().nonzero()[-1].item()
		intBottom = (tenPlane.sum(3, True) > 0.0).flatten().nonzero()[-1].item()

		tenAdjusted = ((1.0 - tenAdjust) * tenAdjusted) + (tenAdjust * tenPlane[:, :, int(round(intTop + (0.97 * (intBottom - intTop)))):, :].max())
	# end

	return torch.nn.functional.interpolate(input=tenAdjusted, size=(tenDisparity.shape[2], tenDisparity.shape[3]), mode='bilinear', align_corners=False)
# end