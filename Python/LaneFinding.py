'''
@file	LaneFinding.py
@author	won.seok.django@gmail.com
@brief
'''
#####################
## Import Packages ##
#####################

import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as mpplot

##############################
## Pipeline Stage Functions ##
##############################

#  tidy up image for better processing and return tidy image
#  @param    _img    image to be tidy
#  @param    _gKSize Gaussian blur kernel size
#  @param    _lo     low threshold for Canny edge detection
#  @param    _hi     high threshold for Canny edge detection
#  @param    _dKSize dilation kernel size
#  @return   tidy image of _img
def LaneDetectionStageTidyUp(_img, _gKSize, _lo, _hi, _dKSize):
	_img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
	_img = cv2.GaussianBlur(_img, (_gKSize, _gKSize), 0)
	_img = cv2.Canny(_img, _lo, _hi)
	_img = cv2.dilate(_img, cv2.getStructuringElement(cv2.MORPH_DILATE, (_dKSize, _dKSize)))
	return _img

#  return clipped image. out of polygon region will be eliminated
#  @param    _img    image to be clipped
#  @param    _vtxs   vertices that form a polygon
#  @return   clipped image of _img that out of _vtx-polygon region is blacked out
def LaneDetectionStageClip(_img, _vtxs):
	mask = np.zeros_like(_img)
	maskColor = 255
	cv2.fillPoly(mask, _vtxs, maskColor)
	return cv2.bitwise_and(_img, mask)

#  return detected lane image
#  @param    _img              image to be detected
#  @param    _lh               low, high value of height
#  @param    _minIntersects    the minimal number of intersect in a single Hough-space grid
#  @param    _minPixels        the minimal number of pixels which compose a line
#  @param    _maxGap           the maximal distance between two pixels in a line
#  @return   detected lane image of _img
def LaneDetectionStageHoughLineDetect(_img, _lh, _minIntersects, _minPixels, _maxGap):
	laneImg = np.zeros((_img.shape[0], _img.shape[1], 3), np.uint8)

	lines = cv2.HoughLinesP(_img, 1, np.pi / 180.0, _minIntersects, _minPixels, _maxGap)
	if lines is None or len(lines) == 0:
		return laneImg

	w = _img.shape[1]
	h = _img.shape[0]

	linesL = []
	linesR = []
	for line in lines:
		for x1, y1, x2, y2 in line:
			if x1 < w / 2 and x2 < w / 2:
				linesL.append([x1, y1, x2, y2])
			if x1 > w / 2 and x2 > w / 2:
				linesR.append([x1, y1, x2, y2])

	linesL = list(filter(lambda l : l[0] - l[2] != 0, linesL))
	linesL = list(filter(lambda l : float(l[1] - l[3]) / (l[0] - l[2]) < -0.2, linesL))
	linesR = list(filter(lambda l : l[0] - l[2] != 0, linesR))
	linesR = list(filter(lambda l : float(l[1] - l[3]) / (l[0] - l[2]) >  0.2, linesR))

	xL = []
	yL = []
	for line in linesL:
		xL = xL + [line[0], line[2]]
		yL = yL + [line[1], line[3]]

	xR = []
	yR = []
	for line in linesR:
		xR = xR + [line[0], line[2]]
		yR = yR + [line[1], line[3]]

	if xL is None or xR is None or len(xL) == 0 or len(xR) == 0:
		return laneImg

	xLArr = np.array(xL)
	yLArr = np.array(yL)
	xRArr = np.array(xR)
	yRArr = np.array(yR)

	AL = np.vstack([xLArr, np.ones(len(xLArr))]).T
	AR = np.vstack([xRArr, np.ones(len(xRArr))]).T

	mL, cL = np.linalg.lstsq(AL, yLArr)[0]
	mR, cR = np.linalg.lstsq(AR, yRArr)[0]

	tL = (int((_lh[0] - cL) / mL), int(_lh[0]))
	bL = (int((_lh[1] - cL) / mL), int(_lh[1]))
	tR = (int((_lh[0] - cR) / mR), int(_lh[0]))
	bR = (int((_lh[1] - cR) / mR), int(_lh[1]))
	tC = (int((tL[0] + tR[0]) / 2), int((tL[1] + tR[1]) / 2))
	bC = (int((bL[0] + bR[0]) / 2), int((bL[1] + bR[1]) / 2))

	cv2.line(laneImg, tL, bL, (0, 0  , 255), 10)
	cv2.line(laneImg, tR, bR, (0, 0  , 255), 10)
	cv2.line(laneImg, tC, bC, (0, 255, 255), 10)

	return laneImg

#####################
## start main loop ##
#####################

def showVideo():
	try:
		print('initializing camera')
		cap = cv2.VideoCapture(1)
	except:
		print('initializing camera failed')
		return

	while True:
		ret, frame = cap.read()
		if not ret:
			print('reading frame failed')
			break

		lane = np.copy(frame)
		lane = LaneDetectionStageTidyUp(lane, 5, 150, 250, 3)
		lu = [0.375 * lane.shape[1], 0.650 * lane.shape[0]]
		ll = [0.075 * lane.shape[1], 1.000 * lane.shape[0]]
		rl = [0.925 * lane.shape[1], 1.000 * lane.shape[0]]
		ru = [0.625 * lane.shape[1], 0.650 * lane.shape[0]]
		lane = LaneDetectionStageClip(lane, np.array([[lu, ll, rl, ru]], dtype=np.int32))
		lane = LaneDetectionStageHoughLineDetect(lane, [lu[1], ll[1]], 50, 5, 5)
		frame = cv2.addWeighted(frame, 0.8, lane, 1, 0)
		cv2.imshow('video', frame)

		keyPress = cv2.waitKey(1) & 0xFF
		if keyPress == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

showVideo()
