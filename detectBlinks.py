from scipy.spatial import distance
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from ear import calculateEAR
import constants
import numpy as np
import imutils
import cv2
import dlib
import time

frameCount = 0
totBlinks = 0

faceDetector = dlib.get_frontal_face_detector()
landmarksPredictor = dlib.shape_predictor(shapePredictorPath)

leftEyeIdx = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
rightEyeIdx = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

if FILE_VIDEO_STREAM:
	vs = FileVideoStream(videoFilePath).start()
else:
	vs = VideoStream().start()
time.sleep(2.0)

while True:
	if FILE_VIDEO_STREAM and not vs.more():
		break
	frame = vs.read()
	frame = imutils.resize(frame, width = 500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = faceDetector(gray, 0)
	for face in faces:
		landmarks = landmarksPredictor(gray, face)
		landmarks = face_utils.shape_to_np(landmarks)
		
		leftEye = landmarks[leftEyeIdx[0]:leftEyeIdx[1]]
		rightEye = landmarks[rightEyeIdx[0]:rightEyeIdx[1]]
		leftEAR = calculateEAR(leftEye)
		rightEAR = calculateEAR(rightEye)
		ear = (leftEAR + rightEAR)/2.0
		if ear < EAR_THRESHOLD:
			frameCount += 1
		else:
			totBlinks = 1 if frameCount >= EAR_CONSEC_FRAMES
			frameCount = 0
				
			
