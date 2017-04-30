from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from ear import calculateEAR
from constants import *
import numpy as np
import imutils
import cv2
import dlib
import time

frameCount = 0
blinkCount = 0

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
	#frame = imutils.rotate(frame, angle=-90)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = faceDetector(gray, 0)
	for (i, face) in enumerate(faces):
		(x, y, w, h) = face_utils.rect_to_bb(face)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, '#{}'.format(i+1), (x, y-10), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		landmarks = landmarksPredictor(gray, face)
		landmarks = face_utils.shape_to_np(landmarks)
		
		leftEye = landmarks[leftEyeIdx[0]:leftEyeIdx[1]]
		rightEye = landmarks[rightEyeIdx[0]:rightEyeIdx[1]]
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)		
		cv2.drawContours(frame, [leftEyeHull, rightEyeHull], -1, (0, 0, 255), 1)
		
		leftEAR = calculateEAR(leftEye)
		rightEAR = calculateEAR(rightEye)
		ear = (leftEAR + rightEAR)/2.0
		if ear < EAR_THRESHOLD:
			frameCount += 1
		else:
			if frameCount >= EAR_CONSEC_FRAMES:
				blinkCount += 1 
			frameCount = 0

		cv2.putText(frame, "Blinks: {}".format(blinkCount), 
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(frame, "EAR for #{}: {:05.3f}".format(i+1, ear), (350, 30+20*i), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	
	cv2.imshow("Video output", frame)
	key = cv2.waitKey(0) & 0xFF
	if key == ord('q'):
		break

cv2.destroyAllWindows()
vs.stop() 
		

				
			
