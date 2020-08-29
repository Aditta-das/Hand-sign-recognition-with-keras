import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image

import pyttsx3
import datetime
import speech_recognition as sr
import os
import time
import re
import random

model = tf.keras.models.load_model("hand5.model")
capture = cv2.VideoCapture(0)

while True:
	ret, frame = capture.read()

	cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)

	crop_image = frame[100:300, 100:300]
	

	# Apply Gaussian Blur
	blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

	kernel = np.ones((5, 5))

	dialation = cv2.dilate(mask2, kernel, iterations=1)
	erosion = cv2.erode(dialation, kernel, iterations=1)

	filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
	ret, thresh = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY_INV)

	cv2.imshow("threshold", thresh)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	try:
		# Find contour with maximum area
		contour = max(contours, key=lambda x: cv2.contourArea(x))

		# Create bounding rectangle around the contour
		x, y, w, h = cv2.boundingRect(contour)
		cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

		# Find convex hull
		hull = cv2.convexHull(contour)

		# Draw contour
		drawing = np.zeros(crop_image.shape, np.uint8)
		cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
		cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

	except:
		pass


	roi_gray = thresh
	print(thresh)
	roi_gray = cv2.resize(roi_gray, (28, 28))
	img_pixels = image.img_to_array(roi_gray)
	img_pixels = np.expand_dims(img_pixels, axis=0)
	img_pixels = img_pixels / 255
	predict = model.predict(img_pixels)
	max_index = np.argmax(predict[0])
	labels = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y")
	predict_sign = labels[max_index]
	#print(predict_sign)
	cv2.putText(frame, predict_sign, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)


	cv2.imshow("Gesture", frame)
	all_image = np.hstack((drawing, crop_image))
	cv2.imshow('Contours', all_image)
	if cv2.waitKey(1) == ord("q"):
		break

capture.release()
cv2.destroyAllWindows()

