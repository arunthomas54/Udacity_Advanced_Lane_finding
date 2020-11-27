import pickle
import cv2
import numpy as np
import glob

dist_pickle = pickle.load( open ("camera_cal/calibration_pickle.p","rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

#Capture all the test images to perform the undistortion
images = glob.glob('./test_images/test*.jpg')

#Iterate through the test images and undistort the images
for idx, fname in enumerate(images):
	#Pull in the test images for processing
	img = cv2.imread(fname)
	#Print the currently processing image
	print ('Currently processing image', fname)
	#undistort the images
	undist_img = cv2.undistort(img,mtx,dist,None,mtx)

	print_name = './output_images/Undistorted_test'+str(idx+1)+'.jpg'

	cv2.imwrite(print_name, undist_img)
