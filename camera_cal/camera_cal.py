import pickle
import cv2
import numpy as np
import glob

#Arryas to store object points and image points from all the images
objpoints = [] #3D points in real world space
imgpoints = [] #2D points in image space

#Prepare the object points for the 9*6 checkered board like (0,0,0) , (1,0,0)... and last point (8,5,0)
objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#Capture all the images in the images folder and make a list of calibration images
images = glob.glob('./calibration*.jpg')

#Iterate through the images and search for chessboard corners
for idx, fname in enumerate(images):
	#Pull in each images for processing
	img = cv2.imread(fname)

	#Convert to grayscale the pulled in image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

	#If corners are found, add to the object points and image points
	if ret == True:
		print ('Currently processing image', fname)
		imgpoints.append(corners)
		objpoints.append(objp)

		#Draw and display the corners
		cv2.drawChessboardCorners(img, (9,6), corners, ret)
		print_name = 'corners_found'+str(idx)+'.jpg'
		cv2.imwrite(print_name, img)

#Calculate the image size for reference
img = cv2.imread('./calibration1.jpg')
img_size = (img.shape[1],img.shape[0])

#Perform camera calibration on the saved objectpoints and imagepoints
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

#Storing the camera calibration result for future use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open ("./calibration_pickle.p","wb"))
