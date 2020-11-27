import pickle
import cv2
import numpy as np
import glob

warp_pickle = pickle.load( open ("./warp_pickle.p","rb"))
src = warp_pickle["src"]
dst = warp_pickle["dst"]

#Import the warped image with lane dtected
images = glob.glob('./output_images/Lane_Detec_warp_test*.jpg')

#Iterate through the images and create the prespective transform
for idx, fname in enumerate(images):
	#Pull in the Undistorted images for processing
	img = cv2.imread(fname)
	img_cpy = np.copy(img)
	#Print the currently processing image
	print ('Currently processing image', fname)

	#Performing the inverse prespective transform
	Minv = cv2.getPerspectiveTransform(dst, src)
	img_size = (img.shape[1],img.shape[0])
	#Create the unwarped image
	unwarped_image = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

	print_name = './output_images/Lane_Detec_Unwarpped_test'+str(idx+1)+'.jpg'
	cv2.imwrite(print_name, unwarped_image)