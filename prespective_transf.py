import pickle
import cv2
import numpy as np
import glob

#Import the gradient and color scheme updated Images
images = glob.glob('./output_images/CombinedColorGrad_test*.jpg')

#Iterate through the images and create the prespective transform
for idx, fname in enumerate(images):
	#Pull in the Undistorted images for processing
	img = cv2.imread(fname)
	img_cpy = np.copy(img)

	#Print the currently processing image
	print ('Currently processing image', fname)
	#Define the points for region of interest to perform perspective transform
	img_height = img.shape[0]
	img_width = img.shape[1]

	mid_tpz_offset = 90
	botm_tpz_offset = 100 
	roi_tpz_pt1 = [(img_width*0.5 - mid_tpz_offset), img_height*0.63]
	roi_tpz_pt2 = [(img_width*0.5 + mid_tpz_offset), img_height*0.63]
	roi_tpz_pt3 = [botm_tpz_offset, img_height*0.95]
	roi_tpz_pt4 = [(img_width - botm_tpz_offset), img_height*0.95]
	
	#Assigning the parameters to draw a trpaezium on the image using the defined points of region of interest
	roi_tpz_pts = np.array([roi_tpz_pt1,roi_tpz_pt2,roi_tpz_pt4,roi_tpz_pt3])
	color_tpz = (0,0,255)
	thickness_tpz = 10
	img_tpz = cv2.polylines(img_cpy,np.int32([roi_tpz_pts]),True,color_tpz,thickness_tpz)
	# result = img_tpz

	# print_name = './test_images/ROI_trapezium_test'+str(idx+1)+'.jpg'
	# cv2.imwrite(print_name, result)
	
	#Defining the source points to perform perspective transorm same as the ROI trapezium
	src = np.float32([roi_tpz_pt1,roi_tpz_pt2,roi_tpz_pt3,roi_tpz_pt4])
	#Defining the points for the desstination of prespective transform
	lane_width = img.shape[0]*0.28
	dst_p1 = [lane_width,0]
	dst_p2 = [(img_width-lane_width),0]
	dst_p3 = [lane_width,img_height]
	dst_p4 = [(img_width-lane_width),img_height]
	#Assigning the points to the destination points array
	dst = np.float32([dst_p1,dst_p2,dst_p3,dst_p4])

	#Performing the prespective transform
	M = cv2.getPerspectiveTransform(src, dst)
	#Performing the inverse prespective transform
	Minv = cv2.getPerspectiveTransform(dst, src)
	#Creating the wrapped image
	img_size = (img.shape[1],img.shape[0])
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

	result = warped 

	print_name = './output_images/Warped_test'+str(idx+1)+'.jpg'
	cv2.imwrite(print_name, result)

#Storing the source and destination result for performing unwrap function
warp_pickle = {}
warp_pickle["src"] = src
warp_pickle["dst"] = dst
pickle.dump(warp_pickle, open ("./warp_pickle.p","wb"))	