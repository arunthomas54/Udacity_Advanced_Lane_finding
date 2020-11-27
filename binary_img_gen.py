import pickle
import cv2
import numpy as np
import glob

#Function to convert the distorted image to binary image using the absolute gradient method
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
	#Convert the image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Calculate the derivative in x or y given orientation
	if orient == 'x':
		#Derivative in x direction
		sobelx = cv2.Sobel(gray,cv2.CV_64F, 1, 0)
		#Absolute value of x derivative
		abs_sobel = np.absolute(sobelx)
	elif orient =='y':
		#Derivative in y direction
		sobely = cv2.Sobel(gray,cv2.CV_64F, 0, 1)
		#Absolute value of y derivative
		abs_sobel = np.absolute(sobely)
	#Scale to 8-bit (0 - 255) then convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	#Create a zeros matrix of simailar size as 8-bit converted
	grad_binary = np.zeros_like(scaled_sobel)
	#Mask of 1's where the scaled gradient is less than thresh_max and greater than thresh_min 
	grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel<= thresh_max)] = 1
	return grad_binary 

#Function to convert the distorted image to binary image using the magnitude of the gradient
def mag_thresh(img, sobel_kernel = 3, thresh_min = 0, thresh_max = 255):
	#Convert the image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Derivative in x direction
	sobelx = cv2.Sobel(gray,cv2.CV_64F, 1, 0, sobel_kernel)
	#Derivative in y direction
	sobely = cv2.Sobel(gray,cv2.CV_64F, 0, 1, sobel_kernel)
	#Magnitude of gradient
	gradmag =  np.sqrt(sobelx**2 + sobely**2)
	#Scale to 8-bit (0 - 255) then convert to type = np.uint8
	scaled_gradmag = np.uint8(255*gradmag/np.max(gradmag))
	#Create a zeros matrix of simailar size as 8-bit converted
	mag_binary = np.zeros_like(scaled_gradmag)
	#Mask of 1's where the scaled gradient is less than thresh_max and greater than thresh_min 
	mag_binary[(scaled_gradmag >= thresh_min) & (scaled_gradmag<= thresh_max)] = 1
	return mag_binary

#Function to convert the distorted image to binary image using the direction of the gradient
def dir_threshold(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):
	#Convert the image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Derivative in x direction
	sobelx = cv2.Sobel(gray,cv2.CV_64F, 1, 0, sobel_kernel)
	#Absolute value of x derivative
	abs_sobel_x = np.absolute(sobelx)
	#Derivative in y direction
	sobely = cv2.Sobel(gray,cv2.CV_64F, 0, 1, sobel_kernel)
	#Absolute value of x derivative
	abs_sobel_y = np.absolute(sobely)
	#Direction of gradient
	graddir =  np.arctan2(abs_sobel_y, abs_sobel_x)
	#Create a zeros matrix of simailar size as direction gradient
	dir_binary = np.zeros_like(graddir)
	#Mask of 1's where the scaled gradient is less than thresh_max and greater than thresh_min 
	dir_binary[(graddir >= thresh_min) & (graddir<= thresh_max)] = 1
	return dir_binary

#Function to perform color threshold on the image
def color_threshold(img, sthresh_min=0, sthresh_max=255, vthresh_min=0, vthresh_max=255  ):
	#Converting the image to HLS format
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	#Selecting the s-channel
	s_channel = hls[:,:,2]
	#Create a zeros matrix of simailar size as s-channel
	s_channel_bin = np.zeros_like(s_channel)
	#Mask of 1's where the s-channel is less than thresh_max and greater than thresh_min 
	s_channel_bin[(s_channel >= sthresh_min) & (s_channel<= sthresh_max)] = 1

	#Converting the image to HSV format
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	#Selecting the v-channel
	v_channel = hsv[:,:,2]
	#Create a zeros matrix of simailar size as s-channel
	v_channel_bin = np.zeros_like(v_channel)
	#Mask of 1's where the s-channel is less than thresh_max and greater than thresh_min 
	v_channel_bin[(v_channel >= vthresh_min) & (v_channel<= vthresh_max)] = 1

	#Comining the s-channel and v-channel output binary
	combined_sv_bin = np.zeros_like(s_channel)
	#Mask of 1's where the s-channel and v-channel
	combined_sv_bin[(s_channel_bin == 1) & (v_channel_bin == 1)] = 1
	return combined_sv_bin

#Import the Undistorted Images
images = glob.glob('./output_images/Undistorted_test*.jpg')

#Iterate through the Undistorted images and create the binary images
for idx, fname in enumerate(images):
	#Pull in the Undistorted images for processing
	img = cv2.imread(fname)
	#Print the currently processing image
	print ('Currently processing image', fname)
	#Process the image and create the bianry gradients
	Pre_img = np.zeros_like(img[:,:,0])
	#Gradient in x-direction
	x_grad = abs_sobel_thresh(img, orient ='x', thresh_min = 20, thresh_max = 100 )
	y_grad = abs_sobel_thresh(img, orient ='y', thresh_min = 20, thresh_max = 100 )
	mag_grad = mag_thresh(img, sobel_kernel = 3, thresh_min =30, thresh_max = 100)
	dir_grad = dir_threshold(img, sobel_kernel=3, thresh_min=0.7, thresh_max=1.3)
	color_grad = color_threshold(img,sthresh_min=100, sthresh_max=255, vthresh_min=50, vthresh_max=255)
	Pre_img[(((x_grad == 1) & (y_grad == 1)) | ((mag_grad == 1) & (dir_grad == 1)))| (color_grad ==1)] = 255

	result = Pre_img

	print_name = './output_images/CombinedColorGrad_test'+str(idx+1)+'.jpg'

	cv2.imwrite(print_name, result)