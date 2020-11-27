import pickle
import cv2
import numpy as np
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

dist_pickle = pickle.load( open ("camera_cal/calibration_pickle.p","rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

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

#Function to detect the lane line pixels
def lane_detct_sliding_windows(img):
	#Process only the lines on the bottom half because lane lines are vertical nearest to car
	below_half = img[img.shape[0]//2:,:]
	#Detect peaks in the bottom half of image histogram
	histogram = np.sum(below_half, axis = 0)
	# plt.plot(histogram)
	# plt.show()
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#Create an output image based on the below half histogram
	img_out = np.copy(img)
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#Detect the left and right peaks in the histogram
	mid_hist = np.int(histogram.shape[0]//2)
	#Starting point of the left side
	left_start = np.argmax(histogram[:mid_hist]) // 3
	#print(left_start)
	#Starting point of the right side
	right_start = np.argmax(histogram[mid_hist:]) // 3 + mid_hist
	#print(right_start)

	#Definging the hyper parameters of sliding windows
	#Number of sliding windows
	nwindows = 9
	#Width of the windows margin
	margin_width = 100
	#Minimum number of pixels required to recenter window
	pix_min = 50
	#Height of window based on image height and number of windows
	img_height = img.shape[0]
	window_height = np.int(img_height // nwindows)
	
	#Detect the non-zero pixels in x and y direction
	nonzero_img = img.nonzero()
	nonzero_img_x = np.array(nonzero_img[1])
	nonzero_img_y = np.array(nonzero_img[0])

	#Current positions of window to be updated for each iteration
	left_x_current = left_start
	right_x_current = right_start

	#Array initialization for left and right pixel indices
	left_lane_indcs = []
	right_lane_indcs = []

	#Iterate through each windows
	for window in range(nwindows):
		#Detect the boundaries for x and y in both left and right directions
		win_y_low = img_height - (window+1)*window_height
		win_y_high = img_height - window*window_height
		win_x_left_low = left_x_current - margin_width
		win_x_left_high = left_x_current + margin_width
		win_x_right_low = right_x_current - margin_width
		win_x_right_high = right_x_current + margin_width

		#Drwaing slinding windows for visualization
		color_window = (0,255,0)
		thickness = 2
		start_point_left = (win_x_left_low,win_y_low)
		end_point_left = (win_x_left_high,win_y_high)
		start_point_right = (win_x_right_low,win_y_low)
		end_point_right = (win_x_right_high,win_y_high)
		#Left side window
		cv2.rectangle(img_out,start_point_left,end_point_left,color_window,thickness)
		#Right side window
		cv2.rectangle(img_out,start_point_right,end_point_right,color_window,thickness)

		#Detection of nonzero pixels in x and y inside the window
		non_zero_window_y = (nonzero_img_y >= win_y_low) & (nonzero_img_y < win_y_high)
		non_zero_window_x_left = (nonzero_img_x >= win_x_left_low) & (nonzero_img_x < win_x_left_high)
		non_zero_window_x_right = (nonzero_img_x >= win_x_right_low) & (nonzero_img_x < win_x_right_high)
		left_det_indcs = ((non_zero_window_y) & (non_zero_window_x_left)).nonzero()[0]
		right_det_indcs = ((non_zero_window_y) & (non_zero_window_x_right)).nonzero()[0]

		#Update the detected indices to the lists defined
		left_lane_indcs.append(left_det_indcs)
		right_lane_indcs.append(right_det_indcs)

		#If non zero pixels detced greater than the minimum pixels recenter next window on thier mean position
		if len(left_det_indcs) > pix_min:
			left_x_current = np.int(np.mean(nonzero_img_x[left_det_indcs]))
		if len(right_det_indcs) > pix_min:
			right_x_current = np.int(np.mean(nonzero_img_x[right_det_indcs]))

	#Concatenate the array of indices
	left_lane_indcs = np.concatenate(left_lane_indcs)
	right_lane_indcs = np.concatenate(right_lane_indcs)

	#Extract the left and right pixel positions
	left_x = nonzero_img_x[left_lane_indcs]
	left_y = nonzero_img_y[left_lane_indcs]
	right_x = nonzero_img_x[right_lane_indcs]
	right_y = nonzero_img_y[right_lane_indcs]

	return left_x, left_y, right_x, right_y, img_out

def polynomial_fitting(img):
	#Detect the lane pixels
	left_x, left_y, right_x, right_y, img_out = lane_detct_sliding_windows(img)
	# cv2.imshow('image',img_out)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#Fitting a second order polynomial
	left_poly_fit = np.polyfit(left_y, left_x, 2)
	right_poly_fit = np.polyfit(right_y, right_x, 2)

	#Generate x and y values for plotting
	img_height = img.shape[0]
	y_plot = np.linspace(0,img_height-1, img_height)
	try:
		left_poly_fit_x = left_poly_fit[0]*y_plot**2 + left_poly_fit[1]*y_plot + left_poly_fit[2]
		right_poly_fit_x = right_poly_fit[0]*y_plot**2 + right_poly_fit[1]*y_plot + right_poly_fit[2]
	except TypeError:
		print ('The function failed to fit a polynomial')
		left_poly_fit_x = 1*y_plot**2 + 1*y_plot
		right_poly_fit_x = 1*y_plot**2 + 1*y_plot

	#Visualization of fit polynomials
	# Colors in the left and right lane regions
	img_out[left_y, left_x] = [255, 0, 0]
	img_out[right_y, right_x] = [0, 0, 255]

	# # Plots the left and right polynomials on the lane lines
	# print(len(left_poly_fit_x))
	# print(len(y_plot))
	# print(left_poly_fit_x[360])
	# print(y_plot[360])
	# roi_tpz_pt1 = [(left_poly_fit_x[0]), (y_plot[0])]
	# roi_tpz_pt2 = [(left_poly_fit_x[240]), (y_plot[240])]
	# roi_tpz_pt3 = [(left_poly_fit_x[480]), (y_plot[480])]
	# roi_tpz_pt4 = [(left_poly_fit_x[-1]), (y_plot[-1])]
	# roi_tpz_pts = np.array([roi_tpz_pt1,roi_tpz_pt2,roi_tpz_pt3,roi_tpz_pt4])
	# color_tpz = (0,255,0)
	# thickness_tpz = 10
	# img_out = cv2.polylines(img_out,np.int32([roi_tpz_pts]),False,color_tpz,thickness_tpz)

	return left_poly_fit_x,right_poly_fit_x,y_plot

#Function to find the radius of curvature
def radius_of_curvature(img):
    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ym_per_pix = 30/720 
    # meters per pixel in x dimension
    xm_per_pix = 3.7/700
    #Detect the left side and right side points using the polynomial fit function
    left_poly_fit_x,right_poly_fit_x,y_plot = polynomial_fitting(img)
    #Fitting new polynomials into real world space
    left_fit_cr = np.polyfit(y_plot*ym_per_pix, left_poly_fit_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(y_plot*ym_per_pix, right_poly_fit_x*xm_per_pix, 2)

    #Finding the maximum value of y based on bottom of image
    y_plot_max = np.max(y_plot)
    # Calculation of R_curve (radius of curvature)
    left_rad_curve = ((1 + (2*left_fit_cr[0]*y_plot_max*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_rad_curve = ((1 + (2*right_fit_cr[0]*y_plot_max*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature_rad = (left_rad_curve + right_rad_curve) // 2 
    # print (left_rad_curve)
    # print (right_rad_curve)
    return curvature_rad

#Function to draw the lanes
def draw_lanes(img):
	#Creating a blank image to draw the detected lanes
	img_out = np.zeros_like(img)
	#Detect the left side and right side points using the polynomial fit function
	left_poly_fit_x,right_poly_fit_x,y_plot = polynomial_fitting(img)
	#Width of image
	camera_center = img.shape[1] // 2
	# meters per pixel in y dimension
	ym_per_pix = 30/720 
	# meters per pixel in x dimension
	xm_per_pix = 3.7/700
	#Calculate the middle point
	pts_middle_x = (left_poly_fit_x + right_poly_fit_x)//2
	#Calcualte the left and right points for the lane
	pts_left = np.array([np.flipud(np.transpose(np.vstack([left_poly_fit_x, y_plot])))])
	pts_right = np.array([np.transpose(np.vstack([right_poly_fit_x, y_plot]))])
	pts_middle = np.array([np.flipud(np.transpose(np.vstack([(pts_middle_x), y_plot])))])
	pts = np.hstack((pts_left, pts_right))
	# Drawing the left lane
	cv2.polylines(img_out, np.int_([pts_left]), isClosed=False, color=(255,0,0), thickness = 40)
	# Drawing the left lane
	cv2.polylines(img_out, np.int_([pts_right]), isClosed=False, color=(0,0,255), thickness = 40)
	#Filling the middle of the lane
	cv2.fillPoly(img_out, np.int_([pts]), (0,255,0))


	#Printing distance to middle into the final image
	diff_center = round(((pts_middle_x[1] - camera_center)*xm_per_pix),3)
	return diff_center, img_out

#Function for pipeline for processing images
def pipeline_image_lane_det(img):
	#Undistort the images
	undist_img = cv2.undistort(img,mtx,dist,None,mtx)
	#Appending the name to the otput undistorted image file
	print_name_undist = './output_images/Undisorted_test'+str(idx+1)+'.jpg'
	#Creating the undistorted image
	cv2.imwrite(print_name_undist, undist_img)
	#Process the image and create the bianry gradients
	# undist_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# #Undistorted gray image with 3 channels
	# undist_img_gray_3ch = cv2.merge([undist_img_gray,undist_img_gray,undist_img_gray])
	#bin_grad_img = np.zeros_like(undist_img[:,:,0])
	bin_grad_img = np.zeros_like(undist_img)
	#Finding the gradient in x-direction
	x_grad = abs_sobel_thresh(undist_img, orient ='x', thresh_min = 20, thresh_max = 100 )
	#Finding the gradient in y-direction
	y_grad = abs_sobel_thresh(undist_img, orient ='y', thresh_min = 20, thresh_max = 100 )
	#Finding the magnitude gradient
	mag_grad = mag_thresh(undist_img, sobel_kernel = 3, thresh_min =30, thresh_max = 100)
	#Finding the direction gradient
	dir_grad = dir_threshold(undist_img, sobel_kernel=3, thresh_min=0.7, thresh_max=1.3)
	#Finding the color gradient
	color_grad = color_threshold(undist_img,sthresh_min=100, sthresh_max=255, vthresh_min=50, vthresh_max=255)
	#Finding the binary image using all the detected gradients
	bin_grad_img[(((x_grad == 1) & (y_grad == 1)) | ((mag_grad == 1) & (dir_grad == 1)))| (color_grad ==1)] = 255
	#Appending the name to the otput combined binary gradient image file
	print_name_grad = './output_images/CombinedColorGrad_test'+str(idx+1)+'.jpg'
	#Creating the binary gradient image
	cv2.imwrite(print_name_grad, bin_grad_img)

	#Find the image height and width
	img_height = img.shape[0]
	img_width = img.shape[1]
	#Defining the Region of interest trapezium
	mid_tpz_offset = 90
	botm_tpz_offset = 100 
	roi_tpz_pt1 = [(img_width*0.5 - mid_tpz_offset), img_height*0.63]
	roi_tpz_pt2 = [(img_width*0.5 + mid_tpz_offset), img_height*0.63]
	roi_tpz_pt3 = [botm_tpz_offset, img_height*0.93]
	roi_tpz_pt4 = [(img_width - botm_tpz_offset), img_height*0.93]
	
	#Assigning the parameters to draw a trpaezium on the image using the defined points of region of interest
	roi_tpz_pts = np.array([roi_tpz_pt1,roi_tpz_pt2,roi_tpz_pt4,roi_tpz_pt3])
	# color_tpz = (0,0,255)
	# thickness_tpz = 10
	# img_tpz = cv2.polylines(bin_grad_img,np.int32([roi_tpz_pts]),True,color_tpz,thickness_tpz)
	# # print_name_tpz = './output_images/ROI_trapezium_test'+str(idx+1)+'.jpg'
	# # cv2.imwrite(print_name_tpz, img_tpz)
	
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
	warped_image = cv2.warpPerspective(bin_grad_img, M, img_size, flags=cv2.INTER_LINEAR) 

	#Appending the name to the otput combined binary gradient image file
	print_name_warped = './output_images/Warped_test'+str(idx+1)+'.jpg'
	#Creating the warped image
	cv2.imwrite(print_name_warped, warped_image)

	#Perform the lane finding using sliding windows method and polynomial fitting
	diff_center, lane_poly_fit = draw_lanes(warped_image)
	#Calculating the radius of curvature
	rad_curve = radius_of_curvature(warped_image)
	#Appending the name to the otput combined binary gradient image file
	print_name_lane_det = './output_images/Lane_Detec_warp_test'+str(idx+1)+'.jpg'
	#Creating the Lane detected image
	cv2.imwrite(print_name_lane_det, lane_poly_fit)
	#Create the unwarped image
	unwarped_image = cv2.warpPerspective(lane_poly_fit, Minv, img_size, flags=cv2.INTER_LINEAR)
	#Appending the name to the otput combined binary gradient image file
	print_name_lane_det_unwr = './output_images/Lane_Detec_Unwarpped_test'+str(idx+1)+'.jpg'
	#Creating the Lane detected image
	cv2.imwrite(print_name_lane_det_unwr, unwarped_image)

	img_out = cv2.addWeighted(img,1.0,unwarped_image,1.0,0)
	#Printing Radius of curvature and  into the final image
	curvature = round(radius_of_curvature(img),3)
	cv2.putText(img_out, 'Radius of curvature = '+str(curvature)+'m', (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
	#Printing distance to middle into the final image
	cv2.putText(img_out, 'Distance from center = '+str(diff_center)+'m', (40,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
	#Appending the name to the otput combined binary gradient image file
	print_name_lane_det_final = './output_images/Lane_det_final_test'+str(idx+1)+'.jpg'
	#Creating the Lane detected image
	cv2.imwrite(print_name_lane_det_final, img_out)

	return img_out

#Function for pipeline for processing images without outputting anyintermediate images
#This is exactly similar to the previous function but is not geneerating intermediate files into output_images
def pipeline_image_lane_det_for_videogen(img):
	#Undistort the images
	undist_img = cv2.undistort(img,mtx,dist,None,mtx)
	#Process the image and create the bianry gradients
	bin_grad_img = np.zeros_like(undist_img)
	#Finding the gradient in x-direction
	x_grad = abs_sobel_thresh(undist_img, orient ='x', thresh_min = 20, thresh_max = 100 )
	#Finding the gradient in y-direction
	y_grad = abs_sobel_thresh(undist_img, orient ='y', thresh_min = 20, thresh_max = 100 )
	#Finding the magnitude gradient
	mag_grad = mag_thresh(undist_img, sobel_kernel = 3, thresh_min =30, thresh_max = 100)
	#Finding the direction gradient
	dir_grad = dir_threshold(undist_img, sobel_kernel=3, thresh_min=0.7, thresh_max=1.3)
	#Finding the color gradient
	color_grad = color_threshold(undist_img,sthresh_min=100, sthresh_max=255, vthresh_min=50, vthresh_max=255)
	#Finding the binary image using all the detected gradients
	bin_grad_img[(((x_grad == 1) & (y_grad == 1)) | ((mag_grad == 1) & (dir_grad == 1)))| (color_grad ==1)] = 255

	#Find the image height and width
	img_height = img.shape[0]
	img_width = img.shape[1]
	#Defining the Region of interest trapezium
	mid_tpz_offset = 90
	botm_tpz_offset = 100 
	roi_tpz_pt1 = [(img_width*0.5 - mid_tpz_offset), img_height*0.63]
	roi_tpz_pt2 = [(img_width*0.5 + mid_tpz_offset), img_height*0.63]
	roi_tpz_pt3 = [botm_tpz_offset, img_height*0.93]
	roi_tpz_pt4 = [(img_width - botm_tpz_offset), img_height*0.93]
	
	#Assigning the parameters to draw a trpaezium on the image using the defined points of region of interest
	roi_tpz_pts = np.array([roi_tpz_pt1,roi_tpz_pt2,roi_tpz_pt4,roi_tpz_pt3])
	
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
	warped_image = cv2.warpPerspective(bin_grad_img, M, img_size, flags=cv2.INTER_LINEAR) 

	#Perform the lane finding using sliding windows method and polynomial fitting
	diff_center, lane_poly_fit = draw_lanes(warped_image)
	#Calculating the radius of curvature
	rad_curve = radius_of_curvature(warped_image)

	#Create the unwarped image
	unwarped_image = cv2.warpPerspective(lane_poly_fit, Minv, img_size, flags=cv2.INTER_LINEAR)

	img_out = cv2.addWeighted(img,1.0,unwarped_image,1.0,0)
	#Printing Radius of curvature and  into the final image
	curvature = round(radius_of_curvature(img),3)
	cv2.putText(img_out, 'Radius of curvature = '+str(curvature)+'m', (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
	#Printing distance to middle into the final image
	cv2.putText(img_out, 'Distance from center = '+str(diff_center)+'m', (40,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

	return img_out
def process_image(img):
	result = pipeline_image_lane_det_for_videogen(img)

	return result

Output_video = 'output_videos/harder_challenge_video_video_lane_find.mp4'
Input_video = 'harder_challenge_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(Output_video, audio=False)

