import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

dist_pickle = pickle.load( open ("camera_cal/calibration_pickle.p","rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

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

	return left_poly_fit_x,right_poly_fit_x,y_plot,img_out

#Function to find the radius of curvature
def radius_of_curvature(img):
    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ym_per_pix = 30/720 
    # meters per pixel in x dimension
    xm_per_pix = 3.7/700
    #Detect the left side and right side points using the polynomial fit function
    left_poly_fit_x,right_poly_fit_x,y_plot,img_poly = polynomial_fitting(img)
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
	img_out = np.zeros_like(img)
	#Detect the left side and right side points using the polynomial fit function
	left_poly_fit_x,right_poly_fit_x,y_plot,img_poly = polynomial_fitting(img)
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

	#Printing Radius of curvature and  into the final image
	curvature = round(radius_of_curvature(img),3)
	#cv2.putText(img_out, 'Radius of curvature = '+str(curvature)+'m', (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
	#Printing distance to middle into the final image
	diff_center = round(((pts_middle_x[1] - camera_center)*xm_per_pix),3)
	#cv2.putText(img_out, 'Distance from center = '+str(diff_center)+'m', (40,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
	return img_out



#Import the Warped Images
images = glob.glob('./output_images/Warped_test*.jpg')

#Iterate through the images and detect the lane and perform a ploynomial fit
for idx, fname in enumerate(images):
	#Pull in the Undistorted images for processing
	img = cv2.imread(fname)
	img_cpy = np.copy(img)

	#Print the currently processing image
	print ('Currently processing image', fname)
	left_poly_fit_x,right_poly_fit_x,y_plot,img_out = polynomial_fitting(img)
	sliding_windows = img_out
	print_name_sw = './output_images/Lane_Detec_slide_test'+str(idx+1)+'.jpg'
	cv2.imwrite(print_name_sw, sliding_windows)
	lane_poly_fit = draw_lanes(img)
	rad_curve = radius_of_curvature(img)
	result = lane_poly_fit 

	print_name = './output_images/Lane_Detec_warp_test'+str(idx+1)+'.jpg'
	cv2.imwrite(print_name, result)







