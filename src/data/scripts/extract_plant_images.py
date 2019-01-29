import cv2
import numpy as np
import os


'''
Given images of individual growth modules, extract images of constituent plants
'''


MINIMUM_LEAF_SIZE = 12000
OFFSET = 20
THRESH_LOWER = np.array([25, 0, 40], np.uint8)
THRESH_UPPER = np.array([55, 230, 100], np.uint8)
REMOVE_INVADING_MATERIAL = False

plant_count = 0

def extract_plants(image_array, module_number, output_path, background_array=None):

	source = image_array.copy()
	image = image_array

	# reduce to single color space
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# blur image to help remove noise
	blur = cv2.GaussianBlur(hsv,(5,5),0)

	# set threshold; assuming bimodal distribution of green leaf on uniform background
	thresh = cv2.inRange(blur, THRESH_LOWER, THRESH_UPPER)
	#cv2.imwrite('thresh_{}.png'.format(module_number), thresh)

	# compute contours
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cim = image.copy()
	cim = cv2.bitwise_not(cv2.drawContours(cim, contours, -1, (0,0,0), -1))

	for j, cnt in enumerate(contours):
		area = cv2.contourArea(cnt)
		# handle bad output from contours
		if (area is None) or (area < MINIMUM_LEAF_SIZE):
			continue
		x,y,w,h = cv2.boundingRect(cnt)
		rect = cv2.minAreaRect(cnt) # return centroid, h&w, angle
		box = cv2.boxPoints(rect) # extract box corners
		box = np.int0(box) # round each to nearest int
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.drawContours(image,[box],0,(0,0,255),2)

		result = source
		if REMOVE_INVADING_MATERIAL:
			stencil = np.zeros(cim.shape).astype(cim.dtype)
			color = [255, 255, 255]
			stencil = cv2.fillPoly(stencil, [cnt], color)
			stencil_not = cv2.bitwise_not(stencil)
			result = cv2.bitwise_and(source, stencil)
			result = cv2.bitwise_or(result, stencil_not)

		# crop image of leaf
		lower_y = y - OFFSET
		upper_y = y + h + OFFSET
		lower_x = x - OFFSET
		upper_x = x + w + OFFSET
		if lower_y < 0: 
			lower_y = 0
		if lower_x < 0: 
			lower_x = 0
		if upper_y > result.shape[0]: 
			upper_y = result.shape[0]
		if upper_x > result.shape[1]: 
			upper_x = result.shape[1]

		leaf_roi = result[lower_y:upper_y, lower_x:upper_x]

		# consider moving this to ram; you know, for performance.
		global plant_count
		plant_count += 1
		filename = '{}.png'.format(plant_count)
		cv2.imwrite(output_path+'/'+filename, leaf_roi)


def main(input_path, output_path): 
	filelist = [f for f in os.listdir(input_path) if '.jpg' in f]
	print('Extracting plant images\n')
	for f in filelist:
		growth_module_number = f.replace(".jpg", "")
		extract_plants(cv2.imread(input_path+'/'+f), growth_module_number, output_path)
	print('Done extracting plant images\n')





