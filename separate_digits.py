#!/usr/bin/env python

import os
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import math

def get_average_angle(lines, filename):
	summed_angle = 0
	count_summed = 0
	if lines is not None:
		for line in lines:
			for x1,y1,x2,y2 in line:
				angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
				if (angle <= 10 and angle >= -10):
					summed_angle += angle
					count_summed += 1
				if angle >= 80 and angle < 90:
					summed_angle += (90 - angle)
					count_summed += 1 
				if angle <= -80 and angle > -90:
					summed_angle += 90 + angle
					count_summed += 1
	count_summed = count_summed if count_summed is not 0 else 1;
	return summed_angle / count_summed

if __name__=="__main__":
	CAM_DIR = "./test_set/"
	OUTPUT_DIR = "./outputs"
	DIGITS_DIR = "./digits"
	for f in os.listdir(OUTPUT_DIR):
		os.remove(os.path.join(OUTPUT_DIR, f))
	for f in os.listdir(DIGITS_DIR):
		os.remove(os.path.join(DIGITS_DIR, f))
	THRESHOLD_FOREGROUND = 0.125
	ANGLE_RANGE = 10
	CANNY_LOW = 50
	CANNY_HIGH = 150
	HOUGH_RHO = 1
	HOUGH_THETA = np.pi / 180
	HOUGH_MIN_LINE_LEN = 40 
	HOUGH_MAX_LINE_GAP = 10
	HOUGH_THRES = 5
	for filename in os.listdir(CAM_DIR):
		gray_image = cv2.imread(os.path.join(CAM_DIR, filename), cv2.IMREAD_GRAYSCALE)
		count, bins = np.histogram(gray_image, bins=range(0, 256))
		total_pixels = gray_image.shape[0] * gray_image.shape[1]
		pixels_covered = 0
		threshold_value = 0
		for bin_idx in range(len(bins) - 2, 0, -1):
			pixels_covered += count[bin_idx]
			if pixels_covered / total_pixels > THRESHOLD_FOREGROUND:
				threshold_value = bins[bin_idx]
				break
		thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)[1]
		edge_image = cv2.Canny(thresholded_image, CANNY_LOW, CANNY_HIGH)
		lines = cv2.HoughLinesP(edge_image, HOUGH_RHO, HOUGH_THETA, HOUGH_THRES, np.array([]), minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP)
		#line_image = np.zeros((edge_image.shape[0], edge_image.shape[1]), dtype=np.uint8)
		#draw_lines(line_image, lines)
		average_angle = get_average_angle(lines, filename)
		print(average_angle, filename)
		rotated_image = rotate(thresholded_image, average_angle, reshape=False)
		intensity_profile = np.sum(rotated_image, axis=0)

		intensity_profile[ intensity_profile < 0.05 * np.max(intensity_profile)] = 0

		# initial cut at the front and end
		zero_areas = [];
		start = 0
		end = 0
		i = 0
		detected = False

		intensity_profile_new = np.zeros(intensity_profile.shape[0])
		# median filter 
		for i in range(1, intensity_profile.shape[0] - 1):
			intensity_profile_new[i] = np.median([intensity_profile[i - 1], intensity_profile[i], intensity_profile[i + 1]])
		i = 0
		while i < intensity_profile_new.shape[0]:
			if intensity_profile_new[i] == 0:
				if detected == True:
					start = i
					detected = False
			else:
				end = i
				if detected == False:
					zero_areas.append([start, end])
					detected = True
			i += 1
		zero_areas.append([start, len(intensity_profile_new) - 1])
		
		cut_images = []
		for i in range(len(zero_areas) - 1):
			center_a = (zero_areas[i][0] + zero_areas[i][1])//2;
			center_b = (zero_areas[i+1][0] + zero_areas[i+1][1])//2;
			cut_images.append(rotated_image[:, center_a : center_b])
			out_file_name = ".".join(filename.split(".")[:-1]) + "_%d.png"%i 
			print(out_file_name)
			cv2.imwrite(os.path.join(DIGITS_DIR, out_file_name), cut_images[-1])
		# fig, ax = plt.subplots()
		# plt.bar(range(len(intensity_profile_new)), intensity_profile_new)
		# plt.show()
		cv2.imwrite(os.path.join(OUTPUT_DIR, filename), np.vstack((gray_image, thresholded_image, edge_image, rotated_image)))	
