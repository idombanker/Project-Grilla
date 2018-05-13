import cv2
import numpy as np 
import os 

def image_to_video(im_loc,vi_loc):
	
	try:
		temp = 1
		frame = cv2.imread(im_loc + "/%#05d"%temp)

	except: 
		print "file doesn't exist"

	
