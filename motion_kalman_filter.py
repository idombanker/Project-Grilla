from Joe_segmentation import Joe_segmentation
import cv2 
import numpy as np
import time
import matplotlib.pyplot as plt
import slic
from filterpy.kalman import KalmanFilter


def color_point(img,index):
	temp = np.ones([img.shape[0],img.shape[1]],dtype = np.uint16)

	for i in range(500):
		temp[index[i][0],index[i][1]] = 255

	out = cv2.merge((temp,temp,temp))
	out = np.array(out, dtype = np.uint8)

	result = slic.contours(out, seg.slic_result)

	return result

def measurement():
	pass

start = time.clock()
# ----------------------------------
# for i in range(10):


for i in range(1):
	input_image = "./frame/000%d.jpg"%(i+23)
	# here have to load svm mat every time, think about whehter can better it
	seg = Joe_segmentation(input_image, './svmfile.pkl')
	sp_mat = seg.slic_result

	index = seg.centroid_calc()
	# draw controid and contour in a canvas
	result = color_point(seg.original_image,index)

	
	cv2.imshow("centroid points", result)
	cv2.waitKey(0)


	# cv2.imshow("sp", seg.label_color())
	# cv2.waitKey(0)




# ----------------------------------
t = time.clock() - start
print "runing time: %f"%t