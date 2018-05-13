from Joe_segmentation import Joe_segmentation
import scipy.io 
from scipy.spatial import ConvexHull
import numpy as np
import os
import cv2
import slic
import matplotlib.pyplot as plt
import sys

print sys.version

mat = scipy.io.loadmat("/Users/zhufucai/Documents/GitHub/TSP_matlab/results/sp_labels_banana.mat")
# image no. 100 ; mat no. 99 : because mat start at 0 n image start at 1 !!!!
sp_labels19 = mat['sp_labels'][:,:,19]

sp_labels19 = np.array(sp_labels19,dtype=np.int32)

def contour(mask):
	output = mask.copy()
	list_sp = np.unique(mask)

	for i in list_sp:
		pass

	return hull

def print_tsp_label(label):
	for i in label:

		print i
		print " "


def test_tsp(sp_labels19):

	output_loc = "./segmentation_result/"

	a = Joe_segmentation('./frame/00020.jpg', './svmfile.pkl')
	result = np.array(a.segmentation_tsp(sp_labels19),dtype=np.uint8)

	# contours = slic.contours(result, a.slic_result)
	sp = np.array(tranpose(sp_labels19,272,480),dtype = np.int32)
	contours = slic.contours(result, sp)


	cv2.imshow("original",a.original_image)
	cv2.imshow("result",contours)
	cv2.waitKey(0)


def generate_tsp_seg_result():
	output_loc = "./segmentation_result/"
	try:
		os.mkdir(output_loc)
	except OSError:
		pass
	range(100)
	for i in range(100):
		print i
		a = Joe_segmentation('./frame/%#05d.jpg'%(i+1), './svmfile.pkl')

		mat = scipy.io.loadmat("/Users/zhufucai/Documents/GitHub/TSP_matlab/results/sp_labels_banana.mat")
		sp_labels = np.array(mat['sp_labels'][:,:,i], dtype = np.int32)
		print sp_labels.shape


		result = np.array(a.segmentation_tsp(sp_labels),dtype=np.uint8)
		# fix the problem of mat & C++ dont share the same indexing way
		sp = np.array(tranpose(sp_labels,sp_labels.shape[0],sp_labels.shape[1]),dtype = np.int32)
		contours = slic.contours(result, sp)

		cv2.imwrite(output_loc + "/%#05d.jpg"%(i+1), contours)

def tranpose(mat,h,w):
	temp = np.zeros((h,w))
	for i,item1 in enumerate(mat):
		for j,item2 in enumerate(item1):
			temp[i,j] = item2

	return temp

test_tsp(sp_labels19)
# test_tsp(sp_labels19)
# print sp_labels19

# # cv2.imshow("result",contour)
# cv2.waitKey(0)