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


def tranpose(mat):
	"""
		The perpose of the function is to fix the problem caused by the matrice indexing 
		difference between matlab and C++
		mat: matrice to be transform
		h: height
		w: width 
	"""

	temp = np.zeros((mat.shape[0],mat.shape[1]))
	for i,item1 in enumerate(mat):
		for j,item2 in enumerate(item1):
			temp[i,j] = item2

	return temp


def test_tsp(sp_labels):

	output_loc = "./segmentation_result/"

	a = Joe_segmentation('./frame/00020.jpg', './svmfile.pkl')
	result = np.array(a.segmentation_tsp(sp_labels),dtype=np.uint8)

	# contours = slic.contours(result, a.slic_result)
	sp = np.array(tranpose(sp_labels),dtype = np.int32)
	contours = slic.contours(result, sp)


	cv2.imshow("original",a.original_image)
	cv2.imshow("result",contours)
	cv2.waitKey(0)

def tracking(list1,label1,label2):

	list_sp = np.unique(label1)
	seg2 = np.zeros((label1.shape[0],label1.shape[1]))
	seg2 = seg2 - 1

	for index , i  in enumerate(list_sp):
		seg2[np.where(label2 == i)] = list1[index]

	out = visualization(seg2)

	return out

def visualization(mat):

	output = [1,2,3]


	for i in range(3):
		output[i] = np.zeros(mat.shape)
		output[i][np.where(mat == i)] = 255

	out = np.array(cv2.merge([output[0],output[1],output[2]]), dtype = np.uint8)

	return out


def main():

	"""
		change the parameter in sp_label20 , sp_label19 , a and b . You can show the seg tracking from 

		previous frame you want.



	"""
	mat = scipy.io.loadmat("/Users/zhufucai/Documents/GitHub/TSP_matlab/results/sp_labels_banana.mat")
	# image no. 100 ; mat no. 99 : because mat start at 0 n image start at 1 !!!!
	sp_labels2 = mat['sp_labels'][:,:,94]

	sp_labels2 = np.array(sp_labels2,dtype=np.int32)

	# sp_labels1 = mat['sp_labels'][:,:,18]
	sp_labels1 = mat['sp_labels'][:,:,93]

	a = Joe_segmentation('./frame/00094.jpg', './svmfile.pkl')

	[result, list_seg19] = a.segmentation_tsp(sp_labels1)
	result19 = np.array(result,dtype=np.uint8)

	seg = tracking(list_seg19, sp_labels1, sp_labels2)
	sp20 = np.array(tranpose(sp_labels2),dtype = np.int32)
	contours_seg = slic.contours(seg, sp20)


	b = Joe_segmentation('./frame/00095.jpg', './svmfile.pkl')
	[result, list_seg20] = b.segmentation_tsp(sp_labels2)
	result20 = np.array(result,dtype=np.uint8)
	# sp20 = np.array(tranpose(sp_labels2),dtype = np.int32)
	contours20 = slic.contours(result20, sp20)



	cv2.imshow("result_track", contours_seg)
	cv2.imshow("result_seg",contours20)
	cv2.waitKey(0)

main()