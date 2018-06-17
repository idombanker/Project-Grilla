from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
import numpy as np 
from numpy import dot
from scipy.linalg import inv
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Joe_segmentation import Joe_segmentation

def xy_measurement(pr_y, pr_x, L, centroid_list, index_mat, label_list,tmp_dmin = 999999):
# return an array : [m_x, m_y]
    
    pr_x = int(pr_x)
    pr_y = int(pr_y)
    h, w = index_mat.shape
    search_range = 10
    tmp_mat = np.zeros((search_range*2, search_range*2),dtype = int)
    
    sp_index = -1

    # time_consuming
    # print "shape of image, h: %f w:%f"%(h,w)
    # print "pr_x:%d,pr_y:%d"%(pr_x,pr_y)
    # print pr_x,pr_y

    # pr_x and pr_y might be float due to the transition
    y_min = pr_y-search_range
    y_max = pr_y+search_range
    x_min = pr_x-search_range
    x_max = pr_x+search_range

    if y_min < 0:
    	y_min = 0
    if y_max > h:
    	y_max = h
    	# print y_max-50
    if x_min < 0:
    	x_min = 0
    if x_max > w:
    	x_max = w

   #  if (pr_x > search_range) and (pr_x < (w - search_range))\
	 	# and (pr_y > (search_range)) and (pr_y < (h - search_range)):
    # print "section"
    # if True:

    tmp_mat =  index_mat[y_min:y_max, x_min:x_max]
    # tmp_mat =  index_mat[x_min:x_max, y_min:y_max]

    # hard coding here: a substitute?
    m_x = pr_x
    m_y = pr_y


    for count, index in enumerate(np.unique(tmp_mat)):
    	# print np.unique(tmp_mat)
    	# if label_list[index] == L:
    	if True :
    		D = np.square(centroid_list[index][0] - pr_y) + np.square(centroid_list[index][1] - pr_x)
    		if D < tmp_dmin:
    			tmp_dmin = D
    			m_y = centroid_list[index][0] 
    			m_x = centroid_list[index][1]
    			sp_index = index
    			# print index

    		else:
    			pass
    	else:
    		# print "label_list[index]:%d, L:%d"%(label_list[index],L)
    		# # print m_x , m_y
    		m_x, m_y = pr_x, pr_y

	

    return m_x, m_y , sp_index

def tracking(pr_y, pr_x, L, centroid_list, index_mat, label_list,tmp_dmin = 999999):
# return an array : [m_x, m_y]
    
    pr_x = int(pr_x)
    pr_y = int(pr_y)
    h, w = index_mat.shape
    search_range = 10
    tmp_mat = np.zeros((search_range*2, search_range*2),dtype = int) 
    sp_index = -1

    y_min = pr_y-search_range
    y_max = pr_y+search_range
    x_min = pr_x-search_range
    x_max = pr_x+search_range

    if y_min < 0:
    	y_min = 0
    if y_max > h:
    	y_max = h
    if x_min < 0:
    	x_min = 0
    if x_max > w:
    	x_max = w

    tmp_mat =  index_mat[y_min:y_max, x_min:x_max]
    m_x = pr_x
    m_y = pr_y

    for count, index in enumerate(np.unique(tmp_mat)):
    	if True :
    		D = np.square(centroid_list[index][0] - pr_y) + np.square(centroid_list[index][1] - pr_x)
    		if D < tmp_dmin:
    			tmp_dmin = D
    			m_y = centroid_list[index][0] 
    			m_x = centroid_list[index][1]
    			sp_index = index
    		else:
    			pass
    	else:
    		m_x, m_y = pr_x, pr_y


	

    return sp_index

def main():
	# kf = KalmanFilter(dim_x=4, dim_z=2)

	# testing #
	###########
	num_kf = 500

	dim_x = 4
	dt = 1.
	x_R_var = 10.0
	y_R_var = 10.0
	Q_var = 0.01
	x = np.array([[10, 1.0, 10, 1.0]]).T
	P = np.diag([500., 49., 500., 49.])
	F = np.array([[1., dt, 0, 0],
	              [0, 1., 0, 0],
	              [0, 0, 1., dt],
	              [0, 0, 0, 1.]])
	H = np.array([[1., 0, 0, 0],
				  [0, 0, 1., 0]])
	R = np.array([[x_R_var, 0.],
				  [0, y_R_var]])
	q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)
	Q = block_diag(q, q)

	# Kalman Filter parameters initialization
	# KF_sp_index: store the corresponding index of superpixel in previous frame (the index of sp not Kalman Filter States)
	# KF_sp_tracking_index: store the corresponding index of superpixel in the current frame  
	KF_X = [x]*num_kf
	KF_P = [P]*num_kf
	KF_F = [F]*num_kf
	KF_H = [H]*num_kf
	KF_R = [R]*num_kf
	KF_Q = [Q]*num_kf
	KF_sp_index = [-1]*num_kf
	KF_sp_tracking_index = [-1]*num_kf
	KF_L = [-1]*num_kf




	# Initialization Preparation...
	im_index = 20
	input_image = "./frame/%#05d.jpg"%im_index
	seg = Joe_segmentation(input_image, './svmfile.pkl')
	sp_mat = seg.slic_result
	cen_index = seg.centroid_calc()

	# testing # # KF_L_Probability 500x3 initialization
	label_result, sp_list, KF_L_proba = seg.segmentation()
	# print len(KF_L_proba)

	# breaking point
	# cv2.waitKey(0)
	# breaking point 


	cv2.imshow("original_image",seg.original_image)
	cv2.imshow("original_label",label_result)

	##########
	# testing #
	# print seg
	# p = seg.color_point(sp_mat, cen_index)
	# cv2.imshow("cen", p)
	# cv2.waitKey(0)
	##########

	label_result, label_index ,label_proba_list= seg.segmentation()
	number_sp = len(np.unique(sp_mat))


	for n in range(number_sp):
		KF_L[n] = label_index[n]


	for n in range(number_sp):
		KF_X[n] = np.array([[cen_index[n][0], 0.0 , cen_index[n][1], 0.0]]).T


	xs = []

	# tracking base on tracking



	# for i in range(num_frame):
	for i in range(5):
		input_image = "./frame/%#05d.jpg"%(im_index+i+1)
		# here have to load svm mat every time, think about whehter can better it
		seg = Joe_segmentation(input_image, './svmfile.pkl')
		sp_mat = seg.slic_result
		label_result, sp_list , label_proba_list = seg.segmentation()
		cen_index = seg.centroid_calc()

		# p = seg.color_point(sp_mat, cen_index)
		# cv2.imshow("cen", p)
		# cv2.waitKey(0)

		# for i in range(num_kf):
		# 	pass

		# temp_sp_index = KF_sp_index.copy()

		for j in range(num_kf):
			
			# predict
			KF_X[j] = dot(KF_F[j], KF_X[j])
			KF_P[j] = dot(KF_F[j], KF_P[j]).dot(KF_F[j].T) + KF_Q[j]


			# update
			S = dot(KF_H[j], KF_P[j]).dot(KF_H[j].T) + KF_R[j]
			K = dot(KF_P[j], KF_H[j].T).dot(inv(S))

			c_x, c_y , KF_sp_index[j] = xy_measurement(np.asscalar(KF_X[j][0]), np.asscalar(KF_X[j][2]),\
			 											KF_L[j], cen_index, sp_mat,sp_list )

					
			z = np.array([[c_y], [c_x]])


			y = z - dot(KF_H[j], KF_X[j])
			KF_X[j] += dot(K, y)
			KF_P[j] = KF_P[j] - dot(K, KF_H[j]).dot(KF_P[j])


			# need to rewrite!!!
			KF_sp_tracking_index[j]= tracking(np.asscalar(KF_X[j][0]), np.asscalar(KF_X[j][2]),\
			 											KF_L[j], cen_index, sp_mat,sp_list) 


		cv2.imshow("xy_measurement",form_label_mat(KF_sp_index,KF_L,sp_mat))
		cv2.imshow("tracking",form_label_mat(KF_sp_tracking_index,KF_L,sp_mat))
		label_combination(KF_sp_tracking_index,KF_L_proba,sp_mat,label_proba_list)

		for index,i in enumerate(KF_sp_index):
			KF_L_proba[index] = label_proba_list[i]


	# print KF_sp_index	
	# testing #
	# print seg

	# test = np.zeros([num_kf,2])


	# for i in range(num_kf):
	# 	test[i][0], test[i][1] = KF_X[i][0],KF_X[i][2]

	# print test
	# p = seg.color_point(sp_mat,test)
	# cv2.imshow("cen", p)
	# print test
	# cv2.waitKey(0)
	# # # # # # 
	

	# testing #
	cv2.imshow("current frame",seg.original_image)
	cv2.imshow("label result from svm", label_result)
	cv2.waitKey(0)

# base on Kalman Filter tutorial to draw the moving trajectory

def form_label_mat(list1,list2,sp_mat):

	"""
		This function form label mat from kalman superpixel index list and kalman 
		state label list.
		list1: superpixel index list
		list2: label list
		sp_mat: current frame superpixel index mat

	"""
	list_sp = np.unique(sp_mat)
	seg = np.zeros((sp_mat.shape[0],sp_mat.shape[1]))
	seg = seg - 1

	# print len(np.unique(list1))
	for index , i  in enumerate(list1):
		seg[np.where(sp_mat == i)] = list2[index]

	out = visualization(seg)

	# testing #
	
	return out

def label_combination(list1,list2,sp_mat,list3):
	pass

	"""
		This function form label mat from kalman superpixel index list and kalman 
		state label list.
		list1: Kalman filter superpixel index list
		list2: label pro list from last frame
		sp_mat: current frame superpixel index mat
		list3: label pro list from current frame
	"""
	list_sp = np.unique(sp_mat)
	seg = np.zeros((sp_mat.shape[0],sp_mat.shape[1]))
	seg = seg - 1

	# base on the order of Kalman filter :
	# 500 of them , and track them through time
	for index , i  in enumerate(list_sp):
		seg[np.where(sp_mat == i)] = generate_label_from_proba(list2[find(list1,i)],list3[i])


	out = visualization(seg)

	cv2.imshow("label_combination",out)
	return out

def find(list1,val):
	index = -1 
	for i, item in enumerate(list1):
		if item == val:
			index = i
		else:
			pass
	return index 


def generate_label_from_proba(arr1,arr2):
	pass
	# return the indices 
	temp = np.argmax((arr1 + arr2)/2.0)
	return temp

def visualization(mat):

	output = [1,2,3]


	for i in range(3):
		output[i] = np.zeros(mat.shape)
		output[i][np.where(mat == i)] = 255

	out = np.array(cv2.merge([output[0],output[1],output[2]]), dtype = np.uint8)

	return out

main()

