from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
import numpy as np 
from numpy import dot
from scipy.linalg import inv
import cv2
import math

from Joe_segmentation import Joe_segmentation

def xy_measurement(pr_x, pr_y, L, centroid_list, index_mat, label_list):
# return an array : [m_x, m_y]
    pr_x = int(pr_x)
    pr_y = int(pr_y)
    h, w = index_mat.shape
    search_range = 10
    tmp_mat = np.zeros((search_range*2, search_range*2),dtype = float)
    sp_index = 0


    # for i in range(search_range*2):

    # 	for j in range(search_range*2):
    		
    
    # 		tmp_mat[i][j] = index_mat[x-search_range+1+i][y-search_range+1+j]

    # time_consuming
    print "shape of image, h: %f w:%f"%(h,w)
    print "pr_x:%d,pr_y:%d"%(pr_x,pr_y)
    print pr_x,pr_y
    # pr_x and pr_y might be float due to the transition

    if (pr_x > search_range+1) and (pr_x < (w - search_range-1))\
	 	and (pr_y > (search_range+1)) and (pr_y < (h - search_range-1)):
	    tmp_mat =  index_mat[(pr_y-search_range):(pr_y+search_range), (pr_x-search_range):(pr_x+search_range)]
	    
	    # hard coding here: a substitute?
	    tmp_dmin = 1000
	    m_x = pr_x
	    m_y = pr_y

	    for sp_index in np.unique(tmp_mat):
	    	if label_list[sp_index] == L:
	    		D = np.square(centroid_list[sp_index][0] - pr_x) + np.square(centroid_list[sp_index][1] - pr_y)
	    		if D < tmp_dmin:
	    			tmp_dmin = D
	    			m_x = centroid_list[sp_index][0] 
	    			m_y = centroid_list[sp_index][1]


	    		else:
	    			pass
	    	else:
	    		pass
    else:
		m_x, m_y = pr_x, pr_y
    pass
    return m_x, m_y , sp_index

def main():
	# kf = KalmanFilter(dim_x=4, dim_z=2)
	num_kf = 500

	dim_x = 4
	dt = 1.
	x_R_var = 10.0
	y_R_var = 10.0
	Q_var = 0.01
	x = np.array([[10, 4.5, 10, 4.5]]).T
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

	KF_X = []
	KF_P = []
	KF_F = []
	KF_H = []
	KF_R = []
	KF_Q = []
	KF_sp_index = []

	for _ in range(num_kf):
		KF_P.append(P)

	for _ in range(num_kf):
		KF_F.append(F)

	for _ in range(num_kf):
		KF_H.append(H)

	for _ in range(num_kf):
		KF_R.append(R)

	for _ in range(num_kf):
		KF_Q.append(Q)

	KF_sp_index = [-1]*num_kf

	KF_X = []

	for _ in range(num_kf):
		KF_X.append(np.array([[0.,0.,0.,0.]]).T)

	KF_L = [-1]*num_kf

	im_index = 20
	input_image = "./frame/%#05d.jpg"%im_index
	seg = Joe_segmentation(input_image, './svmfile.pkl')
	sp_mat = seg.slic_result
	cen_index = seg.centroid_calc()
	label_result, label_index = seg.segmentation()
	number_sp = len(np.unique(label_index))

	for n in range(number_sp):
		KF_L[n] = label_index[n]

	for n in range(number_sp):
		KF_X[n] = np.array([[cen_index[n][0], 4.5 , cen_index[n][1], 4.5]]).T
		
	print KF_X[1]

	xs = []
	# for i in range(num_frame):
	for i in range(2):
		input_image = "./frame/%#05d.jpg"%(im_index+i+1)
		# here have to load svm mat every time, think about whehter can better it
		seg = Joe_segmentation(input_image, './svmfile.pkl')
		sp_mat = seg.slic_result
		label_result, sp_list = seg.segmentation()
		cen_index = seg.centroid_calc()
		# for j in range(seg.number_regions):
		for j in range(479):
			# predict

			KF_X[j] = dot(KF_F[j], KF_X[j])
			KF_P[j] = dot(KF_F[j], KF_P[j]).dot(KF_F[j].T) + KF_Q[j]

			# update
			S = dot(KF_H[j], KF_P[j]).dot(KF_H[j].T) + KF_R[j]
			K = dot(KF_P[j], KF_H[j].T).dot(inv(S))

			# print "shape of K"
			# print K.shape
			# measurement function
			#
			# print np.asscalar(KF_X[j][0])
			print KF_X[j]
			print KF_L[j]
			c_x, c_y , KF_sp_index[j] = xy_measurement(np.asscalar(KF_X[j][0]), np.asscalar(KF_X[j][2]),\
			 											KF_L[j], cen_index, sp_mat, sp_list )
			# z = np.array([[c_x, 0, 0, 0],[0, 0, c_y, 0]]).T
			z = np.array([[c_x], [c_y]])
			# print "shape of z"
			# print z.shape

			y = z - dot(KF_H[j], KF_X[j])

			# print "shape of H"
			# print KF_H[j].shape
			# print "shape of y:"
			# print y.shape
			# print "shape of X:"
			# print KF_X[j].shape
			KF_X[j] += dot(K, y)
			KF_P[j] = KF_P[j] - dot(K, KF_H[j]).dot(KF_P[j])
	form_label_mat(KF_sp_index,KF_L,sp_mat)

	# testing #
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

	for index , i  in enumerate(list1):
		seg[np.where(sp_mat == i)] = list2[index]

	out = visualization(seg)

	# testing #
	cv2.imshow("tracking",out)
	

	return out

def visualization(mat):

	output = [1,2,3]


	for i in range(3):
		output[i] = np.zeros(mat.shape)
		output[i][np.where(mat == i)] = 255

	out = np.array(cv2.merge([output[0],output[1],output[2]]), dtype = np.uint8)

	return out

main()

