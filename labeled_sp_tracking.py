import scipy.io 
import numpy as np
import cv2 
mat = scipy.io.loadmat("/Users/zhufucai/Documents/GitHub/TSP_matlab/results/sp_labels_banana.mat")

# label0 = mat[0]
sp_labels19 = mat['sp_labels'][:,:,19]

seg_mat = np.load('/Users/zhufucai/Documents/GitHub/Project_Grilla/segmentation_result/seg_mat.npy')

seg_labels20 = seg_mat[:,:,20]


def color_seglabels(mat):

	output = [1, 2, 3]
	for i in range(3):
		output[i] = np.zeros(mat.shape)
		output[i][np.where(mat == i)] = 255
	out = cv2.merge([output[0],output[1],output[2]])

	return out

# cv2.imshow("result",color_seglabels(seg_labels20))
# cv2.waitKey(0)
def list_label_sp(mat1,mat2):
	sp = np.unique(mat1)
	# label = 
	# return 

listsp19 = list_sp(sp_labels19)




print seg_labels.shape
print sp_labels.shape
