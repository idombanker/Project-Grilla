from Joe_segmentation import Joe_segmentation
import scipy.io 
import numpy as np
import os
# a = Joe_segmentation('./frame/00019.jpg', './svmfile.pkl')
# seg_labels = a.segmentation()

# mat = scipy.io.loadmat("/Users/zhufucai/Documents/GitHub/TSP_matlab/results/sp_labels_banana.mat")
# sp_labels = mat['sp_labels'][:,:,19]

# print seg_labels.shape

output_loc = "./segmentation_result/"
try:
	os.mkdir(output_loc)
except OSError:
	pass

temp = np.zeros((272,480,100))

# print seg_mat.shape
for i in range(100):
	print i
	a = Joe_segmentation('./frame/%#05d.jpg'%(i+1), './svmfile.pkl')
	result = a.segmentation()
	temp[:,:,i] = result

np.save(output_loc + "/seg_mat.npy",temp)
	# np.save(output_loc + "/%#05d.npy"%(i+1),result)
