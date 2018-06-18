from Joe_segmentation import Joe_segmentation
import cv2
import numpy as np


def kmeans(img):
	'''
		input: image
		output: clusterting visualization , cluster matrix
	'''
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 2
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	# temp is the cluster mat
	temp = np.ones((res2.shape[0],res2.shape[1]),dtype = int)
	for index , j in enumerate(np.unique(res2[:,:,1])):
		temp[np.where(res2[:,:,1]==j)]= index


	return res2, temp

def combine_layer(proba_list, cluster_mat, sp_mat,label_mask):

	'''
		giving a strong pior that the moving flesh along with the manipulator is skin

		output: merge_res
	    we output the label mask and didn't use proba and sp_mat here
	'''


	flesh = 0
	skin = 1
	moving_foregroud = 1
	# remember we use the strong knowledge that cluster_mat = 0 means the foreground
	label_mask[np.where((label_mask == flesh )&(cluster_mat == moving_foregroud))] = skin
	# for index , i in enumerate(np.unique(cluster_mat)):
	return label_mask
	
	
def generate_label_from_proba(arr1,arr2):
	pass
	# return the indices 
	temp = np.argmax((arr1 + arr2)/2.0)
	return temp

def main():

	count = 66
	frame1 = Joe_segmentation('./frame_peeling/000%d.jpg'%count, './svmfile.pkl')

	prvs = cv2.cvtColor(frame1.original_image,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1.original_image)
	hsv[...,1] = 255

	while True:
		count += 1
		frame2 = Joe_segmentation('./frame_peeling/000%d.jpg'%count, './svmfile.pkl')
		next = cv2.cvtColor(frame2.original_image,cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		# cv2.imshow('frame2',bgr)
		# k = cv2.waitKey(30) & 0xff

		result,list_seg,list_pro = frame2.segmentation()

		cluster_res, cluster_mat = kmeans(bgr)
		combine_mask = combine_layer(list_pro,cluster_mat,frame2.label_mask,frame2.label_mask2)

		# I think here I use a better coloring algorithm
		temp_r = np.zeros(combine_mask.shape)
		temp_g = np.zeros(combine_mask.shape)
		temp_b = np.zeros(combine_mask.shape)
		temp_r[np.where(combine_mask==0)] = 255
		temp_g[np.where(combine_mask==1)] = 255
		temp_b[np.where(combine_mask==2)] = 255
		cv2.imshow("combination",cv2.merge([temp_r,temp_g,temp_b]))


		cv2.imshow('dense_optical_flow',bgr)
		cv2.imshow("original",frame2.original_image)
		cv2.imshow("result", result)
		cv2.imshow("cluster",cluster_res)

		cv2.waitKey(0)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		prvs = next
	cv2.destroyAllWindows()

main()

