# Name: Banana's components detection
# Author: Joe
# Note: I reserve all the right for the final explanation
# Genius is one percent inspiration and ninety-nine percent perspiration. 
# Without the one percent of inspiration, all the perspriation in the world 
# is only a bucket of sweat ---- I admit that I don't have the inspiration,
# so now allow me to present you my bucket of sweat. :)

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pymeanshift as pms
from sklearn.svm import SVC
from sklearn.externals import joblib
import slic
from ctypes import c_uint8
import os
from scipy.spatial import ConvexHull
from scipy import ndimage

class Joe_segmentation:

	def __init__(self, image, input_clf, number_regions=500, consistency=5):
		"""def __init__(self, image, number_regions=500, input_clf):"""
		self.original_image = cv2.imread(image)
		self.hsvimage = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
		self.number_regions = number_regions
		self.clf = joblib.load(input_clf)
		self.slic_result = slic.slic_n(self.original_image, number_regions, consistency)
		self.label_mask = np.array(self.slic_result,dtype=np.int32)

		pass

	# set random color for segmentation output
	def label_color(self):
		out = np.ones(self.original_image.shape,dtype=np.uint8)
		for i in range(self.number_regions):
			newval = np.random.randint(256, size=3)
			out[self.label_mask==i] = newval
		return out

	# find centroid for each superpixel
	def centroid_calc(self):
		list_centroid = [0]*self.number_regions
		# for i in range(self.number_regions):
		for i in range(self.number_regions):

			count = (self.slic_result == i).sum()
			# print count
			# if count == 0:
				# print "shit"
				# print i
				# pass
				
			list_centroid[i] = np.argwhere(self.slic_result==i).sum(0)/count

			# cv2.imshow("rr", np.argwhere(self.label_mask==i))
			# print np.argwhere(self.label_mask==i)
		return list_centroid

	def color_point(self,img,index):
		'''
           color_point(img,index):

           		img: the canvas you want to draw the points on
           		index: index (x,y) of  centroids of superpixels
		'''
		temp = np.ones([img.shape[0],img.shape[1]],dtype = np.uint16)
		
		for i in range(self.number_regions):
			temp[int(index[i][0]),int(index[i][1])] = 255

		out = cv2.merge((temp,temp,temp))
		out = np.array(out, dtype = np.uint8)

		result = slic.contours(out, self.slic_result)

		return result 

	
	def extract_mask(self,img,value):
		"""Help to extract the area selected by mouse or input area number"""

		# print("Extracting mask...")
		temp_mask = np.copy(img)
		temp_mask = temp_mask + 1
		temp_mask[temp_mask!=(value+1)]=0
		temp_mask[temp_mask>0]=1
		temp_mask = np.array(temp_mask,dtype=np.uint8)
		return temp_mask

	#	this is the method to define a mouse callback function. Several events are given in OpenCV documentation
	def my_mouse_callback(self,event,x,y,flags,param):

		if event==cv2.EVENT_LBUTTONDOWN:		# here event is left mouse button double-clicked
			print "Position(%d,%d):"%(x,y)
			print "%d"%label_mask[y][x]
			mask = extract_mask(label_mask,label_mask[y][x])
			output = self.original_image * cv2.merge([mask,mask,mask])
			testdata = feature_calc_save(self.hsvimage,mask)
			results = clf.predict([testdata])

			print " svm_result:"
			print results[0]
			feature_show(self.hsvimage,mask)
		
	# save the feature in two adnarray ( flesh and skin ) 
	def feature_calc_save(self, im, roi_mask):

		histh = cv2.calcHist([im],[0],roi_mask,[15],[0,180])
		hists = cv2.calcHist([im],[1],roi_mask,[20],[0,256])
		histv = cv2.calcHist([im],[2],roi_mask,[20],[0,256])

		cv2.normalize(histh,histh,0,1,cv2.NORM_MINMAX)
		cv2.normalize(hists,histv,0,1,cv2.NORM_MINMAX)
		cv2.normalize(histv,hists,0,1,cv2.NORM_MINMAX)

		feature_vector = np.append(histh,hists)
		# feature_vector = np.append(feature_vector,histv)


		return feature_vector


	def feature_show(self, im, roi_mask):
		'''input: rgb image and mask
		   output: histogram of the roi area 
		''' 
		histh = cv2.calcHist([im],[0],roi_mask,[15],[0,180])
		hists = cv2.calcHist([im],[1],roi_mask,[20],[0,256])
		histv = cv2.calcHist([im],[2],roi_mask,[20],[0,256])
		# normalize the feature to 0,1. it might cause the problems of under fit
		cv2.normalize(histh,histh,0,1,cv2.NORM_MINMAX)
		cv2.normalize(hists,hists,0,1,cv2.NORM_MINMAX)
		cv2.normalize(histv,histv,0,1,cv2.NORM_MINMAX)
		plt.figure()
		plt.text(17,1,"h:red\ns:green\ni:blue")
		plt.plot(histh,color = 'r')
		plt.plot(histv,color = 'g')
		plt.plot(hists,color = 'b')
		plt.xlim([0,20])
		plt.show()

		feature_vector = np.append(histh,hists)
		# feature_vector = np.append(feature_vector,histv)


		return feature_vector


	def test_click(self):
		"""test_click(o_im,label_mask,number_regions)"""
		output_mask = self.label_mask.copy()
		for i in range(self.number_regions):
			mask = extract_mask(self.label_mask,i)
			testdata = feature_calc_save(self.hsvimage,mask)
			results= self.clf.predict([testdata])
			output_mask[np.where(self.label_mask == i)] = results[0]
			print results


		output = [1,2,3]

		for i in range(3):
			out_put = np.zeros(self.label_mask.shape)
			out_put[np.where(output_mask == i)] = 255
			output[i] = out_put
		out = cv2.merge([output[0],output[1],output[2]])

		while(1):
			cv2.namedWindow("Display",1)
			cv2.setMouseCallback("Display",my_mouse_callback,self.segment_result)	#binds the screen,function and image
			cv2.imshow("Display",self.segment_result)
			cv2.imshow("out",out)
			cv2.imshow("original",self.original_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			if cv2.waitKey(15)%0x100==27:
			    cv2.destroyAllWindows()
			    break	



	def test(self):
		output_mask = self.label_mask.copy()

		for i in range(self.number_regions):
			mask = self.extract_mask(self.label_mask, i)
			testdata = self.feature_calc_save(self.hsvimage,mask)
			results= self.clf.predict([testdata])
			output_mask[np.where(self.label_mask == i)] = results[0]

		output = [1,2,3]

		for i in range(3):
			out_put = np.zeros(self.label_mask.shape)
			out_put[np.where(output_mask == i)] = 255
			output[i] = out_put

		out = cv2.merge([output[0],output[1],output[2]])
		cv2.namedWindow("out_put",cv2.WINDOW_NORMAL)
		cv2.namedWindow("original",cv2.WINDOW_NORMAL)
		cv2.imshow("original",self.original_image)
		cv2.imshow("out_put",out)
		cv2.imwrite("svm_output.jpg",out)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def segmentation(self):
		output_mask = self.label_mask.copy()
		list_sp = np.unique(output_mask)
		list_seg = list_sp.copy()
		# the size of list_seg_pro woould become 1500: not correct way to  !!!
		# list_seg_pro = [0.,0.,0.]*self.number_regions
		list_seg_pro = [0]*self.number_regions

		for index, i in enumerate(list_sp):
			mask = self.extract_mask(self.label_mask, i)
			testdata = self.feature_calc_save(self.hsvimage,mask)
			results= self.clf.predict([testdata])
			list_seg[index] = results 
			list_seg_pro[index] = self.clf.predict_proba([testdata])
			output_mask[np.where(self.label_mask == i)] = results[0]

		output = [1,2,3]


		for i in range(3):
			output[i] = np.zeros(self.label_mask.shape)
			output[i][np.where(output_mask == i)] = 255
			# output[i] = out_put

		out = cv2.merge([output[0],output[1],output[2]])
		return out, list_seg, list_seg_pro
		# return output_mask

	def segmentation_tsp(self,tsp_mask):

		output_mask = tsp_mask.copy()
		list_sp = np.unique(output_mask)
		list_seg = list_sp.copy()
		print " list_sp:\n"
		# print list_seg
		print list_seg.shape
		list_seg_pro = [0]*len(list_sp)


		for index, i in enumerate(list_sp):
			mask = self.extract_mask(tsp_mask, i)
			testdata = self.feature_calc_save(self.hsvimage,mask)
			results = self.clf.predict([testdata])
			list_seg_pro[index] = self.clf.predict_proba([testdata])
			list_seg[index] = results 
			output_mask[np.where(tsp_mask == i)] = results[0]

		output = [1,2,3]


		for i in range(3):
			output[i] = np.zeros(tsp_mask.shape)
			output[i][np.where(output_mask == i)] = 255
			# output[i] = out_put

		out = cv2.merge([output[0],output[1],output[2]])

		return out, list_seg, list_seg_pro
		# return output_mask

	# not working
	def drawcontour(self, img, mask):
		borders = np.zeros(mask.shape)

		list_sp = np.unique(mask)

		for i in list_sp:
			temp_mask = self.extract_mask(mask, i)
			hull = cv2.convexHull(temp_mask)
			print hull
			borders[np.where(hull)] = 1

		r,g,b = cv2.split(img)
		for i in range(3):
			# this might cause some time waste
			r[np.where(borders == 1)] = 255
			g[np.where(borders == 1)] = 255
			b[np.where(borders == 1)] = 255
		img = cv2.merge((r,g,b))
		cv2.imshow("contours",img)
		cv2.waitKey(0)

		return borders

# flesh = 0 skin = 1 other =2
def svm_training(sk_feature, fl_feature, ot_feature,data_size=150):
	"""generate clf of svm"""

	total_size = data_size * 3

	sk = np.load(sk_feature)
	fl = np.load(fl_feature)
	ot = np.load(ot_feature)
	temp_sk = range(data_size)
	temp_fl = range(data_size)
	temp_ot = range(data_size)
	out = range(total_size)
	for i in temp_sk:
		temp_sk[i] = sk[i]


	for i in temp_fl:
		temp_fl[i] = fl[i]

	for i in temp_ot:
		temp_ot[i] = ot[i]

	temp_sk = np.array(temp_sk)
	temp_fl = np.array(temp_fl)
	temp_ot = np.array(temp_ot)

	sk_label = np.ones((data_size,),dtype = np.int)
	fl_label = np.ones((data_size,),dtype = np.int)-1
	ot_label = np.ones((data_size,),dtype = np.int)+1

	# flesh = 0 skin = 1 other =2

	index = np.append([sk_label],[fl_label])
	index = np.append(index,[ot_label])

	# combine above label together we have label_axis for svm

	X = np.append([temp_sk],[temp_fl],axis=1)
	X = np.append([X[0]],[temp_ot],axis=1)
	X = X[0]

	# append above together we have feature axis for svm

	X = np.array(X)

	clf = SVC(C=1,kernel='rbf',gamma='auto',probability=True,decision_function_shape='ovo')
	clf.fit(X,index)
	joblib.dump(clf, './svmfile.pkl')
	return clf

def test_something_wrong():
	# c = svm_training('./feature/outfile_sk.npy', './feature/outfile_fl.npy', './feature/outfile_ot.npy')
	a = Joe_segmentation('./frame/00020.jpg', './svmfile.pkl')
	result = np.array(a.segmentation(),dtype=np.uint8)
	print result

	# slic labels dtype will cuase buffer mismatch if it's changed into other type i.e. uint8 rather than original uint8_t(according to .dtype = int32)
	region_labels = a.slic_result

	im = np.array(a.original_image, dtype = np.uint16)
	sum_temp = np.add(result,im)
	im[np.where(sum_temp > 255)] = 255
	im = np.array(im, dtype = np.uint8)

	contours = slic.contours(result, region_labels,1)

	cv2.imshow("result", contours)
	cv2.waitKey(0)

	return True

def generate_semantic_segmantation_result():
	output_loc = "./segmentation_result/"
	try:
		os.mkdir(output_loc)
	except OSError:
		pass
	range(100)
	for i in range(100):
		print i
		a = Joe_segmentation('./frame/%#05d.jpg'%(i+1), './svmfile.pkl')
		result = np.array(a.segmentation(),dtype=np.uint8)
		region_labels = a.slic_result
		contours = slic.contours(result, region_labels)

		cv2.imwrite(output_loc + "/%#05d.jpg"%(i+1), contours)




if __name__ == "__main__":


	# cv2.waitKey(0)


	# hist + slic + svm : segmente banana
	# 1: training new svm model 2&3: use existed model to predict
	# c = svm_training('./feature/outfile_sk.npy', './feature/outfile_fl.npy', './feature/outfile_ot.npy')

	# a = Joe_segmentation('opticalfb.png', './svmfile.pkl')
	a = Joe_segmentation('./frame_peeling/00071.jpg', './svmfile.pkl')
	
	result,temp1,temp2 = a.segmentation()
	cv2.imshow("result", result)
	cv2.waitKey(0)

	generate_semantic_segmantation_result()


	pass