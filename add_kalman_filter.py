from Joe_segmentation import Joe_segmentation
import cv2 
import numpy as np
from sklearn.svm import SVC
import time

start = time.clock()
a = Joe_segmentation('./frame/00094.jpg', './svmfile.pkl')
result = a.segmentation()

cv2.imshow("result", result)
t = time.clock() - start
print "runing time: %f"%t
cv2.waitKey(0)

def main():
	sk_f = "./feature/outfile_sk.npy"
	fl_f = "./feature/outfile_fl.npy"
	ot_f = "./feature/outfile_ot.npy"
	svm_calculate_accuracy(sk_f, fl_f, ot_f)


def svm_calculate_accuracy(sk_feature, fl_feature, ot_feature,data_size=150):
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

	clf = SVC(C=1.2,kernel='rbf',gamma='auto',probability=True,decision_function_shape='ovo')
	clf.fit(X,index)
	accuracy = clf.score(X,index)
	print X.shape
	print "the accuracy of the svm is : %f"%accuracy
	# joblib.dump(clf, './svmfile.pkl')
	return clf
