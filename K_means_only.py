import numpy as np
import cv2

img = cv2.imread('opticalhsv.png')
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
print np.unique(res2[:,:,2])
print np.unique(res2[:,:,1])
print res2.shape

temp = np.ones((res2.shape[0],res2.shape[1]),dtype = int)

for j in np.unique(res2[:,:,1]):
	temp[np.where(res2[:,:,1]==j)]=j
# temp[np.where(res2[:,:,1]==43)]=1
# temp[np.where(res2[:,:,1]==128)]=2
# temp[np.where(res2[:,:,1]==204)]=3

for index, i in enumerate( np.unique(res2[:,:,1])):
	print i 
	res2[temp==i] = np.array([index*100]*3)
	# np.random.randint(256, size=3)

# orders of the cluster : from big size -----> small size
 
cv2.imshow('k = 2',res2)
cv2.imwrite("temp.jpg",res2)
cv2.waitKey(0)
cv2.destroyAllWindows()