import numpy as np
import cv2

img = cv2.imread('opticalhsv.png')
# rows, cols, channels = img.shape
# img = cv2.resize(img, (cols/2, rows/2), interpolation=cv2.INTER_AREA)
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

# print np.count_nonzero(res2)
print np.where(res2>8)
res2[np.where(res2>8)] = 255
res2[np.where(res2<8)] = 0

# res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
print np.unique(res2[:,:,2])
print res2[:,:,1]
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()