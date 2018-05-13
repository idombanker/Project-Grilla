import cv2
import os

dirpath = os.getcwd()
print dirpath

im = cv2.imread("./banana.jpg")
rows, cols, channels = im.shape
im_zo = cv2.resize(im, (cols/2, rows/2), interpolation=cv2.INTER_AREA)
cv2.imshow("sss",im_zo)
cv2.imwrite("./a.jpg",im_zo)
cv2.waitKey(0)