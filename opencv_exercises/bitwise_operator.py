import cv2
import numpy as np

#load a color image
img = cv2.imread('data/lena.jpg')

#create a binary mask
mask = np.zeros(img.shape, dtype=np.uint8)
mask = cv2.rectangle(mask,(100,100),(250,250),(255,255,255),-1)

#apply the and operator
result = cv2.bitwise_and(img, mask)
cv2.imshow('mask',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#apply the and operator
result = cv2.bitwise_or(img, mask)
cv2.imshow('mask',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#not operator - negative colors on the image
result = cv2.bitwise_not(img, mask)
cv2.imshow('mask',result)
cv2.waitKey(0)
cv2.destroyAllWindows()