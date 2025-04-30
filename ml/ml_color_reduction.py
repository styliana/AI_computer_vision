'''
program that uses unsupervised machine learning for reducing
the number of colors on an image
'''

import numpy as np  # poprawka: było "numppy"
import cv2
from sklearn.cluster import KMeans

# load the image
img_arr = cv2.imread('../data/lena.png')

# prepare the image for sklearn
(h, w, c) = img_arr.shape
img2D = img_arr.reshape(h * w, c)

# train and use the model
kmeans_model = KMeans(n_clusters=5, n_init='auto') #number of clusters in number of colors!!!!!!!!!
cluster_label = kmeans_model.fit_predict(img2D)

# convert the centroids coordinates to int
rgb_colors = kmeans_model.cluster_centers_.round(0).astype(int) 

# reconstruct the image
img_quant = rgb_colors[cluster_label].reshape(h, w, c).astype(np.uint8)

# display
cv2.imshow('Quantized Image', img_quant)
cv2.waitKey(0)
cv2.destroyAllWindows()
