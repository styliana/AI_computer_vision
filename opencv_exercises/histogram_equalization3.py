import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/lena.png',0) #0 for greyscale

eq_channels = []
channels = cv2.split(img)
img_eq = cv2.equalizeHist(img)

color = ('b','g','r')
for i, col in enumerate(color):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])

    #plot the histogram
    plt.plot(hist, color=col)
plt.xlim([0,256])
plt.show()
