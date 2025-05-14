'''
Program that cartoonize an image
'''
import cv2
import numpy as np

#img = cv2.imread('data/lena.png')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()


    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # apply a light blur for cleaning the image a bit
    img_gray = cv2.medianBlur(img_gray,5)

    # we use the laplacian filter for extracting the contours
    edges = cv2.Laplacian(img_gray,cv2.CV_8U,ksize=5)

    # threshold the edges image to get only good edges
    ret, thresholded = cv2.threshold(edges,70,255,cv2.THRESH_BINARY_INV)

    # use the bilateral filter with high values for getting the colors
    color_img = cv2.bilateralFilter(img,10,250,250)

    # put together color and sketch
    skt = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2BGR)

    # let's do the bitwise and for merging sketch and color
    output = cv2.bitwise_and(color_img,skt)

    cv2.imshow('Output',output)
    k = cv2.waitKey(10)

    if k == ord('q'):
        break