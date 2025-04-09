import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load and convert the image to grayscale
img = cv2.imread('./data/tsukuba.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply CLAHE and standard histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray_img)
eq_img = cv2.equalizeHist(gray_img)

# Display the results side by side
show_img = np.hstack([gray_img, eq_img, clahe_img])
cv2.imshow('Equalization: Original | Standard | CLAHE', show_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
'''