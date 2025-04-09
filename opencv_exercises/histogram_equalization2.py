import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale mode (0)
img = cv2.imread('data/lena.png', 0)  

# Apply histogram equalization to enhance contrast
img_eq = cv2.equalizeHist(img)

# Calculate and plot histograms
plt.figure(figsize=(12, 6))

# Original image histogram
plt.subplot(1, 2, 1)
hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist_orig, color='black')
plt.title('Original Image Histogram')
plt.xlim([0, 256])

# Equalized image histogram
plt.subplot(1, 2, 2)
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
plt.plot(hist_eq, color='black')
plt.title('Equalized Image Histogram')
plt.xlim([0, 256])

plt.tight_layout()

# Display images with WINDOW_KEEPRATIO
cv2.namedWindow('Image Comparison', cv2.WINDOW_KEEPRATIO)
combined = np.hstack([img, img_eq])
cv2.imshow('Image Comparison', combined)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()