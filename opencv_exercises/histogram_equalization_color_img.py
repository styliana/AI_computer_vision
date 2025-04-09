import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load color image (remove the 0 to keep color)
img = cv2.imread('data/lena.png')  
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display

# Split into channels and equalize each one
channels = cv2.split(img)
eq_channels = []

# Equalize each channel separately
for ch in channels:
    eq_channels.append(cv2.equalizeHist(ch))

# Merge back the equalized channels
img_eq = cv2.merge(eq_channels)

# Create figure for histograms
plt.figure(figsize=(15, 10))

# Plot original image histograms
plt.subplot(2, 2, 1)
colors = ('r', 'g', 'b')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title('Original Image Histograms')
plt.xlim([0, 256])

# Plot equalized image histograms
plt.subplot(2, 2, 2)
for i, col in enumerate(colors):
    hist = cv2.calcHist([img_eq], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title('Equalized Image Histograms')
plt.xlim([0, 256])

# Show original and equalized images
plt.subplot(2, 2, 3)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_eq)
plt.title('Equalized Image')
plt.axis('off')

plt.tight_layout()

# OpenCV display with window keeping ratio
cv2.namedWindow('Color Image Comparison', cv2.WINDOW_KEEPRATIO)
combined = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 
                     cv2.cvtColor(img_eq, cv2.COLOR_RGB2BGR)])
cv2.imshow('Color Image Comparison', combined)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()