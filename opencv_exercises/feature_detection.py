import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('./data/lena.png')

# Initialize AKAZE feature detector
akaze = cv2.AKAZE_create()

# Detect keypoints and compute descriptors
keypoints, descriptor = akaze.detectAndCompute(img, None)

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(
    img, keypoints, None, (255, 0, 0),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display the result
cv2.imshow('AKAZE Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()