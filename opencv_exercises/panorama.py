import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('data/right.png')
img2 = cv2.imread('data/left.png')

# Initialize ORB feature detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kpts1, desc1 = orb.detectAndCompute(img1, None)
kpts2, desc2 = orb.detectAndCompute(img2, None)

# Initialize BFMatcher for feature matching
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(desc1, desc2, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)

# Ensure there are enough good matches to compute homography
if len(good_matches) > 4:
    src_points = np.float32([kpts1[m.queryIdx].pt for m in good_matches])
    dst_points = np.float32([kpts2[m.trainIdx].pt for m in good_matches])

    # Compute homography matrix
    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Warp the first image to align with the second
    panorama_width = img1.shape[1] + img2.shape[1]
    panorama_height = max(img1.shape[0], img2.shape[0])
    panorama = cv2.warpPerspective(img1, M, (panorama_width, panorama_height))

    # Blend the second image into the panorama
    panorama[0:img2.shape[0], 0:img2.shape[1]] = img2

    # Display the result with WINDOW_KEEPRATIO flag
    cv2.namedWindow('Panorama', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough good matches to create a panorama.")