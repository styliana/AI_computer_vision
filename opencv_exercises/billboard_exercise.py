import cv2
import numpy as np

# Mouse callback function ONCLICK YEAAHHH
def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(dst_points) < 4:
            dst_points.append([x, y])
            cv2.circle(img_copy, (x, y), 15, (0, 0, 255), -1)
            cv2.imshow('Click', img_copy)

# Load the image
base_img = cv2.imread('data/billboard.jpg')
img_copy = base_img.copy()
img2 = cv2.imread('data/ezio.jpg')

# Store imgs data
base_h, base_w = base_img.shape[:2]
img2_h, img2_w = img2.shape[:2]

# Create source and destination points
src_points = np.float32([
    [0, 0],
    [0, img2_h],
    [img2_w, img2_h],
    [img2_w, 0]
])

dst_points = []

# Define the window that allows us to click on the billboard
cv2.namedWindow('Click', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('Click', onClick)

# Show the img to be clicked
cv2.imshow('Click', base_img)
cv2.waitKey(0)

# Check if we have exactly 4 points
if len(dst_points) == 4:
    # Compute the homography matrix
    dst_float = np.float32(dst_points)
    H = cv2.getPerspectiveTransform(src_points, dst_float)

    # Apply the transformation to the image that we want over the billboard
    warped = cv2.warpPerspective(img2, H, (base_w, base_h))

    # Create a mask for the warped image
    mask = np.zeros_like(base_img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_float), (255, 255, 255))

    # Combine the warped image with the base image
    result = cv2.bitwise_and(base_img, cv2.bitwise_not(mask))
    result = cv2.bitwise_or(result, warped)

    # Show the result
    cv2.namedWindow('Result', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("You need to click exactly 4 points on the base image.")


'''
cv2.namedWindow('baseimg',cv2.WINDOW_KEEPRATIO)
cv2.imshow('baseimg',base_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()'
'''