import cv2
import numpy as np

# Load the image
img = cv2.imread('data/gerry.png') 
img_copy = img.copy()

# Define points storage
src_points = []
dst_points = np.float32([[0, 0], [600, 0], [600, 800], [0, 800]])

# Mouse callback function
def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        src_points.append([x, y])
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Img', img_copy)

# Create the window that gets the clicks
cv2.namedWindow('Img', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('Img', onClick)

# Show the image for getting the clicks
cv2.imshow('Img', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get the homography matrix
src_float = np.float32(src_points)
H = cv2.getPerspectiveTransform(src_float, dst_points)

# Apply the perspective transformation
output_img = cv2.warpPerspective(img, H, (600, 800))

# Show the final image
cv2.imshow('Result', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('scanned_document.jpg', output_img)