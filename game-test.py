#!/usr/bin/env python2
# import the necessary packages
import numpy as np
import cv2

# load the games image
image = cv2.imread("lightergreen.jpg")

# BGR array
upper = np.array([100, 255, 100], dtype="uint8")
lower = np.array([0, 2, 0], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

# cv2.imshow("Image", image)
cv2.imshow("Mask", mask)
# find contours in the masked image and keep the largest one
(contours, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("contours", contours)
c = max(cnts, key=cv2.contourArea)

# approximate the contour
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.05 * peri, True)

# draw a green bounding box surrounding the red game
cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
cv2.imshow("result", image)
cv2.waitKey(0)
