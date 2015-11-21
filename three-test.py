#!/usr/bin/env python2
import cv2
import numpy as np
import pdb

img = cv2.imread("three.jpg")

lower = np.array([0, 120, 140])
upper = np.array([15, 201, 255])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, lower, upper)
(contours, cnts, _) = cv2.findContours(mask.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

c = max(cnts, key=cv2.contourArea)

pdb.set_trace()

peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.05 * peri, True)

masked_result = cv2.bitwise_and(img, img, mask = mask)

cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
cv2.imshow("result", img)
cv2.waitKey(0)


