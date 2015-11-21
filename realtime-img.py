#!/usr/bin/env python2
import sys
import os
import cv2
import numpy as np
sys.path.append(os.getcwd() + "/lib")
import compositor
import heapq

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Creating track bar
cv2.createTrackbar('h', 'result',0,179,nothing)
cv2.createTrackbar('hw', 'result',179,179,nothing)
cv2.createTrackbar('s', 'result',0,255,nothing)
cv2.createTrackbar('sw', 'result',255,255,nothing)
cv2.createTrackbar('v', 'result',0,255,nothing)
cv2.createTrackbar('vw', 'result',255,255,nothing)

frame = cv2.imread(sys.argv[1])

while(1):
    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # get info from track bar and appy to result
    hw = cv2.getTrackbarPos('hw','result')
    sw = cv2.getTrackbarPos('sw','result')
    vw = cv2.getTrackbarPos('vw','result')
    h = cv2.getTrackbarPos('h','result')
    s = cv2.getTrackbarPos('s','result')
    v = cv2.getTrackbarPos('v','result')

    # Normal masking algorithm
    lower_filter = np.array([h,s,v])
    upper_filter = np.array([hw,sw,vw])

    mask = cv2.inRange(hsv,lower_filter, upper_filter)

    comp = compositor.compositor()
    comp.push(frame)
    comp.mask(mask)

    (contours, cnts, _) = cv2.findContours(mask.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    areas = heapq.nlargest(4, cnts, key=cv2.contourArea)
    for c in areas:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        cv2.drawContours(comp.composited_img, [approx], -1, (0, 255, 0), 4)
    cv2.imshow('result', comp.composited_img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
