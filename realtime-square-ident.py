#!/usr/bin/env python2
import cv2
import numpy as np
import heapq
import pdb

def nothing(x):
    pass

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
    # for gray in [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]:
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=3)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.025*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt) and cv2.contourArea(cnt) < (img.shape[0] * img.shape[1])*.95:
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def orientsquare(cont):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow('controls')
cv2.createTrackbar('h', 'controls',0,179,nothing)
cv2.createTrackbar('hw', 'controls',179,179,nothing)
cv2.createTrackbar('s', 'controls',0,255,nothing)
cv2.createTrackbar('sw', 'controls',255,255,nothing)
cv2.createTrackbar('v', 'controls',0,255,nothing)
cv2.createTrackbar('vw', 'controls',255,255,nothing)

while(1):
    _, img = cap.read()

    # --- square finding ---
    edges = find_squares(img)
    edges = (x for x in edges if lambda x: cv2.contourArea(x) < .95*imgarea)
    edges = sorted(edges, key=cv2.contourArea, reverse=True)[:1]

    persp_img = None
    for edge in edges:
        pts = edge.reshape(4,2)
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]


        actualboard = np.float32([[0,0], [400, 0], [400, 300], [0, 300]])
        persp_M = cv2.getPerspectiveTransform(rect, actualboard)
        persp_img = cv2.warpPerspective(img, persp_M, (400,300))

        cv2.drawContours(img, [edge], -1, (0, 0, 255), 3)

    # --- masking ---
    # controls
    if persp_img != None:
        hw = cv2.getTrackbarPos('hw','controls')
        sw = cv2.getTrackbarPos('sw','controls')
        vw = cv2.getTrackbarPos('vw','controls')
        h = cv2.getTrackbarPos('h','controls')
        s = cv2.getTrackbarPos('s','controls')
        v = cv2.getTrackbarPos('v','controls')

        lower = np.array([h,s,v])
        upper = np.array([hw,sw,vw])

        hsv = cv2.cvtColor(persp_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,lower, upper)
        persp_img = cv2.bitwise_and(persp_img,persp_img,mask = mask)
        (contours, cnts, _) = cv2.findContours(mask.copy(),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        areas = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
        for c in areas:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
            (x,y), radius = cv2.minEnclosingCircle(c)
            print "x: " + str(200-x) + "y: " + str(y+50) + " rad: " + str(radius)
            cv2.drawContours(persp_img, [approx], -1, (0, 255, 0), 4)
        cv2.imshow('persp', persp_img)

    cv2.imshow('original', img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
