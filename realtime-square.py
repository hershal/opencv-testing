#!/usr/bin/env python2
import cv2
import numpy as np
import heapq
import pdb

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

while(1):
    _, img = cap.read()

    edges = find_squares(img)

    edges = (x for x in edges if lambda x: cv2.contourArea(x) < .95*imgarea)
    edges = sorted(edges, key=cv2.contourArea, reverse=True)[:1]

    for edge in edges:
        pts = edge.reshape(4,2)
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]


        actualboard = np.float32([[0,0], [0, 600], [800, 600], [800, 0]])
        persp_M = cv2.getPerspectiveTransform(rect, actualboard)
        persp_img = cv2.warpPerspective(img, persp_M, (800,600))

        cv2.imshow('result', persp_img.copy())
        cv2.drawContours(img, [edge], -1, (0, 0, 255), 3)

    cv2.imshow('original', img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
