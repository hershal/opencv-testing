import cv2
import numpy as np
import heapq

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    # for gray in cv2.split(img):
    for gray in [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]:
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

cap = cv2.VideoCapture(0)

while(1):
    img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # edges = cv2.Canny(gray, )
    edges = find_squares(img)
    edges = heapq.nlargest(2, edges, key=cv2.contourArea)
    edges = [edges[1]]
    print edges
    cv2.drawContours(img, edges, -1, (0, 255, 0), 3)

    actualboard = np.float32([[400,0], [0, 0], [0, 300], [400, 300]])
    sensedboard = np.float32(edges)
    persp_M = cv2.getPerspectiveTransform(sensedboard, actualboard)
    print persp_M

    persp_img = cv2.warpPerspective(img, persp_M, (300,300))

    cv2.imshow('original', img)
    cv2.imshow('result', persp_img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
