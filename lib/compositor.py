import cv2
import numpy as np


class compositor:
    composited_img = None
    def __init__(self):
        self.composited_img = None


    def push(self, img, invert=False):
        img = img.copy()
        if invert == True:
            img = cv2.bitwise_not(img, img)
        if self.composited_img == None:
            self.composited_img = img
        else:
            self.composited_img = cv2.bitwise_and(img, self.composited_img)

    def mask(self, msk):
        self.composited_img = cv2.bitwise_and(self.composited_img, self.composited_img, mask=msk)

    def reset(self):
        composited_img = None
