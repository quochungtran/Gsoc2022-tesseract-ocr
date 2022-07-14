from ast import withitem
from copy import copy
import cv2 
import pytesseract
import numpy as np
import imutils
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv
import os
import re
from utils import *


class PreProcessing_engine:

    def __init__(self, image) -> None:    

        self.image     = image 
        self.img_pre   = image 

    # set current image  
    def set_image(self, img):
        self.image = img

    # get current image
    def get_image(self): 
        return self.image
    
    # get current preprocessing image
    def get_image_proccesing(self):
        return self.img_pre 
    
    # get grayscale image
    def get_grayscale(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    # noise removal
    def remove_noise(self):
        return cv2.GaussianBlur(self.image,(3,3),0)
    
    #thresholding
    def thresholding(self):
        #return cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,  cv2.THRESH_BINARY, 35, 2)
        #return cv2.adaptiveThreshold(self.image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) 
        blur = cv2.GaussianBlur(self.image,(3,3),0)
        return  cv2.threshold(self.image ,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #dilation
    def dilate(self):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(self.image, kernel, iterations = 1)
    
    #erosion
    def erode(self):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(self.image, kernel, iterations = 1)
    
    #opening - erosion followed by dilation
    def opening(self):
        kernel = np.ones((1,1),np.uint8)
        return cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
    
    #canny edge detection
    def canny(self):
        return cv2.Canny(self.image, 100, 200)
    
    #skew correction
    def deskew(self):
        rotated = (self.image).copy()            
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        thresh = cv2.threshold(gray, 0, 255,
    	     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1] 
        angle = -angle
        (h, w) = rotated.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(rotated, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)     

        return rotated

    def rotate_4_oritation(self):
        
        rotated = self.image.copy()
        (h, w) = rotated.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(rotated, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)     

        return rotated
        


    # remove shadow
    def remove_shadow(self):
        
        rgb_planes = cv2.split(self.image)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((3,3), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

            result = cv2.merge(result_planes)
            result_norm = cv2.merge(result_norm_planes)
        return result

    # correct distortion
    def correct_distortion(self) :

        gray  = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        # show the original image and the edge detected image
        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv2.findContours(edged.copy(), 
                                cv2.RETR_LIST, 
                                cv2.CHAIN_APPROX_SIMPLE)
        
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen

            if len(approx) == 4:
                screenCnt = approx
                break
        
        # show the contour (outline) of the piece of paper
        
        return cv2.drawContours(self.image, [screenCnt], -1, (0, 255, 0), 2)
        
    

def Preprocessing_img_pipeline(img,
                               types):

    pre_eng = PreProcessing_engine(img)
    im = pre_eng.get_image()
    
    for type in types :
     
            if type ==  "gray" : 
                im  = pre_eng.get_grayscale()
                pre_eng.set_image(im)
            if type == "noise":    
                im = pre_eng.remove_noise()
                pre_eng.set_image(im)
            if type == "binary":
                im = pre_eng.thresholding()
                pre_eng.set_image(im)
            if type == "dilation":                
                im = pre_eng.dilate()
                pre_eng.set_image(im)
            if type == "erosion":
                im = pre_eng.erode()
                pre_eng.set_image(im)
            if type == "opening": 
                im = pre_eng.opening()
                pre_eng.set_image(im)
            if type == "canny": 
                im = pre_eng.canny()  
                pre_eng.set_image(im)
            if type == "deskew" : 
                im = pre_eng.deskew()  
                pre_eng.set_image(im)
            if type == "scanner":                 
                im = pre_eng.correct_distortion()  
                pre_eng.set_image(im)
            if type == "remove_shadow":
                im = pre_eng.remove_shadow()
                pre_eng.set_image(im)
    
    #visualize(pre_eng.get_image())
    return pre_eng.get_image()


