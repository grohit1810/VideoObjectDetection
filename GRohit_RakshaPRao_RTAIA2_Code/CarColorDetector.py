# -*- coding: utf-8 -*-
"""
Created on Sat Apr 04 02:29:05 2020

@author: grohit
"""

import cv2
import numpy as np
from os import walk

class ColorDetection:
    
    def __init__(self):
        #the rgb value of the target colours
        self.TARGET_COLORS = {"Red": (255, 0, 0), "White": (255, 255, 255), "Blue": (0, 0, 255), "Black":(0,0,0), "Silver":(192,192,192)}
    
    def color_difference (self,color1, color2):
        #get absolute difference between the rgb value of the input colors
        return sum([abs(component1-component2) for component1, component2 in zip(color1, color2)])
    
    def get_dominant_color(self,color):
        #this function find the target color which is nearest(rgb value) to the input color
        differences = [[self.color_difference(color, target_value), target_name] for target_name, target_value in self.TARGET_COLORS.items()]
        differences.sort()
        return differences[0][1]
    
    def color_detector1(self,img):
        #uses kmeans to get the most dominant color in the image
        data = np.reshape(img, (-1,3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)
        return tuple(centers[0].astype(np.int32))
    
    def color_detector2(self,img):
        #finds the most dominant color by finding the rgb which has occured the most
        colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
        return tuple(colors[count.argmax()])
    
    def color_detector3(self,img):
        #finds the most dominant color by averaging all the rgb values in the input image
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return tuple(avg_color)
    
    def get_prominent_color(self,filename):
        #this function does a best of three of the above given three methods to find the
        #color of the car in the cropped input image
        img = cv2.imread(filename)
        vals = []
        vals.append(self.get_dominant_color(self.color_detector1(img)))
        vals.append(self.get_dominant_color(self.color_detector2(img)))
        vals.append(self.get_dominant_color(self.color_detector3(img)))
        color_value = max(set(vals), key = vals.count) 
        return color_value
    
if __name__ == "__main__":
    colDetection = ColorDetection()
    imageFileNames = []
    for (dirpath, dirnames, filenames) in walk("CropImageDir"):
        imageFileNames.extend(filenames)
        break
    for file in imageFileNames:
        img = cv2.imread('CropImageDir/' + file)
        print(colDetection.get_prominent_color('CropImageDir/' + file))