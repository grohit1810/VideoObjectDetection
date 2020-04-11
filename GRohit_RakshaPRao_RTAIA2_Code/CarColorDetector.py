# -*- coding: utf-8 -*-
"""
Created on Sat Apr 04 02:29:05 2020

@author: grohit
"""

import cv2
import numpy as np
from os import walk
imageFileNames = []
for (dirpath, dirnames, filenames) in walk("CropImageOutput"):
    imageFileNames.extend(filenames)
    break
class ColorDetection:
    
    def __init__(self):
        self.TARGET_COLORS = {"Red": (255, 0, 0), "White": (255, 255, 255), "Blue": (0, 0, 255), "Black":(0,0,0), "Silver":(192,192,192)}
    
    def color_difference (self,color1, color2):
        return sum([abs(component1-component2) for component1, component2 in zip(color1, color2)])
    
    def get_dominant_color(self,color):
        differences = [[self.color_difference(color, target_value), target_name] for target_name, target_value in self.TARGET_COLORS.items()]
        differences.sort()
        return differences[0][1]
    
    def color_detector1(self,img):
        data = np.reshape(img, (-1,3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)
        return tuple(centers[0].astype(np.int32))
    
    def color_detector2(self,img):
        colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
        return tuple(colors[count.argmax()])
    
    def color_detector3(self,img):
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return tuple(avg_color)
    
    def get_prominent_color(self,filename):
        img = cv2.imread(filename)
        vals = []
        vals.append(self.get_dominant_color(self.color_detector1(img)))
        vals.append(self.get_dominant_color(self.color_detector2(img)))
        vals.append(self.get_dominant_color(self.color_detector3(img)))
        color_value = max(set(vals), key = vals.count) 
        return color_value
    
if __name__ == "__main__":
    colDetection = ColorDetection()

    for file in imageFileNames:
        img = cv2.imread('CropImageOutput/' + file)
        print(colDetection.get_prominent_color('CropImageOutput/' + file))