# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:09:01 2020

@author: Raksha
"""
import pickle
import cv2
import csv
import os
#%%

class ProcessYOLOImages:
    def __init__(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        yolo_pkl_path = data_dir + "\Code\ReseachTopicsAssignment2\TinyYolo.pkl"
        self.data = pickle.load(open(yolo_pkl_path, "rb"))
        self.images_dir = data_dir + "\Code\ReseachTopicsAssignment2\imageDir\\"
        self.sedans_dir = data_dir + "\imagesdata\yolodata\sedans\\"
        self.hatchbacks_dir = data_dir + "\imagesdata\yolodata\hatchbacks\\"
        
        self.csv_file = data_dir + "\Ground_Truth.csv"
        
        car_key = 2
        image_key = 'image'
        
        sedan_index = 1
        hatchback_index = 5
        image_index = 0
    
    def processImages(self):
        with open(self.csv_file, newline='') as file:
            reader = csv.reader(file, delimiter=' ', quotechar='|')
            next(reader)
            next(reader)
            for row in reader:
                line = row[0].split(',')
                image_name = "image" + line[image_index] + ".jpg"
                total_cars = int(line[11])
                if total_cars > 0:
                    value = self.data[image_name]
                    object_ids = value[0]
                    confidences = value[1]
                    box = value[2]
                    car_indexes = [ i for i, x in enumerate(line[1: -1]) if x == '1']
                    car_indexes = car_indexes[:1]
        
                    for index in car_indexes:
                        isHatchback = index >= hatchback_index
                        isSedan = index < hatchback_index
                        if len(object_ids) > 0 and car_key in object_ids:
                            car_indices = [i for i, x in enumerate(object_ids) if x == car_key]
                            confidences_list = [confidences[i] for i in car_indices]
                            max_confidence = max(confidences_list)
                            max_conf_index = confidences.index(max_confidence)
                            bounding_box = box[max_conf_index]
                            
                            x = bounding_box[0]
                            y = bounding_box[1]
                            w = bounding_box[2]
                            h = bounding_box[3]
                            image = cv2.imread(self.images_dir + image_name)
                            crop = image[y: y + h, x: x + w]
                        if isHatchback and len(crop[0]) > 0:
                            cv2.imwrite(self.hatchbacks_dir + image_name, crop)
                        if isSedan and len(crop[0]) > 0:
                            cv2.imwrite(self.sedans_dir + image_name, crop)

if __name__ == "__main__":
    yoloImages = ProcessYOLOImages()
    yoloImages.processImages()