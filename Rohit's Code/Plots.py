# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:06:30 2020

@author: Raksha
"""
import pickle
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

class Plots:
    def plot(self):
        obj_data = []
        car_data = []
        color_data = []
        with open("Output Csv.csv") as file:
            reader = csv.reader(file, delimiter=' ')
            next(reader)
            for row in reader:
                line = row[0].split(",")
                frame_num = int( line[0])
                car_count = int(line[1])
                obj_detection_time = float(line[7])
                
                if (car_count > 0):
                    car_detection_time = float(line[8])
                    color_detection_time = float(line[9])
                    car_data.append([frame_num, car_detection_time])
                    color_data.append([frame_num, color_detection_time])
                obj_data.append([frame_num, obj_detection_time])
            file.close()
            
        obj_data = np.asarray(obj_data)
        car_data = np.asarray(car_data)
        color_data = np.asarray(color_data)
        
        figure(figsize=(20,10))
        plt.subplot(1, 3, 1)
        plt.title("Object detection time in seconds")
        plt.plot(obj_data[:, 0], obj_data[:, 1])
        
        plt.subplot(1, 3, 2)
        plt.title("Car detection time in seconds")
        plt.plot(car_data[:, 0], car_data[:, 1])
        
        plt.subplot(1, 3, 3)
        plt.title("Color detection time in seconds")
        plt.plot(color_data[:, 0], color_data[:, 1])
        
if __name__ == "__main__":
    pl = Plots().plot()
    