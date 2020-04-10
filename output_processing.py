# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 01:21:17 2020

@author: Raksha
"""
import pickle
import csv
import os
from sklearn.metrics import f1_score

classes = ["hatchback", "sedan"]
colors = ["black", "silver", "red", "white", "blue"]
#%%
current_dir = os.path.dirname(os.path.realpath(__file__))

class ProcessOutput:
    def process_predictions_pkl(self):
        predictions_dict = pickle.load(open(current_dir + "\predictions.pkl", "rb"))
        print(predictions_dict)
        tiny_yolo_pkl = pickle.load(open(current_dir + "\TinyYolo.pkl", "rb"))
        no_of_frames_not_in_pred = 0
        y_true = []
        y_predicted = []
        gt_predictions_dict = {}
        sedan_count = 0
        hatchback_count = 0

        with open(current_dir + "\Ground_Truth.csv") as gt_csv:
            reader = csv.reader(gt_csv, delimiter=' ', quotechar='/')
            next(reader)
            next(reader)
            for row in reader:
                line = row[0].split(',')
                frame_num = line[0]
                total_cars = int(line[11])
                if (total_cars > 0):
                    car_indexes = [ i for i, x in enumerate(line[1: -1]) if x == '1']
                    car_indexes = car_indexes[:1]
                    for index in car_indexes:
                        is_hatchback = index >= 5
                        is_sedan = index < 5
                        if is_sedan:
                            sedan_count += 1
                        if is_hatchback:
                            hatchback_count += 1
                        actual_class = 0 if is_hatchback else 1
                        try:
                            predicted_class = predictions_dict[frame_num]
                            y_true.append(actual_class)
                            y_predicted.append(predicted_class)
                        except Exception:
                            no_of_frames_not_in_pred += 1
        
        print(no_of_frames_not_in_pred)
        print(f1_score(y_true, y_predicted, average="weighted"))

if __name__ == "__main__":
    p = ProcessOutput()
    p.process_predictions_pkl()