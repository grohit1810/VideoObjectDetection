# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 01:21:17 2020

@author: Raksha
"""
import pickle
import csv
import os
from sklearn.metrics import f1_score
from collections import defaultdict

classes = ["hatchback", "sedan"]
colors = ["black", "silver", "red", "white", "blue"]
#%%
current_dir = os.path.dirname(os.path.realpath(__file__))

class ProcessOutput:
    def process_pkl_count(self):
        predictions_dict = pickle.load(open(current_dir + "\predictions.pkl", "rb"))
        frame_count_map = defaultdict(int)
        for key in predictions_dict:
            frame_num = key[7: -4]
            frame_count_map[frame_num] += 1
        self.frame_count_map = frame_count_map
        
        
    def process_predictions_pkl(self):
        predictions_dict = pickle.load(open(current_dir + "\predictions.pkl", "rb"))
        pred_frame_count_map = self.frame_count_map
        tiny_yolo_pkl = pickle.load(open(current_dir + "\TinyYolo.pkl", "rb"))
        no_of_frames_not_in_pred = 0
        y_count_true = []
        y_count_predicted = []
        gt_predictions_dict = {}
        sedan_count = 0
        hatchback_count = 0
        actual_num_cars = 0
        predicted_num_cars = 0

        with open(current_dir + "\Ground_Truth.csv") as gt_csv:
            reader = csv.reader(gt_csv, delimiter=' ', quotechar='/')
            next(reader)
            next(reader)
            for row in reader:
                line = row[0].split(',')
                frame_num = line[0]
                total_cars = int(line[11])
                y_count_true.append(total_cars)
                try:
                    y_count_predicted.append(pred_frame_count_map[frame_num])
                except Exception:
                    y_count_predicted.append(0)
        print("The F1 score for Q1 is", f1_score(y_count_true, y_count_predicted, average="weighted"))

if __name__ == "__main__":
    p = ProcessOutput()
    p.process_pkl_count()
    p.process_predictions_pkl()