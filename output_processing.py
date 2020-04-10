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
        frame_car_map = defaultdict(lambda: defaultdict(int))
        for key in predictions_dict:
            frame_num = key[7: -4]
            frame_count_map[frame_num] += 1
            predicted_class = predictions_dict[key]
            frame_car_map[frame_num][classes[predicted_class]] += 1
        
        self.frame_count_map = frame_count_map
        self.frame_car_map = frame_car_map
        
    def process_predictions_pkl(self):
        pred_frame_count_map = self.frame_count_map
        y_count_true = []
        y_count_predicted = []
        
        tphb = 0
        tpsd = 0
        tphb = 0
        fphb = 0
        fnhb = 0
        fpsd = 0
        fnsd = 0

        with open(current_dir + "\Ground_Truth.csv") as gt_csv:
            reader = csv.reader(gt_csv, delimiter=' ', quotechar='/')
            next(reader)
            next(reader)
            #print(self.frame_car_map)
            for row in reader:
                line = row[0].split(',')
                frame_num = line[0]
                total_cars = int(line[11])
                y_count_true.append(total_cars)
                try:
                    y_count_predicted.append(pred_frame_count_map[frame_num])
                except Exception:
                    y_count_predicted.append(0)
                
                # Calculations for Q2
                if total_cars > 0:
                    car_indexes = [ i for i, x in enumerate(line[: -1]) if x == '1']
                    for index in car_indexes:
                        predicted = self.frame_car_map[frame_num]
                        is_true_hatchback = index >= 6
                        is_true_sedan = index < 6 and index > 0
                        predicted_hatchback = predicted["hatchback"]
                        predicted_sedan = predicted["sedan"]
                        
                        if is_true_hatchback and predicted_hatchback > 0:
                            tphb += 1
                        if is_true_hatchback and predicted_hatchback == 0:
                            fnhb += 1
                        if not is_true_hatchback and predicted_hatchback > 0:
                            fphb += 1
                        
                        if is_true_sedan and predicted_sedan > 0:
                            tpsd += 1
                        if is_true_sedan and predicted_sedan == 0:
                            fnsd += 1
                        if not is_true_sedan and predicted_sedan > 0:
                            fpsd += 1
        
        precisionhb = tphb / (tphb + fphb)
        precisionsd = tpsd / (tpsd + fpsd)
        recallhb = tphb / (tphb + fnhb)
        recallsd = tpsd / (tpsd + fnsd)
        f1_hb = 2 * precisionhb * recallhb / (precisionhb + recallhb)
        f1_sd = 2 * precisionsd * recallsd / (precisionsd + recallsd)        
        f1_model = 1 - ((f1_hb + f1_sd) / 2)

        print("The F1 score for Q1 is", f1_score(y_count_true, y_count_predicted, average="weighted"))
        print("The F1 score for Q2 is", f1_model)

if __name__ == "__main__":
    p = ProcessOutput()
    p.process_pkl_count()
    p.process_predictions_pkl()