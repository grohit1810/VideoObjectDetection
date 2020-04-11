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
    
    def process_color_pkl(self):
        color_dict = pickle.load(open(current_dir + "\color_detection.pkl", "rb"))
        #print(color_dict)
        frame_color_map = defaultdict(lambda: defaultdict(int))
        for key in color_dict:
            frame_num = key[7: -4]
            frame_color_map[frame_num][color_dict[key]] += 1
        print(frame_color_map)
        self.frame_color_map = frame_color_map
        
    def get_color_from_position(self, position):
        positions = {
                1: "Black",
                2: "Silver",
                3: "Red",
                4: "White",
                5: "Blue",
                6: "Black",
                7: "Silver",
                8: "Red",
                9: "White",
                10: "Blue"
            }
        return positions[position]
    
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
        
        correctly_predicted_colors = 0
        total_colors = 0
        
        with open(current_dir + "\Output.csv", "w", newline='') as writercsv:
            field_names = ["Frame No", "Count", "Type1", "Type2", "Color1", "Color2"]
            writer = csv.DictWriter(writercsv, fieldnames=field_names)
            writer.writeheader()

            with open(current_dir + "\Ground_Truth.csv") as gt_csv:
                write_dict = {}
                reader = csv.reader(gt_csv, delimiter=' ', quotechar='/')
                next(reader)
                next(reader)
                #print(self.frame_car_map)
                for row in reader:
                    write_dict_row = {}
                    line = row[0].split(',')
                    frame_num = line[0]
                    total_cars = int(line[11])
                    y_count_true.append(total_cars)
                    try:
                        y_count_predicted.append(pred_frame_count_map[frame_num])
                        if (pred_frame_count_map[frame_num] > 0):    
                            write_dict_row["Frame No"] = frame_num
                            write_dict_row["Count"] = pred_frame_count_map[frame_num]
                    except Exception:
                        y_count_predicted.append(0)
                    
                    # Calculations for Q2 and Q3
                    if total_cars > 0:
                        car_indexes = [ i for i, x in enumerate(line[: -1]) if x == '1']
                        car_indexes = [x for x in car_indexes if x!= 0]
                        predicted_colors = self.frame_color_map[frame_num]

                        predicted = self.frame_car_map[frame_num]
 
                        for color in predicted_colors.keys():
                            if predicted_colors[color] > 1:
                                write_dict_row["Color1"] = color
                                write_dict_row["Color2"] = color
                            else:
                                write_dict_row["Color1"] = color
                        
                        for car in predicted.keys():
                            if predicted[car] > 1:
                                write_dict_row["Type1"] = car
                                write_dict_row["Type2"] = car
                            else:
                                write_dict_row["Type1"] = car
                                
                        if (write_dict_row.keys()):
                            writer.writerow(write_dict_row)
                            write_dict[frame_num] = write_dict_row
    
                        for index in car_indexes:
                            actual_color = self.get_color_from_position(index)
                            
                            if (actual_color in predicted_colors.keys()):
                                correctly_predicted_colors += 1
                            total_colors += 1
    
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
            
            pickle.dump(write_dict, open(current_dir + "\output.pkl", "wb"))
            precisionhb = tphb / (tphb + fphb)
            precisionsd = tpsd / (tpsd + fpsd)
            recallhb = tphb / (tphb + fnhb)
            recallsd = tpsd / (tpsd + fnsd)
            f1_hb = 2 * precisionhb * recallhb / (precisionhb + recallhb)
            f1_sd = 2 * precisionsd * recallsd / (precisionsd + recallsd)        
            f1_model = 1 - ((f1_hb + f1_sd) / 2)
    
            print("The F1 score for Q1 is", f1_score(y_count_true, y_count_predicted, average="weighted"))
            print("The F1 score for Q2 is", f1_model)
            print("Accuracy for Q3 is", correctly_predicted_colors / total_colors)

if __name__ == "__main__":
    p = ProcessOutput()
    p.process_color_pkl()
    p.process_pkl_count()
    p.process_predictions_pkl()