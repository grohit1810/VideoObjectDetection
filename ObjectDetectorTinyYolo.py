# -*- coding: utf-8 -*-
"""
Created on Sat Apr 04 02:25:59 2020

@author: grohit
"""

import cv2
import os
import numpy as np
import time
from os import walk
import pickle
import itertools
imageFileNames = []
for (dirpath, dirnames, filenames) in walk("imageDir"):
    imageFileNames.extend(filenames)
    break
class TinyYolo:
    
    def __init__(self, yolo_weights = "yolov3-tiny.weights", yolo_config = "yolov3-tiny.cfg"):
        self.net = cv2.dnn.readNet(yolo_weights,yolo_config) #Tiny Yolo
        self.classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.COLORS= np.random.uniform(0,255,size=(len(self.classes),3))
        self.layer_names = self.net.getLayerNames()
        self.outputlayers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.font = cv2.FONT_HERSHEY_PLAIN
        
    def crop_img_and_save(self, img, class_id, x, y, x_plus_w, y_plus_h, filename_to_save):
        if str(self.classes[class_id]) == "car":
            height,width,channels = img.shape
            x,y,x_plus_w, y_plus_h = max(0,x), max(0,y), max(0,x_plus_w), max(0,y_plus_h)
            x,y,x_plus_w, y_plus_h = min(360,x), min(288,y), min(360,x_plus_w), min(288,y_plus_h)
            crop_img = img[y:y_plus_h, x:x_plus_w]
            cv2.imwrite(filename_to_save, crop_img)
            
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = (xB - xA) * (yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def create_dir_if_not_exits(self, dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass

    def remove_overlapping_boxes(self, class_ids, confidences, boxes):
        boxes_comb = list(itertools.combinations(boxes, 2))
        remove_list = []
        for current_comb in boxes_comb:
            boxA,boxB = current_comb
            if self.bb_intersection_over_union(boxA, boxB) > 0.2:
                confA = confidences[boxes.index(boxA)]
                confB = confidences[boxes.index(boxB)]
                if confA <= confB:
                    remove_list.append(boxA)
                else:
                    remove_list.append(boxB)
        remove_list.sort()
        remove_list = list(remove_list for remove_list,_ in itertools.groupby(remove_list))
        for current_box in remove_list:
            remove_index = boxes.index(current_box)
            class_ids.pop(remove_index)
            confidences.pop(remove_index)
            boxes.pop(remove_index)
    
    def classify_car_image(self, filename):
        img = cv2.imread(filename)
        height,width,channels = img.shape
        blob = cv2.dnn.blobFromImage(img, (1.0/255.0), (416,416), (0,0,0), True, crop=False) 
        self.net.setInput(blob)
        outs = self.net.forward(self.outputlayers)
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #object detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    if(class_id == 2):
                        boxes.append([x,y,w,h]) #put all rectangle areas
                        confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                        class_ids.append(class_id) #name of the object tha was detected
        self.remove_overlapping_boxes(class_ids, confidences, boxes)
        return class_ids, confidences, boxes
    
    def bound_and_crop_image(self, imageFileName, yoloDirName = "TinyYoloOutput", croppedDirName = "CropImageOutput"):
        orig_image = cv2.imread(imageFileName)
        bound_image = cv2.imread(imageFileName)
        self.create_dir_if_not_exits(yoloDirName)
        self.create_dir_if_not_exits(croppedDirName)
        filename = imageFileName[imageFileName.rfind('/')+1:]
        class_ids, confidences, boxes = self.classify_car_image(imageFileName)
        count = 0 
        yoloFilename = yoloDirName + "/"+ filename
        for i in range(len(class_ids)):
            count+=1
            box = boxes[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cropFilename = croppedDirName+"/"+str(count) + "_"+ filename
            self.draw_bounding_box(bound_image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            self.crop_img_and_save(orig_image, class_ids[i], round(x), round(y), round(x+w), round(y+h), cropFilename)
        cv2.imwrite(yoloFilename, bound_image)
    
    def __del__(self):
        # release resources
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    tinyYolo = TinyYolo()
    for fileName in imageFileNames:
        imageFile = 'imageDir/' + fileName
        tinyYolo.bound_and_crop_image(imageFile)