# -*- coding: utf-8 -*-
"""
Created on Sat Apr 04 02:44:32 2020

@author: grohit
"""
from VideoReader import ConvertVideoToFrames
from ObjectDetectorTinyYolo import TinyYolo
from CarColorDetector import ColorDetection
from CarTypeDetector import TransferLearning
from VideoFileWriter import ConvertFramesToVideo
from Results import GetResults
from Plots import Plots
import time
import pickle
import cv2
import os

q1_time= 0
yolo_output = {}
final_output = {}
cropDir = 'CropImageDir/'

def get_files(folder_name):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(folder_name):
        files.extend(filenames)
        break
    return files

#Q1
def reader():
    videoToFrames = ConvertVideoToFrames()
    beforetime = time.time()
    files = videoToFrames.convert_video_to_frames()
    aftertime = time.time()
    q1_time = round(((aftertime-beforetime)/len(files)), 3)
    #print("Time to extract each frame for Q1: ",q1_time, "secs")
    return files,q1_time

#Q2
def detector(files):
    tinyYolo = TinyYolo()
    class_ids, confidences, boxes = tinyYolo.bound_and_crop_image(files[0])
    for file in files:
        beforetime = time.time()
        class_ids, confidences, boxes = tinyYolo.bound_and_crop_image(file)
        aftertime = time.time()
        yolo_output[file[file.rfind('/')+1:]] = (class_ids, confidences, boxes, round((aftertime-beforetime), 3))
    pickle.dump(yolo_output,open("TinyYolo.pkl",'wb'))
    
#Q3
def type_classifier(croppedFiles, cropDir = "CropImageDir/"):
    # Uncomment below lines for training the model
    # tlTrain = TransferLearning()
    # tlTrain.train_model()
    transferLearning = TransferLearning(batch_size=1)
    for file in croppedFiles:
        if(file.startswith("1_")):
            beforetime = time.time()
            frameName = file[2:]
            frameNumber = file[7:len(file)-4]
            car_type = transferLearning.test_model_yolo_image(file, cropDir)
            if(frameNumber not in final_output.keys()):
                final_output[frameNumber] = {}
                final_output[frameNumber]['Count'] = 1
                final_output[frameNumber]['Frame No'] = frameNumber
            final_output[frameNumber]['Type1'] = car_type
            if("2_"+frameName in croppedFiles):
                car_type = transferLearning.test_model_yolo_image(file, cropDir)
                final_output[frameNumber]['Count'] = 2
                final_output[frameNumber]['Type2'] = car_type
            aftertime = time.time()
            final_output[frameNumber]['TypeDetectionTime'] = round((aftertime-beforetime), 3)

#Q4
def color_classifier(croppedFiles, cropDir = 'CropImageDir/'):
    colDetection = ColorDetection()
    for file in croppedFiles:
        if(file.startswith("1_")):
            beforetime = time.time()
            frameName = file[2:]
            frameNumber = file[7:len(file)-4]
            color = colDetection.get_prominent_color(cropDir + file)
            if(frameNumber not in final_output.keys()):
                final_output[frameNumber] = {}
                final_output[frameNumber]['Count'] = 1
                final_output[frameNumber]['Frame No'] = frameNumber
            final_output[frameNumber]['Color1'] = color
            if("2_"+frameName in croppedFiles):
                color = colDetection.get_prominent_color(cropDir + "2_"+frameName)
                final_output[frameNumber]['Count'] = 2
                final_output[frameNumber]['Color2'] = color
            aftertime = time.time()
            final_output[frameNumber]['ColorDetectionTime'] = round((aftertime-beforetime), 3)

def draw_box_with_annotations(img, x, y, x_plus_w, y_plus_h, car_type, car_color, count):
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (230, 230, 230), 2)
    cv2.putText(img, "Car count: " + count, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 2)
    cv2.putText(img, car_type, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 2)
    cv2.putText(img, car_color, (x+10,y_plus_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 2)

def process_output(csv_file="Output CSV.csv", final_image_folder = "Final Output Images"):
    file = open(csv_file,"w+")
    printLine = "Frame No" +","+ "Count" +","+ "Type1" +","+ "Color1" +","+ "Type2" +","+ "Color2" +","+ "Question1 Time" +","+ "Question2 Time" +","+ "Question3 Time" +","+ "Question4 Time"
    file.write(printLine+"\n")
    videoReader = ConvertVideoToFrames()
    videoReader.create_dir_if_not_exits(final_image_folder)
    for frame in range(1,1496):
        frame = str(frame)
        count,color1,carType1,color2,carType2,q1,q2_time,q3_time,q4_time = "0", "", "", "", "", "", "", "", ""
        image_file = cv2.imread(videoReader.dir_to_save + "/image"+frame+".jpg")
        if frame in final_output.keys():
            color1 = final_output[frame]['Color1']
            carType1 = final_output[frame]['Type1'] 
            count = str(final_output[frame]['Count'])
            box = yolo_output['image'+frame+".jpg"][2][0]
            x,y,w,h = box[0],box[1],box[2],box[3]
            draw_box_with_annotations(image_file,round(x), round(y), round(x+w), round(y+h),carType1,color1, count)
            if 'Color2' in final_output[frame].keys():
                color2 = final_output[frame]['Color1']
                carType2 = final_output[frame]['Type1'] 
                box = yolo_output['image'+str(frame)+".jpg"][2][1]
                x,y,w,h = box[0],box[1],box[2],box[3]
                draw_box_with_annotations(image_file,round(x), round(y), round(x+w), round(y+h),carType2,color2, count)
                
        q1 = str(q1_time)
        q2_time = str(yolo_output['image'+frame+".jpg"][3])
        if str(frame) in final_output.keys() and 'ColorDetectionTime' in final_output[frame].keys():
            q4_time = str(final_output[frame]['ColorDetectionTime'])
        if str(frame) in final_output.keys() and 'TypeDetectionTime' in final_output[frame].keys():
            q3_time = str(final_output[frame]['TypeDetectionTime'])
        printLine = frame +","+ count +","+ carType1 +","+ color1+","+ carType2 +","+ color2 +","+ q1 +","+ q2_time +","+ q3_time +","+ q4_time
        file.write(printLine+"\n")
        cv2.imwrite(final_image_folder + "/image"+frame+".jpg",image_file)
    file.close()

def get_results(output):
    GetResults().get_results(output)

def show_plots():
    Plots().plot()
    
def writer():
    converter = ConvertFramesToVideo()
    converter.frames_to_video('Final Output Images/', 'Rohit_Raksha_CaseStudyAssignment2_video.mp4', 25.0)


#yolo_output = pickle.load(open("TinyYolo.pkl","rb")) # for faster runtime uncomment this line
image_files, q1_time = reader() # for faster runtime comment this line
detector(image_files)  # for faster runtime comment this line
croppedFiles = get_files(cropDir)
type_classifier(croppedFiles)
color_classifier(croppedFiles)
process_output()
get_results(final_output)
show_plots()
writer()