# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:50:54 2020

@author: grohit
"""

import cv2
from os import listdir
from os.path import isfile, join
class ConvertFramesToVideo():
    
    def frames_to_video(self,pathIn,pathOut,fps):
        #function to convert image frames to mo4 video
        frame_array = []
        files = [f for f in listdir(pathIn) if isfile(join(pathIn, f))]
        files = ['image'+str(i)+".jpg" for i in range(1,len(files)+1)]
    
        for i in range(len(files)):
            filename=pathIn + files[i]
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)
    
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
        
if __name__ == "__main__":
    converter = ConvertFramesToVideo()
    converter.frames_to_video('Final Output Images/', 'demo_video.mp4', 25.0)