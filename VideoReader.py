# -*- coding: utf-8 -*-
"""
Created on Sat Apr 04 02:21:56 2020

@author: grohit
"""

import os
import cv2
class ConvertVideoToFrames():
    def __init__(self,video = "video.mp4", dirName = "imageDir"):
        self.videoFile = video
        self.dir_to_save = dirName
        self.create_dir_if_not_exits(self.dir_to_save)
    
    def create_dir_if_not_exits(self, dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass
        
    def get_frame(self,videoCap, sec, image_count):
        videoCap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = videoCap.read()
        if hasFrames:
            cv2.imwrite(self.dir_to_save + "/image"+str(image_count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    
    def convert_video_to_frames(self):
        vidcap = cv2.VideoCapture('video.mp4')
        sec = 0
        frame_rate = 0.037
        count=1
        success = self.get_frame(vidcap, sec, count)
        while success:
            count = count + 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            success = self.get_frame(vidcap, sec, count)
    
if __name__ == "__main__":
    videoToFrames = ConvertVideoToFrames()
    videoToFrames.convert_video_to_frames()