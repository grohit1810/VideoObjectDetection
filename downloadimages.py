import scipy.io
import numpy as np
import cv2
import os

"""
    This class creats the training data images from the cars dataset of Stanford
"""
class TransferLearningImages:
    def __init__(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        # The matlab file which has the annotations of the images
        mat = scipy.io.loadmat(data_dir + "\devkit\cars_train_annos.mat")
        carsmeta = scipy.io.loadmat(data_dir + "\devkit\cars_meta.mat")
        self.labels = np.asarray(carsmeta["class_names"])[0]
        # Directory to save images
        self.imageDir = "cars_train/cars_train/"
        self.hatchbackDir = "imagesdata/data/hatchbacks/"
        self.sedanDir = "imagesdata/data/sedans/"
        self.images = mat["annotations"][0]
    
    def create_images_in_folder(self):
        for i in self.images:
            labelIndex = int(i[4][0][0])
            classLabel = self.labels[labelIndex-1][0]
            isSedan =  ("Sedan" or "sedan") in classLabel
            isHatchback = ("Hatchback" or "hatchback") in classLabel
            if isSedan or isHatchback:
                imageName = i[5][0]
                image = cv2.imread(self.imageDir + imageName)
                minx = i[0][0][0]
                maxx = i[1][0][0]
                miny = i[2][0][0]
                maxy = i[3][0][0]
                crop = image[minx: maxy, maxx: miny]
            try:
                if isSedan:
                    print("Sedan..." + i[5][0] + classLabel)
                    cv2.imwrite(self.sedanDir + imageName, crop)
                if isHatchback:
                    print("Hatchback..." + i[5][0] + classLabel)
                    cv2.imwrite(self.hatchbackDir + imageName, crop)
            except:
                print("error")

if __name__ == "__main__":
    tli = TransferLearningImages()
    tli.create_images_in_folder()