# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:53:55 2020

@author: Raksha
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import warnings
import time
import datetime as dt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import multiprocessing as mp

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2, l1_l2

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import psutil
import random

warnings.filterwarnings("ignore")
#%matplotlib inline
#%system nvidia-smi
#%%time
epochs = 10
batch_size = 10
testsplit = .2
targetx = 512
targety = 512
learning_rate = 0.001
classes = 2
#seed = random.randint(1, 1000)
current_dir = os.path.dirname(os.path.realpath(__file__))
seed = 15

data_dir = current_dir + "\imagesdata\data\\" 
validation_dir = current_dir + "\imagesdata\\validationdata\\"
test_dir = current_dir + "\imagesdata\yolodata\\"

#%%

class TransferLearning:
    def end_epoch(self, epoch, logs):
        message = "End of epoch "+str(epoch)+". Learning rate: "+str(K.eval(self.model.optimizer.lr))
        os.system('echo '+message)
    
    def start_epoch(self, epoch, logs):
        print("Learning rate: ", K.eval(self.model.optimizer.lr))
    
    
    def start_train(self, logs):
        os.system("echo Beginning training")
        
    def load_model(self, name="car_classifier.h5"):
        self.model = keras.models.load_model(name)
        
    def initialize_model(self):
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(targetx, targety, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu',
                  kernel_initializer=glorot_uniform(seed),
                  kernel_regularizer=l2(0.01),
                  bias_regularizer=l2(0.01),
                  bias_initializer='zeros')(x)
        x = Dropout(rate = .3)(x)
        x = BatchNormalization()(x)
        predictions = Dense(classes, activation='softmax', 
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01),
                            kernel_initializer='random_uniform', 
                            bias_initializer='zeros')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        optimizer = Adam(lr=learning_rate)
        loss = "categorical_crossentropy"
        #loss = "binary_crossentropy"

        for layer in model.layers[:154]:
            layer.trainable = False
        
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=["accuracy"])
        
        self.model = model
        
    def __init__(self):
        self.datagen = ImageDataGenerator(
                        horizontal_flip=False,
                        validation_split=testsplit,
                        preprocessing_function=preprocess_input
                        )
        self.train_generator = self.datagen.flow_from_directory(
                                data_dir,
                                target_size=(targetx, targety),
                                batch_size=batch_size,
                                #class_mode="binary",
                                class_mode='categorical',
                                shuffle=True,
                                seed=seed,
                                subset="training"
                        )
    
        self.validation_generator = self.datagen.flow_from_directory(
                                        validation_dir,
                                        target_size=(targetx, targety),
                                        batch_size=batch_size,
                                        #class_mode="binary",
                                        class_mode='categorical',
                                        shuffle=False,
                                        seed=seed,
                                        subset="validation"
                                )
        self.test_generator = self.datagen.flow_from_directory(
                                    test_dir,
                                    target_size=(targetx, targety),
                                    batch_size=batch_size,
                                    #class_mode="binary",
                                    class_mode='categorical',
                                    shuffle=True,
                                    seed=seed)
        self.checkpoint = ModelCheckpoint('car_type_classifier.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             verbose=1,
                             mode='auto',
                             save_weights_only=False,
                             period=1)

        self.tensorboard = TensorBoard(log_dir="logs-"+dt.datetime.now().strftime("%m%d%Y%H%M%S"),
                            histogram_freq=0,
                            batch_size=batch_size,
                            write_graph=False,
                            update_freq='epoch')
        
        
        self.earlystop = EarlyStopping(monitor='val_accuracy',
                          min_delta=.0001,
                          patience=20,
                          verbose=1,
                          mode='auto',
                          baseline=None,
                          restore_best_weights=True)

        self.reducelr = ReduceLROnPlateau(monitor='val_accuracy',
                             factor=np.sqrt(.1),
                             patience=5,
                             verbose=1,
                             mode='auto',
                             min_delta=.0001,
                             cooldown=0,
                             min_lr=0.00001)

        self.lambdacb = LambdaCallback(on_epoch_begin=self.start_epoch,
                          on_epoch_end=self.end_epoch,
                          on_batch_begin=None,
                          on_batch_end=None,
                          on_train_begin=self.start_train,
                          on_train_end=None)
        self.initialize_model()
            
    def train_model(self):
        self.params = self.model.fit_generator(generator=self.train_generator, 
                                     steps_per_epoch=len(self.train_generator), 
                                      validation_data=self.validation_generator, 
                                      validation_steps=len(self.validation_generator),
                                      epochs=epochs,
                                      callbacks=[self.reducelr, self.earlystop, 
                                                  self.lambdacb, self.tensorboard, self.checkpoint])
        self.model.save("car_classifier.h5")
        self.plot_train_test_accuracy_loss()
        
    def plot_train_test_accuracy_loss(self):
        plt.subplot(1, 2, 1)
        plt.title('Training and test accuracy')
        plt.plot(self.params.epoch, self.params.history['accuracy'], label='Training accuracy')
        plt.plot(self.params.epoch, self.params.history['val_accuracy'], label='Test accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.title('Training and test loss')
        plt.plot(self.params.epoch, self.params.history['loss'], label='Training loss')
        plt.plot(self.params.epoch, self.params.history['val_loss'], label='Test loss')
        plt.legend()
        plt.show()
        
    def test_model_yolo_data(self):
        self.test_generator.reset()
        predictions = self.model.predict_generator(self.test_generator, steps=len(self.test_generator))
        y = np.argmax(predictions, axis=1)
        print(y)
        print(accuracy_score(self.test_generator.classes, y))
        print('Classification Report')
        cr = classification_report(y_true=self.test_generator.classes, y_pred=y, target_names=self.test_generator.class_indices)
        print(cr)

#%%
if __name__ == "__main__":
    tl = TransferLearning()
    tl.train_model()
    #tl.load_model()
    tl.test_model_yolo_data()