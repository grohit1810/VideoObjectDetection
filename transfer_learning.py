# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:53:55 2020

@author: Raksha
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import warnings
import datetime as dt
from sklearn.metrics import classification_report, accuracy_score

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
from tensorflow.keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt
import pickle

warnings.filterwarnings("ignore")
#%matplotlib inline
#%system nvidia-smi
#%%time

epochs = 10 # Number of epochs
batch_size = 10 #Batch size
testsplit = .2 # Train and validation split
targetx = 512 # Target shape
targety = 512
learning_rate = 0.001 # Learning rate
classes = 2 # Number of classes for classification
#seed = random.randint(1, 1000)
current_dir = os.path.dirname(os.path.realpath(__file__))
seed = 15

# Directories where the train validation and test data is stored
data_dir = current_dir + "\imagesdata\data\\" 
validation_dir = current_dir + "\imagesdata\\validationdata\\"
test_dir = current_dir + "\imagesdata\yolodata\\"

#%%

class TransferLearning:
    """"
        End of epoch function
    """
    def end_epoch(self, epoch, logs):
        message = "End of epoch "+str(epoch)+". Learning rate: "+str(K.eval(self.model.optimizer.lr))
        os.system('echo '+message)
    
    def start_epoch(self, epoch, logs):
        print("Learning rate: ", K.eval(self.model.optimizer.lr))
    
    
    def start_train(self, logs):
        os.system("echo Beginning training")
        
    """
        Method to load a saved model
    """
    def load_model(self, name="car_classifier.h5"):
        self.model = keras.models.load_model(name)
    
    """
        Method which initializes the model parameters
    """
    def initialize_model(self):
        """
            Using the mobilenet model as the base model for transfer learning 
            targetx and targety the final image is reduces to this size
        """
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(targetx, targety, 3))
        # Getting the model layers
        x = base_model.output
        # Adding the global average pooling 
        x = GlobalAveragePooling2D()(x)
        # Adding batch normalization
        x = BatchNormalization()(x)
        # Adding Dense layer
        x = Dense(512,
                  # Using RELU non linearity
                  activation='relu',
                  # Initilization of the deep layer
                  kernel_initializer=glorot_uniform(seed),
                  # Usinf L2 regularization
                  kernel_regularizer=l2(0.01),
                  # Bias L2 regularization
                  bias_regularizer=l2(0.01),
                  # Initilize bias to zero
                  bias_initializer='zeros')(x)
        # Adding dropout layer to prevent overfitting
        x = Dropout(rate = .3)(x)
        # Normalizing the layer
        x = BatchNormalization()(x)
        # Adding softmax layer to predict classes
        predictions = Dense(classes, activation='softmax', 
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01),
                            kernel_initializer='random_uniform', 
                            bias_initializer='zeros')(x)
        # Modifying the model
        model = Model(inputs=base_model.input, outputs=predictions)
        #Adam optimizer to make use of adaptive learning rate for all parameters
        optimizer = Adam(lr=learning_rate)
        # using cross entropy loss
        loss = "categorical_crossentropy"
        #loss = "binary_crossentropy"
        
        #Freezing the pre trained layers
        for layer in model.layers[:154]:
            layer.trainable = False
        
        # Instanciating model
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=["accuracy"])
        
        self.model = model
        
    def __init__(self):
        # Image generator
        self.datagen = ImageDataGenerator(
                        # Prevent flipping as test data does not have flipped images
                        horizontal_flip=False,
                        validation_split=testsplit,
                        preprocessing_function=preprocess_input)
        
        # Training images generator fpt model
        self.train_generator = self.datagen.flow_from_directory(
                                data_dir,
                                #Image converted to target size
                                target_size=(targetx, targety),
                                batch_size=batch_size,
                                #class_mode="binary",
                                class_mode='categorical',
                                shuffle=True,
                                seed=seed,
                                # Training data
                                subset="training")
        
        # Validation  images generator
        self.validation_generator = self.datagen.flow_from_directory(
                                        validation_dir,
                                        target_size=(targetx, targety),
                                        batch_size=batch_size,
                                        #class_mode="binary",
                                        class_mode='categorical',
                                        shuffle=False,
                                        seed=seed,
                                        subset="validation")
        
        # Test image generator from the YOLO images
        self.test_generator = self.datagen.flow_from_directory(
                                    test_dir,
                                    target_size=(targetx, targety),
                                    batch_size=batch_size,
                                    #class_mode="binary",
                                    class_mode='categorical',
                                    shuffle=True,
                                    seed=seed)
        #Adding checkpoint to save parameters of model 
        self.checkpoint = ModelCheckpoint('car_type_classifier.h5',
                             monitor='val_accuracy',
                             #Saving only the best model
                             save_best_only=True,
                             verbose=1,
                             mode='auto',
                             save_weights_only=False,
                             period=1)
        
        #Log directory
        self.tensorboard = TensorBoard(log_dir="logs-"+dt.datetime.now().strftime("%m%d%Y%H%M%S"),
                            histogram_freq=0,
                            batch_size=batch_size,
                            write_graph=False,
                            update_freq='epoch')
        
        
        # Stop training of model based on val_accuracy
        self.earlystop = EarlyStopping(monitor='val_accuracy',
                          # when change less than 0.0001      
                          min_delta=.0001,
                          patience=20,
                          verbose=1,
                          mode='auto',
                          baseline=None,
                          restore_best_weights=True)

        
        #Reducing learning rate when val_accuracy has stopped improving
        self.reducelr = ReduceLROnPlateau(monitor='val_accuracy',
                             # LR will be lowered by this factor
                             factor=np.sqrt(.1),
                             #Number of epochs after wcich the LR will be reduced
                             patience=5,
                             verbose=1,
                             mode='auto',
                             min_delta=.0001,
                             # Do not wait for resuming of the normal operation after LR reduced
                             cooldown=0,
                             # Lower bound of LR
                             min_lr=0.00001)
        
        #Anonymous functions called on various times
        self.lambdacb = LambdaCallback(on_epoch_begin=self.start_epoch,
                          on_epoch_end=self.end_epoch,
                          on_batch_begin=None,
                          on_batch_end=None,
                          on_train_begin=self.start_train,
                          on_train_end=None)
        # Initilizing the model
        self.initialize_model()
    """
        Training of the model is here
    """
    def train_model(self):
        self.params = self.model.fit_generator(generator=self.train_generator, 
                                     steps_per_epoch=len(self.train_generator), 
                                      validation_data=self.validation_generator, 
                                      validation_steps=len(self.validation_generator),
                                      epochs=epochs,
                                      callbacks=[self.reducelr, self.earlystop, 
                                                  self.lambdacb, self.tensorboard, self.checkpoint])
        # Save model to disk
        self.model.save("car_classifier.h5")
        self.plot_train_test_accuracy_loss()
        
    """
        Plot of training and test accuray vs number of epochs
        Plot of training and test loss vs number of epochs
    """
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
    
    """
        Prediction of the model on the yolo data
    """
    def test_model_yolo_data(self):
        self.test_generator.reset()
        # Predicting on the test data
        predictions = self.model.predict_generator(self.test_generator, steps=len(self.test_generator))
        y = np.argmax(predictions, axis=1)
        print("Accuracy ", accuracy_score(self.test_generator.classes, y))
        predictions = {}
        #Saving results to pickle file
        for i in range(len(y)):
            predicted = y[i]
            actual = self.test_generator.classes[i]
            predictions[i] = {
                "actual": actual,
                "predicted": predicted
                }
        pickle.dump(predictions, open("transferLearningpredictions.pkl", "wb"))
# =============================================================================
#         print('Classification Report')
#         cr = classification_report(y_true=self.test_generator.classes, y_pred=y, target_names=self.test_generator.class_indices)
#         print(cr)
# =============================================================================

#%%
if __name__ == "__main__":
    tl = TransferLearning()
    tl.train_model()
    #tl.load_model()
    tl.test_model_yolo_data()