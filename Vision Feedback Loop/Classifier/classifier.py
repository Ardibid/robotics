#!/usr/bin/env python

"""
A fast image classifier for robotic vision-based feedback loop. 
Developed by Ardavan Bidgoli at CMU dFab for Robotic Plastering Project.

Tested with:
    Keras       2.1.2
    Tensorflow  1.4.0
    Numpy       1.13.3

"""


################################################
# Import modules
################################################

import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout, MaxPool2D, Flatten
from keras.activations import relu, softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint
import tensorboard

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import time


################################################
# helper functions
################################################
def readImages(path,imgRows,imgCols,channels):
    """reads images in a specific dimension from given path"""
    allImages = []
    images = [f for f in listdir(path) if \
                    (isfile(join(path, f)) and (f[-4:]==".jpg" or f[-4:]==".png"))]

    for img in images:
        tmpPath = join(path, img)
        if channels == 1:
            img = misc.imread(tmpPath,'F')
        else: 
            img = misc.imread(tmpPath)
        img = misc.imresize(img, (imgRows,imgCols))
        img = img.astype(np.float32)
        img /= 255.
        img = np.reshape(img, (imgRows, imgCols,channels))
        allImages.append(img)
    return np.array(allImages)

def formatImages(path,imgRows,imgCols,channels):
    """ 
    format all the images in the dataset folder to match the format
    THIS IS A DESTRUCTIVE METHOD! USE IT WITH CAUTION!
    """
    folders = [folder for folder in listdir(path) if not isfile(join(path,folder))]
    print (folders)

    for folder in folders:
        newPath = join(path, folder)

        images = [f for f in listdir(newPath) if \
                    (isfile(join(newPath, f)) and (f[-4:]==".jpg" or f[-4:]==".png"))]
        for img in images:
            tmpPath = join(newPath, img)
            if channels == 1:
                img = misc.imread(tmpPath,'F')
            else: 
                img = misc.imread(tmpPath)
            misc.imresize(img, (imgRows,imgCols))
    return 



################################################
# Classifier class
################################################
class Classifier(object):
    """ CLass of classifier to classify images from robot camera"""
    def __init__(self,  batchSize = 64, targetSize = 64, dropOut = 0.5, channels =3, 
                        train = False, formatDataSet= False, modelPath = None, epochs = 100):
        
        # data setup
        self.dataSetPath        = 'dataset'
        self.trainSetPath       = 'dataset/training_set'
        self.testSetPath        = 'dataset/test_set'
        self.classificationPath = './dataset/test'
        
        if modelPath == None:
            self.modelPath = './models/model'
        
        # hyper parameters
        self.batchSize = batchSize
        self.targetSize = targetSize
        self.channels = channels
        self.dropOut = dropOut
        self.epochs = epochs
        self.activation = 'relu'


        if train:
            if formatDataSet:
                formatImages(self.dataSetPath ,self.targetSize,self.targetSize,self.channels)
            self.nnModel()
            self.trained = self.train(self.epochs)
        else:
            if self.modelPath:
                self.model = keras.models.load_model(self.modelPath)
                print ("Model loaded!")
            else:
                print ("You need to train your model or load the model from a given path")


    def nnModel(self):
        """ creates a NN model"""

        self.model = Sequential()
        self.model.add(Convolution2D(filters = 64, kernel_size= (3,3), 
                    input_shape= (self.targetSize,self.targetSize,3),activation = self.activation))
        self.model.add(MaxPool2D(pool_size = (2,2)))

        self.model.add(Convolution2D(filters =64, kernel_size= (3,3),activation = self.activation))
        self.model.add(MaxPool2D(pool_size = (2,2)))

        self.model.add(Convolution2D(filters =64, kernel_size= (3,3),activation = self.activation))
        self.model.add(MaxPool2D(pool_size = (2,2)))

        self.model.add(Convolution2D(filters =64, kernel_size= (3,3),activation = self.activation))
        self.model.add(MaxPool2D(pool_size = (2,2)))

        self.model.add(Flatten())
        self.model.add(Dropout(self.dropOut))

        self.model.add(Dense(units = 128, activation = self.activation))
        self.model.add(Dropout(self.dropOut))

        self.model.add(Dense(units = 128, activation = self.activation))
        self.model.add(Dropout(self.dropOut))

        self.model.add(Dense(units = 4, activation = 'softmax'))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])
        self.model.summary()
        return

    def train(self, epochs = 100):
        """trains the model using data augmentation and fit_generator method"""
       
        # data augmentaion
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        test_datagen = ImageDataGenerator(
                rescale=1./255)

        training_set = train_datagen.flow_from_directory(
            self.trainSetPath,
            target_size=(self.targetSize, self.targetSize),
            batch_size= self.batchSize,
            class_mode='categorical')

        test_set = test_datagen.flow_from_directory(
            self.testSetPath,
            target_size=(self.targetSize, self.targetSize),
            batch_size= self.batchSize,
            class_mode='categorical')

        # saving checkpoints
        checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=False)
        # saves the graphs for visualization on tensorboard
        # go to the project folder on run this in command line:
        # tensorboard --logdir=./logs
        # then open a this address in a browser: http://localhost:6006
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size = self.batchSize, write_grads=True, write_images=True)

        # traiing
        self.model.fit_generator(
            training_set,
            max_queue_size=10,
            workers = 10,
            steps_per_epoch=8000/self.batchSize,
            epochs=epochs,
            validation_data=test_set,
            validation_steps=2000/self.batchSize,callbacks=[tensorboard] )
        self.latestModelName = "./models/{}".format(time.time())
        self.model.save(self.latestModelName)
        return

    def classify(self,path = None):
        """
        classifies a group of images provided in images as numpy array
        the shape of the inpit should be (number of images, colSize, rowSize, channels).
        Fro example (225,64,64,3), the last three numbers should be the same as the training set.
        """
        if path == None:
            path = self.classificationPath

        images = readImages(path,self.targetSize,self.targetSize,self.channels)
        self.results = self.model.predict_classes(images,batch_size = 16)
        return self.results
    
    def saveAsMainModel(self):
        self.model.save(self.modelPath)


if __name__ == "__main__":
    # sample for training a model
    classifier = Classifier(train =True, epochs = 50)
    # sample for loading a model
    #classifier = Classifier(modelPath = "./models/model")
    result = classifier.classify()
    print (result)



__author__ = "Ardavan Bidgoli"
__copyright__ = "Copyright 2018, Robotic Plastering Project"
__credits__ = ["Ardavan Bidgoli", "Josh Bard"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Ardavan Bidgoli"
__email__ = "abidgoli@andrew.cmu.edu"
__status__ = "Development"