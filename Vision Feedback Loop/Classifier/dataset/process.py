import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


from os import listdir
from os.path import isfile, join
import cv2

import time
from time import gmtime, strftime
import imageio


def readImages(path,imgRows, imgCols,channels):
    folders = [folder for folder in listdir(path) if not isfile(join(path,folder))]
    print (folders)
    allImages = []
    for folder in folders:
        newPath = join(path, folder)

        images = [f for f in listdir(newPath) if (isfile(join(newPath, f)) and (f[-4:]==".jpg" or f[-4:]==".png"))]
        
        for img in images:
            tmpPath = join(newPath, img)
            if channels == 1:
                img = misc.imread(tmpPath,'F')
            else: 
                img = misc.imread(tmpPath)
            img = misc.imresize(img, (imgRows,imgCols))

            misc.imsave(tmpPath,img)
            img = img.astype(np.float32)
            img /= 255.
            img = np.reshape(img, (imgRows, imgCols,channels))
            allImages.append(img)
    return np.array(allImages)


def showImage(img):
    print("Image shape: {}".format(img.shape))
    plt.imshow(img)
    plt.show()
    return None

def processImages(imgRows, imgCols , channels, path = None):
    # reads images from the source directory as black and white with range of [0.0-1]
    # the output will be a numpy array of shape: [number of images, size*size]
    if path ==None:
        path = "./lfw"
    print (path)
    images = readImages(path,imgRows, imgCols, channels = channels)
    print ("{} images loaded".format(len(images)))
    return images

images = processImages(28,28,3,"./training_set")
showImage(images[1])