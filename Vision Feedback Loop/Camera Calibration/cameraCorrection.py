
#!/usr/bin/env python

"""
A fast image classifier for robotic vision-based feedback loop. 
Developed by Ardavan Bidgoli at CMU dFab for Robotic Plastering Project.

Tested with:
    openCV      3.3.1
    Numpy       1.13.3

"""


################################################
# Import modules
################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
import copy
from os import listdir, path
from os.path import isfile, join


################################################
# Image processing class
################################################

class ImageProcess(object):
    """Process and calibrate images from camera"""
    
    def __init__(self, calibrate = False, matrixPath = None, size= 1000):
        """Creates an object of image correction """
        
        if matrixPath == None:
            self.matrixPath = "calibration_data.npz"
        else:
            self.matrixPath = matrixPath+".npz"
        
        # setup parameters 
        self.size = size
        self.imageReadPath=  "./calibrationData"
        
        if calibrate:
            self.calibrateCamera()
            
        #loading the matrix calibration data
        calibrationData = np.load(self.matrixPath)
        distCoeff, intrinsic_matrix = calibrationData.files
        self.intrinsic_matrix = calibrationData[intrinsic_matrix]
        self.distCoeff = calibrationData[distCoeff]
    
    def readImages(self, path = None, crop= False, size = None):
        """reads images from a giben path and crop them with a desired size"""
        if path == None:
            path = self.imageReadPath
        if size == None:
            size = self.size
        names = [join(path, f) for f in listdir(path) 
                                     if (isfile(join(path, f)) 
                                    and (f[-4:]==".jpg" or f[-4:]==".png"))]

        self.sourceImages = [cv2.imread(name) for name in names ]
        self.images = copy.deepcopy(self.sourceImages)
        self.h,self.w,_ = self.sourceImages[0].shape
        
        if crop:
            wC = self.w//2
            hC = self.h//2
            self.images = [img[hC-size//2:hC+size//2, wC-size//2:wC+size//2] for img in self.images]
        
        return self.images
        
    def lensCorrection(self, images= None, save = False):
        """applies lens correction to the images"""
        
        if type(images) != np.ndarray and type(images) != list: 
            images = self.readImages()
        if type(images) != list: 
            images = [images]
                    
        self.corrected = [cv2.undistort(img, self.intrinsic_matrix, self.distCoeff, None) for img in images]
        print (len(self.corrected))
        if save:
            for i in range (len(self.corrected)):
                name = "corrected{}.jpg".format(i)
                cv2.imwrite(name,self.corrected[i])
                
        return self.corrected
    
    def saveImages (self, images):
        """Saves given images"""
        if images== None:
            return
        if type(images) != list: images = [images]
        
        for i in range (len(images)):
            name = "corrected{}.jpg".format(i)
            cv2.imwrite(name,images[i])
        return
    
    def calibrateCamera(self):
        """calibrate camera based on a set of given images"""
        # based on: https://www.theeminentcodfish.com/gopro-calibration/
        
        images = self.readImages()
        n_boards = len(images)
        board_w = 9
        board_h= 6
        board_dim= 25
        h,w,_ = images[0].shape
        image_size = (w,h)

        board_n = board_w * board_h
        opts = []
        ipts = []
        npts = np.zeros((n_boards, 1), np.int32)
        intrinsic_matrix = np.zeros((3, 3), np.float32)
        distCoeffs = np.zeros((5, 1), np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        # prepare object points based on the actual dimensions of the calibration board
        # like (0,0,0), (25,0,0), (50,0,0) ....,(200,125,0)
        
        objp = np.zeros((board_h*board_w,3), np.float32)
        objp[:,:2] = np.mgrid[0:(board_w*board_dim):board_dim,0:(board_h*board_dim):board_dim].T.reshape(-1,2)

        #Loop through the images.  Find checkerboard corners and save the data to ipts.
        for i in range(1, n_boards + 1):

            #Loading images
            print ('Loading... Calibration image' + str(i))
            image = images[i-1]

            #Converting to grayscale
            grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            #Find chessboard corners
            found, corners = cv2.findChessboardCorners(grey_image, (board_w,board_h),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if found == True:

                #Add the "true" checkerboard corners
                opts.append(objp)

                #Improve the accuracy of the checkerboard corners found in the image and save them to the ipts variable.
                cv2.cornerSubPix(grey_image, corners, (20, 20), (-1, -1), criteria)
                ipts.append(corners)

                #Draw chessboard corners
                cv2.drawChessboardCorners(image, (board_w, board_h), corners, found)

                #Show the image with the chessboard corners overlaid.
                plt.imshow(image)
                plt.show()

        print ('')
        print ('Finished processes images.')

        #Calibrate the camera
        print ('Running Calibrations...')
        print(' ')
        ret, intrinsic_matrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(opts, ipts, grey_image.shape[::-1],None,None)

        #Save matrices
        print('Intrinsic Matrix: ')
        print(str(intrinsic_matrix))
        print(' ')
        print('Distortion Coefficients: ')
        print(str(distCoeff))
        print(' ') 

        #Save data
        print ('Saving data file...')
        np.savez(self.matrixPath, distCoeff=distCoeff, intrinsic_matrix=intrinsic_matrix)
        print ('Calibration complete')
        #Calculate the total reprojection error.  The closer to zero the better.
        tot_error = 0
        for i in range(len(opts)):
            imgpoints2, _ = cv2.projectPoints(opts[i], rvecs[i], tvecs[i], intrinsic_matrix, distCoeff)
            error = cv2.norm(ipts[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error

        print ("total reprojection error: ", tot_error/len(opts))


if __name__ == "__main__":
    # make an object from the ImageProcess class
    # creates a new calibration matrix
    #cameraCorrection = ImageProcess(calibrate=True)
    # uses currently available matrix
    cameraCorrection = ImageProcess()
    # applies lens correction to already available images
    cameraCorrection.lensCorrection()
    # applies lens correction to a given image
    pic = cameraCorrection.lensCorrection(images = cameraCorrection.images[0])
    plt.imshow(pic[0])
    plt.show()
    print ("Done")


__author__ = "Ardavan Bidgoli"
__copyright__ = "Copyright 2018, Robotic Plastering Project"
__credits__ = ["Ardavan Bidgoli", "Josh Bard"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Ardavan Bidgoli"
__email__ = "abidgoli@andrew.cmu.edu"
__status__ = "Development"