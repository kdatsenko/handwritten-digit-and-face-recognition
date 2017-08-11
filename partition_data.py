from numpy import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.io import loadmat, savemat
import matplotlib.image as mpimg
import os
import scipy
from scipy.ndimage import filters
import urllib
from hashlib import sha256
from rgb2gray import rgb2gray
from get_data import get_all
import time


#Separate the dataset into three non-overlapping parts: 
#the training set (100 face images per actor), 
#the validation set (10 face images per actor), 
#and the test set (10 face images per actor). 
#For the report, describe how you did that. (Any method is fine). 
#The training set will contain faces whose labels you assume you know. 
#The test set and the validation set will contain faces whose labels 
#you pretend to not know and will attempt to determine using the data 
#in the training set.

act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

#def fetch_sets(actors=None, n_train=68, n_val=30, n_test=30):

def fetch_sets(n_train=70, n_val=25, n_test=25, actors=None):
    
    if actors is None:
        actors = act
        
    if not os.path.exists('cropped'):
        get_all()
    
    train_set = []
    validation_set = []
    test_set = []

    train_labels = []
    validation_labels = []
    test_labels = []
    total_num = n_train+n_val+n_test
    
    faces_gray = {}
    faces_rgb = {}
    actor_label = 0 #0 to 5
    np.random.seed(0)
    for a in actors: #do this per actor
        name = a.split()[1].lower() #last name of actor
        # This would print all the files and directories
        dirs = os.listdir('cropped')
        imgs_of_actor_gray = []
        imgs_of_actor_rgb = []
        for filename in dirs:
            if name in filename: #if pic has this last name
                im_gray = imread("cropped/"+filename, flatten=True)
                im_rgb = imread("cropped_rgb/"+filename)[:,:,:3]
                
                im_gray = im_gray.flatten() #flatten image
                imgs_of_actor_gray.append(im_gray)
                imgs_of_actor_rgb.append(im_rgb)
                #plt.ion()
                #plt.imshow(im, cmap=cm.gray)
                #break
        
        num_imgs = len(imgs_of_actor_gray)
        
        if num_imgs >= total_num:
            np.random.shuffle(imgs_of_actor_gray)
            np.random.shuffle(imgs_of_actor_rgb)
            
            img_size_gray = len(imgs_of_actor_gray[0])
            train_set_gray = zeros((n_train, img_size_gray))
            valid_set_gray = zeros((n_val, img_size_gray))
            test_set_gray = zeros((n_test, img_size_gray))
            
            dim1, dim2, dim3 = imgs_of_actor_rgb[0].shape
            train_set_rgb = zeros((n_train, dim1, dim2, dim3))
            valid_set_rgb = zeros((n_val, dim1, dim2, dim3))
            test_set_rgb = zeros((n_test, dim1, dim2, dim3))
            
            j = 0
            for i in range(0,n_train):
                train_set_gray[j][:] = imgs_of_actor_gray[i]
                train_set_rgb[j, :, :, :] = imgs_of_actor_rgb[i]
                j += 1
            j = 0
            for i in range(n_train,n_train+n_val):
                valid_set_gray[j][:] = imgs_of_actor_gray[i]
                valid_set_rgb[j, :, :, :] = imgs_of_actor_rgb[i]
                j += 1
            j = 0
            for i in range(n_train+n_val,n_train+n_val+n_test):
                test_set_gray[j][:] = imgs_of_actor_gray[i]
                test_set_rgb[j, :, :, :] = imgs_of_actor_rgb[i]
                j += 1
            faces_gray["train"+str(actor_label)] = train_set_gray
            faces_gray["valid"+str(actor_label)] = valid_set_gray
            faces_gray["test"+str(actor_label)] = test_set_gray
            
            faces_rgb["train"+str(actor_label)] = train_set_rgb
            faces_rgb["valid"+str(actor_label)] = valid_set_rgb
            faces_rgb["test"+str(actor_label)] = test_set_rgb
        else:
            raise ValueError('Not enough data to produce sets of %s - %d needed, %d found' %(a, total_num, num_imgs))
        
        actor_label += 1
    
    savemat('faces.mat', faces_gray)
    savemat('faces_10.mat', faces_rgb)
    
if __name__ == "__main__":
    fetch_sets()
    
