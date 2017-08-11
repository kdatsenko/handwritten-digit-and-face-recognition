from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from hashlib import sha256
from rgb2gray import rgb2gray
import time

act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


def setup():
    '''Create a file of all actors' and actresses' names, cropping boundaries,
    hashes, etc. Create a directory 'uncropped' for the uncropped images.
    '''
    
    filenames = ['facescrub_actors.txt', 'facescrub_actresses.txt']
    if not os.path.isfile('faces_subset.txt'):
        with open('faces_subset.txt', 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
    if not os.path.exists('uncropped'):
        os.mkdir('uncropped')      
    if not os.path.exists('cropped'):
        os.mkdir('cropped')   
    if not os.path.exists('cropped_rgb'):
        os.mkdir('cropped_rgb') 


def get_all():
    plt.ion()
    setup() #call this function to retrieve all uncropped imgs from the web
    testfile = urllib.request.URLopener()
    for a in act:
        name = a.split()[1].lower()
        i = 0
        # This faces_subset.txt contains all actors and actresses
        for line in open("faces_subset.txt"):
            if a in line:
                try:
                    filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                    x1 = int(line.split()[5].split(',')[0])
                    y1 = int(line.split()[5].split(',')[1])  
                    x2 = int(line.split()[5].split(',')[2]) 
                    y2 = int(line.split()[5].split(',')[3])
                    hash = line.split()[6]
                except:
                    continue
                    
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long 
                #to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    #print "uncropped/"+filename
                    continue
                else:
                    # Remove bad images
                    file = open("uncropped/"+filename, "rb").read()
                    actual_hash = sha256(file).hexdigest()
                    if actual_hash != hash:
                        continue
                        
                    try:
                        # Now crop the image at each loop
                        I = imread("uncropped/"+filename)
                        # Crop the image at each loop and call rgb2gray function
                        
                        out = I[y1:y2, x1:x2]
                        # Resize the image and save it
                        part_798_im = rgb2gray(out)
                        part_798_im = imresize(part_798_im, [64, 64])
                        
                        part_10_im = imresize(out, [227, 227, 3])
                        
                        #plt.imshow(out, cmap=cm.gray)
                        
                        imsave("cropped/"+filename, part_798_im)
                        imsave("cropped_rgb/"+filename, part_10_im)
                    except Exception as e:
                        print("Couldn't read the file: " + str(e))
                        continue 
                print(filename)
                i += 1
       
if __name__ == "__main__":
    get_all()
                
                
                
                
                