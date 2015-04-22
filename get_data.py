
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib



act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']





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

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("faces_subset.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            
            if os.path.isfile("uncropped/"+filename):
                try:
                    coor =  np.fromstring(line.split()[5], dtype=int, sep=',')
                    imgFace = imread("uncropped/"+filename, True)
                    cropFace = imgFace[coor[1]:coor[3], coor[0]:coor[2]]

                    while cropFace.shape[0]>64:
                        cropFace = imresize(cropFace, .5)
                    cropFace = imresize(cropFace, [32,32])

                    imsave("cropped/"+filename, cropFace)
                except:
                    pass
            
            print filename
            i += 1
    
    