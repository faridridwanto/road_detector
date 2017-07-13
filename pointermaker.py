# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:07:58 2016

@author: Farid
This code was created to make a list of order to take the images as the input
(result of this code will then be used in 'chunk-making.py').
"""

#### Libraries
# Standard library
import pickle
import gzip
import sys
import pprint

# Third-party libraries
import scipy
import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.inf)
import theano
import theano.tensor as T
from scipy import misc
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from six.moves import cPickle

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
import os
from os.path import basename
from os.path import splitext
from os import walk
from os import listdir
import random
#theano.config.compute_test_value = 'warn'

#### Constants
#THEANO_FLAGS='exception_verbosity=high'
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."
array_of_pointer = []
#k=0
#u=0
#pointer = ""
#sum_of_pointer = 0
#urutan = 1
#rootdir1 = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\massachussets training maps\\23579005_15.tif'
#img = misc.imread(rootdir1)
#img = np.asarray(img)
#
#jumlah0 =  T.sum(T.eq(img,0)).eval()
#print jumlah0
#jumlah1 = T.sum(img).eval()/255
#print jumlah1
#print jumlah1/jumlah0
#files = os.listdir(rootdir1)
#for namefile in files:
#    img = misc.imread(rootdir1+namefile)
#    img = np.asarray(img)
#    print urutan
#    urutan = urutan+1
#    if(T.sum(img).eval()>sum_of_pointer):
#        pointer = namefile
#        sum_of_pointer = T.sum(img).eval()
    
#jumlah = 0
#while jumlah < 2700:
#    k = np.random.randint(8000000)
##    imgy = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_16_maps\\'+str(k)+'.tif')
##    imgy = np.asarray(imgy)
###    imgy = np.reshape(imgy,(1,256))
##    if (T.sum(imgy).eval()!=0):
#    array_of_pointer.append(k)
##        print T.sum(imgy).eval()
#    jumlah = jumlah+1
array_of_pointer = random.sample(range(1,8000000), 2700)    
#    u = u+1
f = open('pointerswithfiltering-test2', 'wb')
cPickle.dump(array_of_pointer, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
    
