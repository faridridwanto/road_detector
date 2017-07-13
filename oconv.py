# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:16:40 2016

@author: Farid
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

the_input = [[[0,0,0,0,0,0,0],
             [0,0,0,2,1,1,0],
             [0,2,1,2,2,1,0],
             [0,0,1,0,1,1,0],
             [0,0,1,1,2,1,0],
             [0,1,1,0,2,2,0],
             [0,0,0,0,0,0,0]],
             [[0,0,0,0,0,0,0],
             [0,2,2,1,2,1,0],
             [0,0,1,2,0,1,0],
             [0,2,2,2,2,0,0],
             [0,1,0,1,2,1,0],
             [0,0,2,2,0,0,0],
             [0,0,0,0,0,0,0]],
             [[0,0,0,0,0,0,0],
             [0,0,0,0,1,0,0],
             [0,1,2,0,0,2,0],
             [0,0,1,1,2,1,0],
             [0,2,0,1,0,0,0],
             [0,0,1,2,0,1,0],
             [0,0,0,0,0,0,0]]]
the_input = np.asarray(the_input,dtype=theano.config.floatX)
the_input = np.reshape(the_input,(1,3,7,7))
the_filter = [[[0,0,1],
              [-1,0,0],
              [1,-1,1]],
              [[1,-1,0],
              [-1,0,-1],
              [-1,0,1]],
              [[0,1,0],
               [0,1,-1],
               [1,1,-1]]]
the_filter = np.asarray(the_filter,dtype=theano.config.floatX)
the_filter = np.reshape(the_filter,(1,3,3,3))
conv_out = conv.conv2d(
            input=the_input, filters=the_filter, filter_shape=the_filter.shape,
            image_shape=the_input.shape, subsample=(2,2))
print conv_out.eval()