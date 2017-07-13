# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 06:23:54 2017

@author: Farid
"""

import pickle
import gzip
import sys
import pprint
import time

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
from theano.ifelse import ifelse
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
def save_net(net,filename='sampah3'):
    f = open(filename, 'wb')
    cPickle.dump(net, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
def load_file(filename="sampah2"):
    f = open(filename,'rb')
    net = cPickle.load(f)
    f.close()
    return net
a = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\pred_crossval2-315.tiff')
a = np.asarray(a, dtype='float')
a = a/255
b = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\target_crossval2-315.tiff')
b = np.asarray(b,dtype='float')
b = b/255
threshold = 0.0
aofprec = []
aofrec = []
precs = []
recs = []
for i in xrange(100):
    cond = T.switch(T.lt(threshold,a), 1.0, 0.0)
    #misc.imsave('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\'+str(i)+'.tiff', cond.eval())
    dividee = T.sum(T.eq(cond,1)*T.eq(b,1))
    divider_p = T.sum(cond)
    divider_r = T.sum(b)
    aofprec.append(dividee/divider_p)
    aofrec.append(dividee/divider_r)
    threshold = threshold+0.01
for i in xrange(100):
    #print aofprec[i].eval()
    precs.append(aofprec[i].eval())
for i in xrange(100):
    #print aofrec[i].eval()
    recs.append(aofrec[i].eval())
save_net([recs,precs],filename='rec-prec-crossval3-refined315')