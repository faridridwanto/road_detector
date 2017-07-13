# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:51:25 2016

@author: Farid
"""

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

from shutil import copyfile
import os
from os.path import basename
from os.path import splitext
from os import walk
from os import listdir
#theano.config.compute_test_value = 'warn'

#### Constants
#THEANO_FLAGS='exception_verbosity=high'
GPU = True
#======
rootdir1 = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\hasilcrossvalall\\'
##rootdir2 = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\\\'
#array_of_pointer = []
##dataynya = []
##datayynya = []
files = os.listdir(rootdir1)
def check_pred(name):
    if name in files:
        imgy = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\hasilcrossvalall\\'+name)
    else:
        imgy = np.ones((16,16))*256
    return imgy
imgy = check_pred('745201.tiff')
gabunganh = np.asarray(imgy)
p = 745202
for i in xrange(89):
        imgy = check_pred(str(p)+'.tiff')
        gabunganh = np.append(gabunganh,imgy,axis=1)
        p = p+1
gabungan = np.asarray(gabunganh)
for a in xrange(89):
    imgy = check_pred(str(p)+'.tiff')
    gabunganh = np.asarray(imgy)
    p = p+1
    for i in xrange(89):
        imgy = check_pred(str(p)+'.tiff')
        gabunganh = np.append(gabunganh,imgy,axis=1)
        p = p+1
    #imgy = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\'+str(p)+'.tiff')
    gabungan = np.append(gabungan,gabunganh,axis=0)
misc.imsave('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\vizcrossvalall.tiff', gabungan)
#result0 = T.sum(T.eq(gabungan,0))
#print result0.eval()
#result1 = T.sum(T.eq(gabungan,255))
#print result1.eval()
#print float(result1.eval())/float(result0.eval())
print 'done'
#====
#print result1.eval()/(1440*1440)
#for i in xrange(len(files)):
##    imgy = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\hasilpreds\\'+files[i])
##    imgy = np.asarray(imgy)
##    imgy = np.reshape(imgy,(1,256))
##    dataynya.append(imgy)
## 
##    imgyy = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\hasilmappreds\\'+files[i])
##    imgyy = np.asarray(imgyy)
##    imgyy = np.reshape(imgyy,(1,256))
##    datayynya.append(imgyy)
#    base = os.path.splitext(files[i])[0]
##    array_of_pointer.append(int(namanya))
##    print dataynya
##    print datayynya
##hasil = T.mean(T.eq(dataynya, datayynya))
##print hasil.eval()
#    namafile = int(base)
#    array_of_pointer.append(namafile)
#for i in xrange(745201,753301):
#    copyfile('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_64_inputs\\'+str(i)+'.tiff', 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\theinputoftraining\\'+str(i)+'.tiff' )
#for i in xrange(len(array_of_pointer)):
#copyfile('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\massachussets validation maps\\'+'25229230_15'+'.tiff', 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\gabunganval.tiff' )