# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 19:27:27 2016

@author: Farid
"""
import pickle
import cPickle
import pickle
import cPickle
import gzip
import sys
import pprint

import scipy
import numpy as np
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

GPU = True

img = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_64_inputs\\1.tiff')
img = np.asarray(img)
img = img.transpose(2,0,1).reshape(1,3,64,64)
dataxnya = np.asarray(img, dtype='float64')
#dataynya = np.random.randint(10, size=999)
#dataynya = np.array(dataynya, dtype=np.int64)
#initializing array for y
imgy = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_16_maps\\1.tif')
imgy = np.asarray(imgy)
imgy = np.reshape(imgy,(1,256))
dataynya = np.asarray(imgy)
#finishing array x and y
for urutan in range(2,11):
    #x
    base = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_64_inputs\\'
    extension = '.tiff'
    nomor = np.random.randint(8900000)
    nomor = str(urutan)
    link = base+nomor+extension
    img = misc.imread(link)
    img = np.asarray(img, dtype='float64')
    img = img.transpose(2,0,1).reshape(1,3,64,64)
    dataxnya = np.append(dataxnya,img,axis=0)
    #y
    basey = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_16_maps\\'
    extensiony = '.tif'
    linky = basey+nomor+extensiony
    imgy = misc.imread(linky)
    imgy = np.asarray(imgy)
    imgy = np.reshape(imgy,(1,256))
    dataynya = np.append(dataynya,imgy,axis=0)

#creating validation data
#initializing array for x
img = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\validation_64_inputs\\1.tiff')
img = np.asarray(img, dtype='float64')
img = img.transpose(2,0,1).reshape(1,3,64,64)
datavxnya = np.asarray(img, dtype='float64')
#datavynya = np.random.randint(10, size=999)
#datavynya = np.array(datavynya, dtype=np.int64)
#initializing array for y
imgy = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\validation_16_maps\\1.tiff')
imgy = np.asarray(imgy)
imgy = np.reshape(imgy,(1,256))
datavynya = np.asarray(imgy)
#finishing array x and y

for urutan in range(2,9):
    base = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\validation_64_inputs\\'
    extension = '.tiff'
    nomor = np.random.randint(113400)
    nomor = str(nomor)
    link = base+nomor+extension
    img = misc.imread(link)
    img = np.asarray(img, dtype='float64')
    img = img.transpose(2,0,1).reshape(1,3,64,64)
    datavxnya = np.append(datavxnya,img,axis=0)
    #y
    basey = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\validation_16_maps\\'
    extensiony = '.tiff'
    linky = basey+nomor+extensiony
    imgy = misc.imread(linky)
    imgy = np.asarray(imgy)
    imgy = np.reshape(imgy,(1,256))
    datavynya = np.append(datavynya,imgy,axis=0)

img = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\test_64_inputs\\1.tiff')
img = np.asarray(img)
img = img.transpose(2,0,1).reshape(1,3,64,64)
datatxnya = np.asarray(img, dtype='float64')
#datatynya = np.random.randint(10, size=999)
#datatynya = np.array(datatynya, dtype=np.int64)
#initializing array for y
imgy = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\test_16_maps\\1.tiff')
imgy = np.asarray(imgy)
imgy = np.reshape(imgy,(1,256))
datatynya = np.asarray(imgy)
#finishing array x and y

for urutan in range(2,9):
    base = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\test_64_inputs\\'
    extension = '.tiff'
    nomor = np.random.randint(396900)
    nomor = str(nomor)
    link = base+nomor+extension
    img = misc.imread(link)
    img = np.asarray(img, dtype='float64')
    img = img.transpose(2,0,1).reshape(1,3,64,64)
    datatxnya = np.append(datatxnya,img,axis=0)
    #y
    basey = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\test_16_maps\\'
    extensiony = '.tiff'
    linky = basey+nomor+extensiony
    imgy = misc.imread(linky)
    imgy = np.asarray(imgy)
    imgy = np.reshape(imgy,(1,256))
    datatynya = np.append(datatynya,imgy,axis=0)

tra_data = [dataxnya,dataynya]
val_data = [datavxnya,datavynya]
tes_data = [datatxnya,datatynya]
tes_all = [tra_data,val_data,tes_data]
thefile = open("../data/datasatnya2.pkl", "wb")
cPickle.dump(tes_all, thefile, protocol=cPickle.HIGHEST_PROTOCOL)
thefile.close
