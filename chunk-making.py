# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 16:48:00 2016

@author: Farid
"""

import time

# Third-party libraries
#import scipy
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
#from theano.ifelse import ifelse
from six.moves import cPickle
from PIL import Image
# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
#from theano.tensor import tanh
import random
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image = np.asarray(image)
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(image.shape)
#f = open('urutangambar','rb')
#array_of_pointers = cPickle.load(f)
#f.close()
#print array_of_pointers
#training_pointer = 0
#urutanchunk = 1
#initial_input_file = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_64_inputs\\'+str(array_of_pointers[training_pointer])+'.tiff'
#initial_target_file = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_16_maps\\'+str(array_of_pointers[training_pointer])+'.tif'        
#img = misc.imread(initial_input_file)
#img = rgb2gray(img)
#img = image_histogram_equalization(img)
#img = np.asarray(img)
#img = img.reshape(1,1,64,64)
#print img
#dataxnya = np.asarray(img, dtype='float32')
##initializing array for y
#imgy = misc.imread(initial_target_file)
#imgy = np.asarray(imgy)
#imgy = np.reshape(imgy,(1,256))
#dataynya = np.asarray(imgy, dtype='float32')
#training_pointer = training_pointer+1
#for i in xrange(2699):
#    #finishing array x and y
#    #x
#    base = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_64_inputs\\'
#    extension = '.tiff'
#    link = base+str(array_of_pointers[training_pointer])+extension
#    img = misc.imread(link)
#    img = rgb2gray(img)
#    img = image_histogram_equalization(img)
#    img = img.reshape(1,1,64,64)
#    print img
#    img = np.asarray(img, dtype='float32')
#    dataxnya = np.append(dataxnya,img,axis=0)
#    #y
#    basey = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_16_maps\\'
#    extensiony = '.tif'
#    linky = basey+str(array_of_pointers[training_pointer])+extensiony
#    imgy = misc.imread(linky)
#    imgy = np.asarray(imgy,dtype='float32')
#    imgy = np.reshape(imgy,(1,256))
#    dataynya = np.append(dataynya,imgy,axis=0)
#    training_pointer = training_pointer+1
#dataynya[dataynya>0] = 1.0
#o = open('testdatagrayhisteqed', 'wb')
#cPickle.dump([dataxnya,dataynya], o, protocol=cPickle.HIGHEST_PROTOCOL)
#o.close()

initial_input_file = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\massachussets validation input\\10228690_15.tiff'
#initial_target_file = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_16_maps\\'+str(array_of_pointers[training_pointer])+'.tif'        
img = misc.imread(initial_input_file)
img = rgb2gray(img)
#img = image_histogram_equalization(img)
img[img>220] = 0
img[img<170] = 0
misc.imsave('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\hasilcrossvalbuffer\\grays4.jpg', img)