# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:49:51 2016

@author: Farid
In my research, I also tried to do filtering on the output to get better results.
This code is for applying the filter (using median filter).
"""

import scipy
import numpy as np
from scipy import misc
from scipy import signal
from theano import tensor as T

thebest = 0
bestp = 0.0
for i in xrange(100):  
    img = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\'+str(i)+'.tiff')
    imgt = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\target_crossval2.tiff')
    img = np.asarray(img)
    filtered = signal.medfilt(img,3)
    dividee = float(T.sum(T.eq(filtered,255)*T.eq(imgt,255)).eval())
    divider_p = T.sum(filtered).eval()/255
    divider_r = T.sum(imgt).eval()/255
    print dividee/divider_p
    print bestp
    if (dividee/divider_p>bestp and dividee/divider_r>0.8):
        print 'hit'
        bestp = dividee/divider_p        
        thebest = i
img = misc.imread('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\'+str(thebest)+'.tiff')
img = np.asarray(img)
filtered = signal.medfilt(img,3)
dividee = float(T.sum(T.eq(filtered,255)*T.eq(imgt,255)).eval())
divider_p = T.sum(filtered).eval()/255
divider_r = T.sum(imgt).eval()/255
print 'precision= '+str(dividee/divider_p)
print 'recall= '+str(dividee/divider_r)
misc.imsave('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\thetargetoftraining\\filtered.tiff',filtered)