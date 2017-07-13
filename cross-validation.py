# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:31:03 2016

@author: Farid
"""

"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

"""

#### Libraries
# Standard library
#import pickle
#import gzip
#import sys
#import pprint
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

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
#from theano.tensor import tanh
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
def shareddat(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, 'float32')
#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.wa = 1.0  #cost weight of true class
        self.wb = 1.0  #cost weight of false class
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        #self.vs = [v for layer in self.layers for v in layer.vs]
        self.x = T.ftensor4("x")
        self.y = T.fmatrix("y")
        print self.x.dtype
        print self.y.dtype
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def train_SGD(self, num_training_batches, epochs, mini_batch_size, eta, lmbda=0.0, alpha=0.9):
        """Train the network using mini-batch stochastic gradient descent."""
                
        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        print "ok2"
        #define output cost
        cost = T.fscalar()
        cost = self.layers[-1].cost(self.y,self.wa,self.wb) + 0.5*lmbda*l2_norm_squared/mini_batch_size
        print "ok3"
        #compute the gradient
        grads = T.grad(cost, self.params)
        print "ok4"
        #do updating parameters
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]
        print "ok5"
        # Do the actual training
        array_of_cost = []
        training_pointer = 0
        array_of_pointer = T.arange(745201,753301).eval()
        random.shuffle(array_of_pointer)
        o = open('urutangambar', 'wb')
        cPickle.dump(array_of_pointer, o, protocol=cPickle.HIGHEST_PROTOCOL)
        o.close()
        trainingp = array_of_pointer[0:6481]
        testp = array_of_pointer[6481:8101]
        for epoch in xrange(epochs):
            if epoch<30:
                self.wa = 9.0
            else:
                self.wa = 1.0
            training_pointer = 0
            for minibatch_index in xrange(num_training_batches):
                    iteration = num_training_batches*epoch+minibatch_index
                   #creating training data
                    #initializing array for x
                    initial_input_file = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_64_inputs\\'+str(trainingp[training_pointer])+'.tiff'
                    initial_target_file = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_16_maps\\'+str(trainingp[training_pointer])+'.tif'        
                    img = misc.imread(initial_input_file)
                    img = np.asarray(img)
                    img = img.transpose(2,0,1).reshape(1,3,64,64)
                    dataxnya = np.asarray(img, dtype='float32')
                    #initializing array for y
                    imgy = misc.imread(initial_target_file)
                    imgy = np.asarray(imgy)
#                    imgy = imgy[1:15,1:15]
#                    imgy = np.reshape(imgy,(1,196))
                    imgy = np.reshape(imgy,(1,256))
                    dataynya = np.asarray(imgy, dtype='float32')
                    training_pointer = training_pointer+1
                    #finishing array x and y
                    for urutan in range(2,mini_batch_size+1):
                        #x
                        base = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_64_inputs\\'
                        extension = '.tiff'
                        link = base+str(trainingp[training_pointer])+extension
                        img = misc.imread(link)
                        img = np.asarray(img, dtype='float32')
                        img = img.transpose(2,0,1).reshape(1,3,64,64)
                        dataxnya = np.append(dataxnya,img,axis=0)
                        #y
                        basey = 'E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\training_16_maps\\'
                        extensiony = '.tif'
                        linky = basey+str(trainingp[training_pointer])+extensiony
                        imgy = misc.imread(linky)
                        imgy = np.asarray(imgy,dtype='float32')
#                        imgy = imgy[1:15,1:15]
#                        imgy = np.reshape(imgy,(1,196))
                        imgy = np.reshape(imgy,(1,256))
                        dataynya = np.append(dataynya,imgy,axis=0)
                        training_pointer = training_pointer+1
                    dataynya[dataynya>0] = 1.0
                    tra_data = [dataxnya,dataynya]
                    tra_data = shareddat(tra_data)
                    training_x, training_y = tra_data
                    train_mb = theano.function(
                        [], cost, updates=updates,
                        givens={
                            self.x:
                            training_x,
                            self.y:
                            training_y
                                })
                    if iteration % 100 == 0:
                        print("Training mini-batch number {0}".format(iteration))
                    cost_ij = train_mb()
                    array_of_cost.append(cost_ij)
                    print minibatch_index
#            test_precision = test_precision()
#            array_of_tesprec.append(test_precision)
#            print("Epoch {0}: test precision {1:.2%}".format(
#                epoch, test_precision))
#            if test_precision >= best_test_precision:
#                print("This is the best test precision to date.")
#                best_test_precision = test_precision
#                best_iteration_precision = epoch
#            test_recall = test_recall()
#            array_of_tesrec.append(test_recall)
#            print("Epoch {0}: test recall {1:.2%}".format(
#                epoch, test_recall))
#            if test_recall >= best_test_recall:
#                print("This is the best validation recall to date.")
#                best_test_recall = test_recall
#                best_iteration_recall = epoch
#            test_accuracy = test_accuracy
#            array_of_tesacc.append(test_accuracy)
#            print("Epoch {0}: test accuracy {1:.2%}".format(
#                epoch, test_accuracy))
#            if test_accuracy >= best_test_accuracy:
#                print("This is the best test accuracy to date.")
#                best_test_accuracy = test_accuracy
#                best_iteration_accuracy = epoch                     
                        
        print("Finished training network.")
#        print("Best test precision of {0:.2%} obtained at epoch {1}".format(
#            best_test_precision, best_iteration_precision))
#        print("Best test recall of {0:.2%} obtained at epoch {1}".format(
#            best_test_recall, best_iteration_recall))
#        print("Best test accuracy of {0:.2%} obtained at epoch {1}".format(
#            best_test_accuracy, best_iteration_accuracy))
        plt.plot(array_of_cost)
        plt.xlabel('number of epoch')
        plt.ylabel('cost')
        plt.show()
#        plt.plot(array_of_tesprec)
#        plt.xlabel('number of epoch')
#        plt.ylabel('test precision')
#        plt.show()
#        plt.plot(array_of_tesrec)
#        plt.xlabel('number of epoch')
#        plt.ylabel('test recall')
#        plt.show()
#        plt.plot(array_of_tesacc)
#        plt.xlabel('number of epoch')
#        plt.ylabel('test accuracy')
#        plt.show()
    def validate(self, validation_data):
        i = T.lscalar()
        threshold = T.fscalar()
        validation_x, validation_y = validation_data 
        num_validation_batches = size(validation_data)/self.mini_batch_size
        array_of_valacc = []
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_precision = theano.function(
            [i,threshold], self.layers[-1].precision(self.y,threshold),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_recall = theano.function(
            [i,threshold], self.layers[-1].recall(self.y,threshold),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        for minibatch_index in xrange(num_validation_batches):
            validation_accuracy = np.mean(
                [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
            array_of_valacc.append(validation_accuracy)
#            print("Epoch {0}: validation accuracy {1:.2%}".format(
#                epoch, validation_accuracy))
        threshold = 0.1
        aofprec = []
        aofrec = []
        for bilangan in xrange(9):
            for minibatch_index in xrange(num_validation_batches):
                validation_precision = np.mean(
                [validate_mb_precision(j,threshold) for j in xrange(num_validation_batches)])
                aofprec.append(validation_precision)
            for minibatch_index in xrange(num_validation_batches):
                validation_recall = np.mean(
                [validate_mb_recall(j,threshold) for j in xrange(num_validation_batches)])
                aofrec.append(validation_recall)
            threshold = threshold+0.1
        plt.plot(aofprec,aofrec)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.show
        
#        print validation_accuracy
#        print validation_precision
#        print validation_recall
    def test_net(self,test_data):
        test_x, test_y = test_data
        test_mb_accuracy = theano.function(
            [], self.layers[-1].accuracy(self.y),
            givens={
                self.x:test_x,
                self.y:test_y
            })
        test_acc = test_mb_accuracy()
        print test_acc
    def predict(self,test_data,array_of_testp):
        test_x, test_y = test_data
        num_test_batches = size(test_data)/self.mini_batch_size
        i = T.lscalar()
        test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        print "ok"
        array_of_pred = []
        for k in xrange(num_test_batches):
            pred = test_mb_predictions(k)
            print "ok2"
            array_of_pred.append(pred.reshape(100,16,16))
        #array_of_pred = T.round(array_of_pred)
        array_of_pred = np.asarray(array_of_pred)
        array_of_pred = array_of_pred.reshape((num_test_batches*self.mini_batch_size,16,16))
        for l in xrange(self.mini_batch_size*num_test_batches):
                name = str(array_of_testp[l])
                misc.imsave('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\hasilpredswithprec\\'+name+'.tiff', array_of_pred[l])
        
        

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid,stride=(1,1)):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        self.stride = stride
        # initialize weights and biases
        #n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        n_in = (np.prod(image_shape))
        self.w = theano.shared(
            np.asarray(
                np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * np.sqrt(2.0/n_in),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]
#        self.vw = theano.shared(
#            np.asarray(
#                np.zeros(shape=(filter_shape)),dtype=theano.config.floatX),borrow=True)
#        self.vb = theano.shared(
#            np.asarray(
#                np.zeros(shape=(filter_shape[0],)),dtype=theano.config.floatX),borrow=True)
#        self.vs = [self.w, self.b]
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape, subsample=self.stride)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.randn(n_in,n_out) * np.sqrt(2.0/n_in),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
#        self.vw = theano.shared(
#            np.asarray(
#                np.zeros(shape=(n_in, n_out)),
#                dtype=theano.config.floatX),
#            name='vw', borrow=True)
#        self.vb = theano.shared(
#            np.asarray(
#                np.zeros(shape=(n_out,)),
#                dtype=theano.config.floatX),
#            name='vb', borrow=True)
#        self.vs = [self.vw, self.vb]
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        #T.argmax(self.output, axis=1)
        self.y_out = self.output
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)
    
#    def cost(self, net):
#        "Return the log-likelihood cost."
#        print self.output_dropout
#        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]),:,:])

    def cost(self, y, wa, wb):
        #return T.mean(net.y*T.log(self.output_dropout)+(1-net.y)*T.log(1-self.output_dropout))
        conditional_output = T.switch(T.eq(y, 1), T.nnet.binary_crossentropy(self.output_dropout, y)*wa, T.nnet.binary_crossentropy(self.output_dropout, y)*wb)
        return T.mean(conditional_output)
    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
#        y_outpred = self.y_out.eval()
#        y_outpred[y_outpred>0.8] = 1
#        y_outpred[y_outpred<=0.8] = 0
        #avg = T.mean(self.y_out)
        #conditional_output = T.switch(T.lt(avg,self.y_out), 1.0, 0.0)
        return T.mean(T.eq(y, T.round(self.y_out)))
    def precision(self,y,threshold=0.5):
#        y_outpred = self.y_out.eval()
#        y_outpred[y_outpred>0.8] = 1
#        y_outpred[y_outpred<=0.8] = 0
        #avg = T.mean(self.y_out)
        #stddev = T.std(self.y_out)
        #conditional_output = T.switch(T.lt(threshold,self.y_out), 1.0, 0.0)
        divider = T.sum(T.round(self.y_out))
        dividee = T.sum(T.eq(T.round(self.y_out),1)*T.eq(y,1))
#        divider = T.sum(T.round(self.y_out))
#        dividee = T.sum(T.eq(T.round(self.y_out),1)*T.eq(y,1))
        return dividee/divider
        #return dividee/divider
    def recall(self,y,threshold=0.5):
#        y_outpred = self.y_out.eval()
#        y_outpred[y_outpred>0.8] = 1
#        y_outpred[y_outpred<=0.8] = 0
        #avg = T.mean(self.y_out)
        #conditional_output = T.switch(T.lt(threshold,self.y_out), 1, 0)
        divider = T.sum(y)
        dividee = T.sum(T.eq(T.round(self.y_out),1)*T.eq(y,1))
#        divider = T.sum(y)
#        dividee = T.sum(T.eq(T.round(self.y_out),1)*T.eq(y,1))
        return dividee/divider

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return T.nnet.binary_crossentropy(output_dropout, y).mean()

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
def save_net(net,filename='sampah3'):
    f = open(filename, 'wb')
    cPickle.dump(net, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
def load_net(filename="sampah2"):
    f = open(filename,'rb')
    net = cPickle.load(f)
    f.close()
    return net

start = time.clock()
net = Network([ConvPoolLayer(image_shape=(162,3, 64, 64),
                             filter_shape=(64, 3, 16, 16), 
                             poolsize=(2, 2),
                             stride=(4,4), activation_fn=ReLU), 
               ConvPoolLayer(image_shape=(162, 64, 6, 6), 
                             filter_shape=(112, 64, 4, 4), 
                             poolsize=(1, 1), activation_fn=ReLU), 
               ConvPoolLayer(image_shape=(162, 112, 3, 3), 
                             filter_shape=(80, 112, 3, 3), 
                             poolsize=(1, 1), activation_fn=ReLU), 
               FullyConnectedLayer(n_in=80*1*1, n_out=4096,activation_fn=ReLU), 
               FullyConnectedLayer(n_in=4096, n_out=256,p_dropout=0.5)], 162)
#net = Network([FullyConnectedLayer(n_in=3*64*64, n_out=8192,activation_fn=ReLU),
#               FullyConnectedLayer(n_in=8192, n_out=16)],100)
#net = Network([ConvPoolLayer(image_shape=(111,3, 64, 64), filter_shape=(64, 3, 12, 12), poolsize=(3, 3), activation_fn=ReLU), FullyConnectedLayer(n_in=64*17*17, n_out=256)], 111)
#net = load_net(filename='hasilprediksi21')
#net.y = T.fmatrix('y')
#print net.y.dtype
net.train_SGD( 40, 285, 162,  0.005, lmbda=0.0002)
save_net(net,filename='cross-val-train1')
#net.validate(val_data)
#net.predict(val_data,array_of_valp)
end = time.clock()-start
print 'overall time='+str(end)+'seconds'
#net.test_net(tes_data)

#net.test()
#print tra_data
#print val_data
#print tes_data
#print dataynya
#str1 = ''.join(str(e) for e in img)
#print str1
#f = open( 'arraynyabaru.txt', 'w' )
#f.write(str1)
#f.close()

#diganti patchnya (bandingkan 16, 8, dan 4)
#diganti output 3x3
#diganti cost-functionnya


