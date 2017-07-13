# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 18:29:26 2016

@author: Farid
"""

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
from collections import OrderedDict

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
array_of_cost = []
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
    def updateADADELTA(self,grads, params, eta=1.0, rho=0.95, epsilon=1e-6):
        updates = OrderedDict()
        one = T.constant(1)
        for param, grad in zip(self.params, grads):
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable)
            # update accu (as in rmsprop)
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates[accu] = accu_new
    
            # compute parameter update, using the 'old' delta_accu
            update = (grad * T.sqrt(delta_accu + epsilon) /
                      T.sqrt(accu_new + epsilon))
            updates[param] = param - eta * update
    
            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
            updates[delta_accu] = delta_accu_new
        return updates
    def adadelta(self,grads, learning_rate=1.0, rho=0.95, epsilon=1e-6):
        """ Adadelta updates
        Scale learning rates by the ratio of accumulated gradients to accumulated
        updates, see [1]_ and notes for further description.
        Parameters
        ----------
        loss_or_grads : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        rho : float or symbolic scalar
            Squared gradient moving average decay factor
        epsilon : float or symbolic scalar
            Small value added for numerical stability
        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression
        Notes
        -----
        rho should be between 0 and 1. A value of rho close to 1 will decay the
        moving average slowly and a value close to 0 will decay the moving average
        fast.
        rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
        work for multiple datasets (MNIST, speech).
        In the paper, no learning rate is considered (so learning_rate=1.0).
        Probably best to keep it at this value.
        epsilon is important for the very first update (so the numerator does
        not become 0).
        Using the step size eta and a decay factor rho the learning rate is
        calculated as:
        .. math::
           r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
           \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                                 {\sqrt{r_t + \epsilon}}\\\\
           s_t &= \\rho s_{t-1} + (1-\\rho)*(\\eta_t*g)^2
        References
        ----------
        .. [1] Zeiler, M. D. (2012):
               ADADELTA: An Adaptive Learning Rate Method.
               arXiv Preprint arXiv:1212.5701.
        """
        #grads = get_or_compute_grads(loss_or_grads, params)
        grads = grads
        updates = OrderedDict()
        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)
    
        for param, grad in zip(self.params,grads):
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable)
    
            # update accu (as in rmsprop)
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates[accu] = accu_new
    
            # compute parameter update, using the 'old' delta_accu
            update = (grad * T.sqrt(delta_accu + epsilon) /
                      T.sqrt(accu_new + epsilon))
            updates[param] = param - learning_rate * update
    
            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
            updates[delta_accu] = delta_accu_new
    
        return updates
   
    def image_histogram_equalization(image, number_bins=256):
        # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    
        # get image histogram
        image = np.asarray(image)
        image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize
        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        return image_equalized.reshape(image.shape),cdf
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
        #updates = self.adadelta(grads)
        updates = [(param, param-eta*grad) for param, grad in zip(self.params, grads)]
        print "ok5"
        # Do the actual training
        array_of_cost = []
        i = T.lscalar()
        filename = 'trainingdatagrayhisteqed'
        the_data = load_file(filename)
        the_data = shareddat(the_data)
        training_x,training_y = the_data
        #training_x = self.image_histogram_equalization(training_x)
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                    })
        for epoch in xrange(epochs):
#            if epoch<60:
#                self.wa = 9.0
#            else:
#                self.wa = 1.0
            
            for minibatch_index in xrange(num_training_batches):
                    iteration = num_training_batches*epoch+minibatch_index
                    if iteration % 100 == 0:
                        print("Training mini-batch number {0}".format(iteration))
                    cost_ij = train_mb(minibatch_index)
                    array_of_cost.append(cost_ij)
                    print minibatch_index
                    #self.Save(filename='cross-val-train2')
            write_something(epoch,filename='state-epochawalsampai80.txt')
                                                  
        print("Finished training network.")
        plt.plot(array_of_cost)
        plt.xlabel('number of minibatch')
        plt.ylabel('cost')
        plt.show()
        save_net(array_of_cost,'trainutuhawalsampai80')
        
    def test_net(self,test_data):
        test_x, test_y = test_data
        num_of_minibatch = size(test_data)/135
        i = T.lscalar()
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_precision = theano.function(
            [i], self.layers[-1].precision(self.y),
            givens={
                self.x:test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_recall = theano.function(
            [i], self.layers[-1].recall(self.y),
            givens={
                self.x:test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_acc = np.mean([test_mb_accuracy(j) for j in xrange(num_of_minibatch)])
        test_prec = np.mean([test_mb_precision(j) for j in xrange(num_of_minibatch)])
        test_rec = np.mean([test_mb_recall(j) for j in xrange(num_of_minibatch)])
        print test_acc
        print test_prec
        print test_rec
    def Save(self,filename):
        f = open(filename,'wb')
        cPickle.dump(self.__dict__,f,2)
        f.close()
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
            array_of_pred.append(pred.reshape(self.mini_batch_size,16,16))
        #array_of_pred = T.round(array_of_pred)
        array_of_pred = np.asarray(array_of_pred)
        array_of_pred = array_of_pred.reshape((num_test_batches*self.mini_batch_size,16,16))
        for l in xrange(self.mini_batch_size*num_test_batches):
                name = str(array_of_testp[l])
                misc.imsave('E:\\BackUp-Belum dirapikan\\Sekolah\\SKRIPSI\\DATA\\hasilcrossvalbuffer\\'+name+'.tiff', array_of_pred[l])


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
def load_file(filename="sampah2"):
    f = open(filename,'rb')
    net = cPickle.load(f)
    f.close()
    return net
def write_something(the_text,filename='tulisan.txt'):
    f = open( filename, 'w' )
    f.write(str(the_text))
    f.close()

start = time.clock()
#net = Network([ConvPoolLayer(image_shape=(135,1, 64, 64),
#                             filter_shape=(64, 1, 16, 16), 
#                             poolsize=(2, 2),
#                             stride=(4,4), activation_fn=ReLU), 
#               ConvPoolLayer(image_shape=(135, 64, 6, 6), 
#                             filter_shape=(112, 64, 4, 4), 
#                             poolsize=(1, 1), activation_fn=ReLU), 
#               ConvPoolLayer(image_shape=(135, 112, 3, 3), 
#                             filter_shape=(80, 112, 3, 3), 
#                             poolsize=(1, 1), activation_fn=ReLU), 
#               FullyConnectedLayer(n_in=80*1*1, n_out=4096,activation_fn=ReLU), 
#               FullyConnectedLayer(n_in=4096, n_out=256,p_dropout=0.5)], 135)
#net = Network([FullyConnectedLayer(n_in=3*64*64, n_out=8192,activation_fn=ReLU),
#               FullyConnectedLayer(n_in=8192, n_out=16)],100)
#net = Network([ConvPoolLayer(image_shape=(111,3, 64, 64), filter_shape=(64, 3, 12, 12), poolsize=(3, 3), activation_fn=ReLU), FullyConnectedLayer(n_in=64*17*17, n_out=256)], 111)
#
#net.y = T.fmatrix('y')
#print net.y.dtype
net = load_file(filename='BaruCobaHistoEqPadaTrainingSet3-285')
#net.train_SGD(40,255, 135,  0.005, lmbda=0.0002)
#save_net(net,filename='BaruCobaHistoEqPadaTrainingSet3-285')
the_data = load_file(filename='testdatagrayhisteqed')
the_data = shareddat(the_data)
#net.test_net(the_data)
f = open('urutangambar','rb')
array_of_valp = cPickle.load(f)
f.close()
array_of_valp = array_of_valp[0:2700]
#print net.mini_batch_size
net.predict(the_data,array_of_valp)
#urutan = T.arange(1,8000000).eval()
#random.shuffle(urutan)
#save_net(urutan,filename='urutanacak')
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


#diganti patchnya (bandingkan 16, 8, dan 4)
#diganti output 3x3
#diganti cost-functionnya


