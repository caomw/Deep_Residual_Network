#!/usr/bin/env python
#----------------------------------------------------#
# Written by Mrinal Haloi,NTU -----------------------#
#----------------------------------------------------#

from __future__ import print_function

import sys
import os
import time
import string

import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.dnn
import BatchNormLayer
from load_cifar10 import load_data


sys.setrecursionlimit(10000)

class ResNet(object):
    def __init__(self):
        self.conv = lasagne.layers.dnn.Conv2DDNNLayer 
        self.nonlinearity = lasagne.nonlinearities.linear
        self.sumlayer = lasagne.layers.ElemwiseSumLayer
        self.batchnorm = BatchNormLayer.batch_norm
        self.pool = lasagne.layers.dnn.MaxPool2DDNNLayer
        self.concat = lasagne.layers.ConcatLayer
        self.dropout = lasagne.layers.DropoutLayer
        self.inputLayer = lasagne.layers.InputLayer
        self.dense = lasagne.layers.DenseLayer
        self.globalpool = lasagne.layers.GlobalPoolLayer
        self.output = lasagne.layers.get_output
        self.cs = lasagne.objectives.categorical_crossentropy
        self.getparams = lasagne.layers.get_all_params
        self.momentum = lasagne.updates.nesterov_momentum


    def convLayer(self, l, num_filters, nonlinearity, filter_size=(1, 1), stride=(1, 1), pad='same', W=lasagne.init.HeNormal(gain='relu')):
        l = self.conv(l, num_filters=num_filters, filter_size=filter_size, stride=stride, nonlinearity=nonlinearity, pad=pad, W=W)
        l = self.batchnorm(l)
        return l
        
    def poolLayer(self, l, pool_size=(3,3), stride=(2,2), pad=1):
        l = self.pool(l, pool_size=pool_size, pad=pad,stride=stride)
        return l 

    def bottleneckInceptionv3(self, l, num_filters, nonlinearity, stride=(1,1)):
        l1 = self.convLayer(l, num_filters, nonlinearity=nonlinearity)
        l2 = selfconvLayer(l1, num_filters, nonlinearity, filter_size=(1, 7), stride=stride)
        l2_1 = self.convLayer(l2, num_filters, nonlinearity,  filter_size=(7, 1))
        l3 = self.convLayer(l1, num_filters, nonlinearity, filter_size=(1, 7), stride=stride) 
        l3_1 = self.convLayer(l3, num_filters, nonlinearity, filter_size=(7, 1))
        l3_2 = self.convLayer(l3_1, num_filters, nonlinearity, filter_size=(1, 7)) 
        l3_3 = self.convLayer(l3_2, num_filters, nonlinearity, filter_size=(7, 1)) 
        l4 = self.convLayer(l, num_filters, stride=stride)
        l5 = self.concat(incomings=[l2_1, l3_3, l4], axis=1)
        l = self.convLayer(l5, num_filters*4, nonlinearity=nonlinearity)
        return l
    def bottleneckDeep(self, l, num_filters, nonlinearity, stride=(1, 1)):
        l = self.convLayer(l, num_filters, nonlinearity, stride=stride)
        l = self.convLayer(l, num_filters, nonlinearity, filter_size=(3, 3))
        l = self.convLayer(l, num_filters*4, nonlinearity=nonlinearity)
        return l

    def bottleneckDeep2(self, l, num_filters, nonlinearity, stride=(1, 1)):
        l = self.convLayer(l, num_filters, nonlinearity=nonlinearity)
        l = self.convLayer(l, num_filters, nonlinearity, filter_size=(3, 3), stride=stride)
        l = self.convLayer(l, num_filters*4, nonlinearity=nonlinearity)
        return l

    def bottleneckShallow(self, l, num_filters, nonlinearity, stride=(1, 1)):
        l = self.convLayer(l, num_filters*4, nonlinearity, filter_size=(3, 3), stride=stride)
        l = self.convLayer(l, num_filters*4, nonlinearity, filter_size=(3, 3))
        return l

    def bottleneckInceptionDropout(self, l, num_filters, nonlinearity, stride=(1, 1)):
        l_1 = self.convLayer(l, num_filters*4, nonlinearity, filter_size=(3, 3), stride=stride)
        l_2 = self.convLayer(l_1, num_filters*4, nonlinearity, filter_size=(3, 3))
        l1 = self.convLayer(l, num_filters*4, nonlinearity, stride=stride)
        l2 = self.dropout(l1, p=0.5, rescale=True)
        l3 = self.concat(incomings=[l_2, l2], axis=1)
        l4 = self.convLayer(l3, num_filters*4, nonlinearity=nonlinearity)
        return l4

    def bottleneckInception(self, l, num_filters, nonlinearity, stride=(1,1)):
        l1 = self.convLayer(l, num_filters, nonlinearity=nonlinearity)
        l2 = self.convLayer(l1, num_filters, nonlinearity, filter_size=(3, 3), stride=stride)
        l3 = self.convLayer(l1, num_filters, nonlinearity, filter_size=(5, 5), stride=stride) 
        l4 = self.convLayer(l, num_filters, nonlinearity, stride=stride)
        l5 = self.concat(incomings=[l2, l3, l4], axis=1)
        l = self.convLayer(l5, num_filters*4, nonlinearity=nonlinearity)
        return l
    def bottlestack(self, l, n, num_filters, bottleneck):
        for _ in range(n):
            l = self.sumlayer([bottleneck(l, num_filters, self.nonlinearity), l])
        return l
    def model_resnet(self, input_var=None, n=1, num_filters=8, nettype='bottleneckdeep2'):
        # Setting up layers
        
        '''def getBottleneck(x):
            return {
                'inceptionv3':bottleneckInceptionv3,
                'bottleneckdeep':bottleneckDeep,
                'bottleneckdeep2':bottleneckDeep2,
                'bottleneckshallow':bottleneckShallow,
                'bottleneckdropout':bottleneckInceptionDropout,
                'bottleneckinception':bottleneckInception,
            }.get(x, 'bottleneckdeep2')

        bottleneck = getBottleneck(nettype)    
        print('Selecting {}'.format(nettype))
        '''
        if nettype == 'inceptionv3':
            bottleneck = self.bottleneckInceptionv3
        elif nettype == 'bottleneckdeep':
            bottleneck = self.bottleneckDeep
        elif nettype == 'bottleneckdeep2':
            bottleneck = self.bottleneckDeep2
        elif nettype == 'bottleneckshallow':
            bottleneck = self.bottleneckShallow
        elif nettype == 'bottleneckdropout':
            bottleneck = self.bottleneckDropout
        else:
            bottleneck = self.bottleneckInception

        # Building the network
        l_in = self.inputLayer(shape=(None, 3, 32, 32), input_var=input_var)
        l1 = self.convLayer(l_in, num_filters*4, self.nonlinearity, filter_size=(3, 3)) 

        l1_bottlestack = self.bottlestack(l1, n-1, num_filters, bottleneck) 
        l1_residual = self.convLayer(l1_bottlestack, num_filters*4*2, None, stride=(2, 2)) 

        l2 = self.sumlayer([bottleneck(l1_bottlestack, num_filters*2, self.nonlinearity, stride=(2, 2)), l1_residual])
        l2_bottlestack = self.bottlestack(l2, n, num_filters*2, bottleneck)
        l2_residual = self.convLayer(l2_bottlestack, num_filters*2*2*4, None, stride=(2, 2))

        l3 = self.sumlayer([bottleneck(l2_bottlestack, num_filters*2*2, self.nonlinearity, stride=(2, 2)), l2_residual])
        l3_bottlestack = self.bottlestack(l3, n, num_filters*2*2, bottleneck)
            
        lp = self.globalpool(l3_bottlestack)

        network = self.dense(lp, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

        return network



    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False, augment=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if augment:
                padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
                random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
                crops = np.random.random_integers(0,high=8,size=(batchsize,2))
                for r in range(batchsize):
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
                inp_exc = random_cropped
            else:
                inp_exc = inputs[excerpt]

            yield inp_exc, targets[excerpt]


    def train_val(self, network, input_var, target_var, num_epochs, X_train, Y_train, X_test, Y_test, batch_size):
        train_prediction = self.output(network)
        loss = self.cs(train_prediction, target_var)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training
        params = self.getparams(network, trainable=True)
        lr = 0.1
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = self.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)

        # Create a loss expression for validation/testing.
        # deterministic forward pass through the network, disabling dropout layers.
        test_prediction = self.output(network, deterministic=True)
        test_loss = self.cs(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        print("Starting training...")
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, Y_train, batch_size, shuffle=True, augment=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_test, Y_test, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
            if (epoch % 49) == 0:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

    def saveModel(self, modelname, network):
        #modelname: filename.npz
        np.savez(modelname, *lasagne.layers.get_all_param_values(network))

    def loadModel(self, modelname, network):
        with np.load(modelname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
