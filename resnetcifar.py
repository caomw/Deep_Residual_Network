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
from lasagne.layers import GlobalPoolLayer

sys.setrecursionlimit(10000)


def model_resnet(input_var=None, n=1, num_filters=8, nettype='bottleneckdeep2'):
    # Setting up layers
    conv = lasagne.layers.dnn.Conv2DDNNLayer 
    nonlinearity = lasagne.nonlinearities.linear
    sumlayer = lasagne.layers.ElemwiseSumLayer
    batchnorm = BatchNormLayer.batch_norm
    # Batchnorm is applied before activation function
    def convLayer(l, num_filters, filter_size=(1, 1), stride=(1, 1), nonlinearity=nonlinearity, pad='same', W=lasagne.init.HeNormal(gain='relu')):
        l = conv(l, num_filters=num_filters, filter_size=filter_size,
            stride=stride, nonlinearity=nonlinearity,
            pad=pad, W=W)
        l = batchnorm(l)
        return l
    
    def poolLayer(l, pool_size=(3,3), stride=(2,2), pad=1):
        l = lasagne.layers.dnn.MaxPool2DDNNLayer(l, pool_size=pool_size, pad=pad,stride=stride)
        return l 
    # Bottleneck architecture as descriped in paper

    def bottleneckInceptionv3(l, num_filters, stride=(1,1), nonlinearity=nonlinearity):
        l1 = convLayer(l, num_filters=num_filters, nonlinearity=nonlinearity)
        l2 = convLayer(l1, num_filters=num_filters, filter_size=(1, 7), stride=stride, nonlinearity=nonlinearity)
        l2_1 = convLayer(l2, num_filters=num_filters, filter_size=(7, 1), nonlinearity=nonlinearity)
        l3 = convLayer(l1, num_filters=num_filters, filter_size=(1, 7), stride=stride, nonlinearity=nonlinearity) 
        l3_1 = convLayer(l3, num_filters=num_filters, filter_size=(7, 1), nonlinearity=nonlinearity)
        l3_2 = convLayer(l3_1, num_filters=num_filters, filter_size=(1, 7), nonlinearity=nonlinearity) 
        l3_3 = convLayer(l3_2, num_filters=num_filters, filter_size=(7, 1), nonlinearity=nonlinearity) 
        l4 = convLayer(l, num_filters=num_filters, stride=stride, nonlinearity=nonlinearity)
        l5 = lasagne.layers.ConcatLayer(incomings=[l2_1, l3_3, l4], axis=1)
        l = convLayer(l5, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l
    def bottleneckDeep(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(l, num_filters=num_filters, stride=stride, nonlinearity=nonlinearity)
        l = convLayer(l, num_filters=num_filters, filter_size=(3, 3), nonlinearity=nonlinearity)
        l = convLayer(l, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l

    def bottleneckDeep2(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(l, num_filters=num_filters, nonlinearity=nonlinearity)
        l = convLayer(l, num_filters=num_filters, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l = convLayer(l, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l

    def bottleneckShallow(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(l, num_filters=num_filters*4, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l = convLayer(l, num_filters=num_filters*4, filter_size=(3, 3), nonlinearity=nonlinearity)
        return l

    def bottleneckInceptionDropout(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l_1 = convLayer(l, num_filters=num_filters*4, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l_2 = convLayer(l_1, num_filters=num_filters*4, filter_size=(3, 3), nonlinearity=nonlinearity)
        l1 = convLayer(l, num_filters=num_filters*4, stride=stride, nonlinearity=nonlinearity)
        l2 = lasagne.layers.DropoutLayer(l1, p=0.5, rescale=True)
        l3 = lasagne.layers.ConcatLayer(incomings=[l_2, l2], axis=1)
        l4 = convLayer(l3, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l4

    def bottleneckInception(l, num_filters, stride=(1,1), nonlinearity=nonlinearity):
        l1 = convLayer(l, num_filters=num_filters, nonlinearity=nonlinearity)
        l2 = convLayer(l1, num_filters=num_filters, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l3 = convLayer(l1, num_filters=num_filters, filter_size=(5, 5), stride=stride, nonlinearity=nonlinearity) 
        l4 = convLayer(l, num_filters=num_filters, stride=stride, nonlinearity=nonlinearity)
        l5 = lasagne.layers.ConcatLayer(incomings=[l2, l3, l4], axis=1)
        l = convLayer(l5, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l
    
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
        bottleneck = bottleneckInceptionv3
    elif nettype == 'bottleneckdeep':
        bottleneck = bottleneckDeep
    elif nettype == 'bottleneckdeep2':
        bottleneck = bottleneckDeep2
    elif nettype == 'bottleneckshallow':
        bottleneck = bottleneckShallow
    elif nettype == 'bottleneckdropout':
        bottleneck = bottleneckDropout
    else:
        bottleneck = bottleneckInception
    #bottleneck = bottleneckDeep2
    #bottleneck = bottleneckShallow
    #bottleneck = bottleneckInceptionDropout

    def bottlestack(l, n, num_filters):
        for _ in range(n):
            l = sumlayer([bottleneck(l, num_filters=num_filters), l])
        return l

    # Building the network
    l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    l1 = convLayer(l_in, num_filters=num_filters*4, filter_size=(3, 3)) 

    l1_bottlestack = bottlestack(l1, n=n-1, num_filters=num_filters) 
    l1_residual = convLayer(l1_bottlestack, num_filters=num_filters*4*2, stride=(2, 2), nonlinearity=None) 

    l2 = sumlayer([bottleneck(l1_bottlestack, num_filters=num_filters*2, stride=(2, 2)), l1_residual])
    l2_bottlestack = bottlestack(l2, n=n, num_filters=num_filters*2)
    l2_residual = convLayer(l2_bottlestack, num_filters=num_filters*2*2*4, stride=(2, 2), nonlinearity=None)

    l3 = sumlayer([bottleneck(l2_bottlestack, num_filters=num_filters*2*2, stride=(2, 2)), l2_residual])
    l3_bottlestack = bottlestack(l3, n=n, num_filters=num_filters*2*2)
        
    lp = GlobalPoolLayer(l3_bottlestack)

    network = lasagne.layers.DenseLayer(lp, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    return network



def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
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


def train_val(network, input_var, target_var, num_epochs, X_train, Y_train, X_test, Y_test):
    train_prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = 0.1
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=sh_lr, momentum=0.9)

    # Create a loss expression for validation/testing.
    # deterministic forward pass through the network, disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, 500, shuffle=True, augment=True):
    	    inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
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

def saveModel(modelname, network):
    #modelname: filename.npz
    np.savez(modelname, *lasagne.layers.get_all_param_values(network))

def loadModel(modelname, network):
    with np.load(modelname) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

#-------------------------  Main program -------------------------------------#

def main(n=1, num_filters=22, num_epochs=99, bottlenecktype = 'bootleneckdeep'):
    assert n>=0
    assert num_filters>0
    assert num_epochs>0
    print("Amount of bottlenecks: %d" % n)
    # Load the dataset
    print("Loading data...")
    X_train, Y_train, X_test, Y_test = load_data()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.lvector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = model_resnet(input_var, n, num_filters, bottlenecktype)
    all_layers = lasagne.layers.get_all_layers(network)
    num_params = lasagne.layers.count_params(network)
    num_conv = 0
    num_nonlin = 0
    num_input = 0
    num_batchnorm = 0
    num_elemsum = 0
    num_dense = 0
    num_unknown = 0
    print("  layer output shapes:")
    for layer in all_layers:
	name = string.ljust(layer.__class__.__name__, 32)
	print("    %s %s" %(name, lasagne.layers.get_output_shape(layer)))
	if "Conv2D" in name:
	    num_conv += 1
	elif "NonlinearityLayer" in name:
	    num_nonlin += 1
	elif "InputLayer" in name:
	    num_input += 1
	elif "BatchNormLayer" in name:
	    num_batchnorm += 1
	elif "ElemwiseSumLayer" in name:
	    num_elemsum += 1
	elif "DenseLayer" in name:
	    num_dense += 1
	else:
	    num_unknown += 1
    print("  no. of InputLayers: %d" % num_input)
    print("  no. of Conv2DLayers: %d" % num_conv)
    print("  no. of BatchNormLayers: %d" % num_batchnorm)
    print("  no. of NonlinearityLayers: %d" % num_nonlin)
    print("  no. of DenseLayers: %d" % num_dense)
    print("  no. of ElemwiseSumLayers: %d" % num_elemsum)
    print("  no. of Unknown Layers: %d" % num_unknown)
    print("  total no. of layers: %d" % len(all_layers))
    print("  no. of parameters: %d" % num_params)
    #------------------------ Training and Validation -------------------------------------------#
    train_val(network, input_var, target_var, num_epochs, X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual neural network on MNIST using Lasagne.")
        print("Usage: %s [NUM_BOTTLENECKS] [NUM_FILTERS] [EPOCHS] [BOTTLENECKTYPE]" % sys.argv[0])
        print()
        print("NUM_BOTTLENECKS: Define amount of bottlenecks with integer, e.g. 3")
	print("NUM_FILTERS: Defines the amount of filters in the first layer(doubled at each filter halfing)")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
	if len(sys.argv) > 2:
	    kwargs['num_filters'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['num_epochs'] = int(sys.argv[3])
        if len(sys.argv) > 4:
            kwargs['bottlenecktype'] = sys.argv[4]
        main(**kwargs)
