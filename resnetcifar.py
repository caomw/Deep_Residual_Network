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
import resnetmodel 

sys.setrecursionlimit(10000)


#-------------------------  Main program -------------------------------------#

def main(n=1, num_filters=22, num_epochs=99, bottlenecktype = 'bootleneckShallow'):
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
    model = resnetmodel.ResNet()
    network = model.model_resnet(input_var, n, num_filters, bottlenecktype)
    all_layers = lasagne.layers.get_all_layers(network)
    num_params = lasagne.layers.count_params(network)
    num_conv = 0
    num_nonlin = 0
    num_input = 0
    num_batchnorm = 0
    num_elemsum = 0
    num_dense = 0
    num_unknown = 0
    batch_size = 64
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
    model.train_val(network, input_var, target_var, num_epochs, X_train, Y_train, X_test, Y_test, batch_size)

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
