import pickle
import numpy

import lasagne
from selector import *


class Model:
    def __init__(self, n_chunks, input_width, input_height, output_dim, batch_size):

        print("Building model...")

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, 3, input_width, input_height),
            )
        self.l_ins = [l_in]

        l_conv1 = lasagne.layers.Conv2DLayer(
            l_in,
            num_filters=16,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1,
            num_filters=32,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

        l_conv3 = lasagne.layers.Conv2DLayer(
            l_pool2,
            num_filters=64,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, ds=(2, 2))

        l_conv4 = lasagne.layers.Conv2DLayer(
            l_pool3,
            num_filters=128,
            filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool4 = lasagne.layers.MaxPool2DLayer(l_conv4, ds=(2, 2))

        l_conv5 = lasagne.layers.Conv2DLayer(
            l_pool4,
            num_filters=256,
            filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool5 = lasagne.layers.MaxPool2DLayer(l_conv5, ds=(2, 2))

        l_conv6 = lasagne.layers.Conv2DLayer(
            l_pool5,
            num_filters=256,
            filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool6 = lasagne.layers.MaxPool2DLayer(l_conv6, ds=(2, 2))

        l_hidden1 = lasagne.layers.DenseLayer(
            l_pool6,
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )

        l_hidden2 = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )

        self.l_out = lasagne.layers.DenseLayer(
            l_hidden2,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )

    def save(self, path):
        all_params = lasagne.layers.get_all_params(self.l_out)
        all_param_values = [p.get_value() for p in all_params]
        pickle.dump(all_param_values, open(path, "wb"))

    def load(self, path):
        all_param_values = pickle.load( open( path, "rb" ) )
        all_params = lasagne.layers.get_all_params(self.l_out)
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)