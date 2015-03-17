import pickle
import numpy

import lasagne
from selector import *
from network import Network


class Network_8(Network):
        def create_model_virtual(self):

            l_conv1 = lasagne.layers.Conv2DLayer(
                self.concat_in,
                num_filters=16,
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.1),
                b=lasagne.init.Constant(0),
                untie_biases=True,
                )
            l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

            l_conv2 = lasagne.layers.Conv2DLayer(
                l_pool1,
                num_filters=32,
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.1),
                b=lasagne.init.Constant(0),
                untie_biases=True,
                )
            l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

            l_conv3 = lasagne.layers.Conv2DLayer(
                l_pool2,
                num_filters=64,
                filter_size=(3, 3),
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
                num_filters=128,
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.1),
                b=lasagne.init.Constant(0),
                untie_biases=True,
                )

            l_conv6 = lasagne.layers.Conv2DLayer(
                l_conv5,
                num_filters=128,
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.1),
                b=lasagne.init.Constant(0),
                untie_biases=True,
                )

            l_conv6_dropout = lasagne.layers.DropoutLayer(
                l_conv6,
                p=0.5,
                )

            l_hidden1 = lasagne.layers.DenseLayer(
                l_conv6_dropout,
                num_units=256,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.1),
                b=lasagne.init.Constant(0),
                )

            l_hidden1_dropout = lasagne.layers.DropoutLayer(
                l_hidden1,
                p=0.5,
                )

            concat_out = self.combine_features(l_hidden1_dropout)

            l_hidden2 = lasagne.layers.DenseLayer(
                concat_out,
                num_units=128,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.1),
                b=lasagne.init.Constant(0),
                )

            l_hidden2_dropout = lasagne.layers.DropoutLayer(
                l_hidden2,
                p=0.5,
                )

            self.l_out = lasagne.layers.DenseLayer(
                l_hidden2_dropout,
                num_units=self.output_dim,
                nonlinearity=lasagne.nonlinearities.softmax,
                W=lasagne.init.Uniform(0.1),
                b=lasagne.init.Constant(0),
                )