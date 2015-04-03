import pickle
import numpy
import theano.tensor as T

import lasagne
from custom_layers import *
from network import Network


def l2_avg(x, axis):
    return T.sqrt(T.sum(x**2+0.0000001, axis=axis))


class Network_9(Network):
    """
    Small network to evaluate multi-view techniques
    """
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

        # n_branches_to_cut = 2
        # dropout_branches = DropoutBranchLayer(l_pool3, self.n_branches,
        #                                       n_branches_to_cut, self.batch_size)

        # concat_out = self.combine_features(l_pool3)

        concat_out = self.pool_over_branches(l_pool3, T.max)
        # concat_out = self.aggregate_over_branches(l_pool3)
        # concat_out = self.aggregate_over_branches2(l_pool3)

        l_hidden1 = lasagne.layers.DenseLayer(
            concat_out,
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )

        l_hidden1_dropout = lasagne.layers.DropoutLayer(
            l_hidden1,
            p=0.5,
            )

        self.l_out = lasagne.layers.DenseLayer(
            l_hidden1_dropout,
            num_units=self.output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )