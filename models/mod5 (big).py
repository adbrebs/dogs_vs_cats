import numpy

import lasagne
from selector import *


class Model:
    def __init__(self, input_width, input_height, output_dim, batch_size):

        print("Building model...")

        l_in1 = lasagne.layers.InputLayer(
            shape=(batch_size, 3, input_width, input_height),
            )

        l_in2 = lasagne.layers.InputLayer(
            shape=(batch_size, 3, input_width, input_height),
            )
        self.l_ins = [l_in1, l_in2]

        concat_in = lasagne.layers.concat([l_in1, l_in2], axis=0)

        l_conv1 = lasagne.layers.Conv2DLayer(
            concat_in,
            num_filters=16,
            filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1,
            num_filters=32,
            filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))



        ###
        select1 = Selector(
            l_pool2,
            numpy.array([0, 0.5])
        )

        l_conv3 = lasagne.layers.Conv2DLayer(
            select1,
            num_filters=64,
            filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, ds=(2, 2))

        l_conv4 = lasagne.layers.Conv2DLayer(
            l_pool3,
            num_filters=128,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool4 = lasagne.layers.MaxPool2DLayer(l_conv4, ds=(2, 2))

        l_conv5 = lasagne.layers.Conv2DLayer(
            l_pool4,
            num_filters=256,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool5 = lasagne.layers.MaxPool2DLayer(l_conv5, ds=(2, 2))

        l_conv6 = lasagne.layers.Conv2DLayer(
            l_pool5,
            num_filters=512,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool6 = lasagne.layers.MaxPool2DLayer(l_conv6, ds=(2, 2))

        l_hidden1 = lasagne.layers.DenseLayer(
            l_pool6,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )

        l_hidden1_dropout = lasagne.layers.DropoutLayer(
            l_hidden1,
            p=0.5,
            )


        ###
        select2 = Selector(
            l_pool2,
            numpy.array([0.5, 1])
        )

        l_conv3bis = lasagne.layers.Conv2DLayer(
            select2,
            num_filters=64,
            filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool3bis = lasagne.layers.MaxPool2DLayer(l_conv3bis, ds=(2, 2))

        l_conv4bis = lasagne.layers.Conv2DLayer(
            l_pool3bis,
            num_filters=128,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool4bis = lasagne.layers.MaxPool2DLayer(l_conv4bis, ds=(2, 2))

        l_conv5bis = lasagne.layers.Conv2DLayer(
            l_pool4bis,
            num_filters=256,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool5bis = lasagne.layers.MaxPool2DLayer(l_conv5bis, ds=(2, 2))

        l_conv6bis = lasagne.layers.Conv2DLayer(
            l_pool5bis,
            num_filters=512,
            filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            untie_biases=True,
            )
        l_pool6bis = lasagne.layers.MaxPool2DLayer(l_conv6bis, ds=(2, 2))

        l_hidden1bis = lasagne.layers.DenseLayer(
            l_pool6bis,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )

        l_hidden1_dropoutbis = lasagne.layers.DropoutLayer(
            l_hidden1bis,
            p=0.5,
            )




        ###
        concat_out = lasagne.layers.concat([l_hidden1_dropout, l_hidden1_dropoutbis], axis=1)

        l_hidden2 = lasagne.layers.DenseLayer(
            concat_out,
            num_units=256,
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
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )
