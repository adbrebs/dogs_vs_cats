import pickle
import numpy
from theano import tensor as T

import lasagne
from custom_layers import *


class Network:
    def __init__(self, n_chunks, input_width, input_height, output_dim, batch_size):

        self.name = self.__class__.__name__
        self.n_branches = n_chunks
        self.input_width = input_width
        self.input_height = input_height
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.l_out = None
        self.l_ins = None
        self.concat_in = None

        self.create_model()

    def create_model(self):

        print("Building model...")

        self.l_ins = []
        for i in range(self.n_branches):
            l_in = lasagne.layers.InputLayer(
                shape=(self.batch_size, 3, self.input_width, self.input_height),
                )
            self.l_ins.append(l_in)

        self.concat_in = lasagne.layers.concat(self.l_ins, axis=0)

        self.create_model_virtual()

        print "Number of parameters: " + str(lasagne.layers.count_params(self.l_out))

        return self.l_out

    def create_model_virtual(self):
        raise NotImplementedError

    def combine_features(self, layer):
        """
        quite slow, use pool_over_branches instead
        """
        ls_selectors = []

        step = 1 / float(self.n_branches)
        for i in range(self.n_branches):
            a = i / float(self.n_branches)
            select = Selector(
                layer,
                numpy.array([a, a+step])
            )
            ls_selectors.append(select)

        return lasagne.layers.ElemwiseSumLayer(ls_selectors, coeffs=step)

    def pool_over_branches(self, layer, pool_function=T.max):
        """
        pools feature by feature over branches
        """
        if self.n_branches == 1:
            return layer

        reshaped_layer = lasagne.layers.ReshapeLayer(layer, (self.n_branches, self.batch_size, -1))
        reshaped_layer = lasagne.layers.DimshuffleLayer(reshaped_layer, (1, 0, 2))
        reshaped_layer = lasagne.layers.ReshapeLayer(reshaped_layer, (self.batch_size, -1))

        return lasagne.layers.FeaturePoolLayer(reshaped_layer, self.n_branches, 1, pool_function=pool_function)

    def aggregate_over_branches(self, layer):
        if self.n_branches == 1:
            return layer

        reshaped_layer = lasagne.layers.ReshapeLayer(layer, (self.n_branches, self.batch_size, -1))
        reshaped_layer = lasagne.layers.DimshuffleLayer(reshaped_layer, (1, 0, 2))

        conv = lasagne.layers.Conv1DLayer(
            reshaped_layer,
            num_filters=1,
            filter_length=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Constant(1.0/self.n_branches),
            b=lasagne.init.Constant(0),
            untie_biases=True
            )

        return conv

    def aggregate_over_branches2(self, layer):
        if self.n_branches == 1:
            return layer

        reshaped_layer = lasagne.layers.ReshapeLayer(layer, (self.n_branches, self.batch_size, -1))
        reshaped_layer = lasagne.layers.DimshuffleLayer(reshaped_layer, (1, 0, 2))
        reshaped_layer = lasagne.layers.ReshapeLayer(reshaped_layer, (self.batch_size, -1))

        dense = lasagne.layers.DenseLayer(
            reshaped_layer,
            num_units=reshaped_layer.get_output_shape()[1]/self.n_branches,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(0.1),
            b=lasagne.init.Constant(0),
            )

        return dense

    def save(self, path):
        all_params = lasagne.layers.get_all_params(self.l_out)
        all_param_values = [p.get_value() for p in all_params]
        pickle.dump(all_param_values, open(path, "wb"))

    def load(self, path):
        all_param_values = pickle.load( open( path, "rb" ) )
        all_params = lasagne.layers.get_all_params(self.l_out)
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)

    def copy_from_other_model(self, model2):
        all_params = lasagne.layers.get_all_params(self.l_out)
        all_params2 = lasagne.layers.get_all_params(model2.l_out)
        for p, v in zip(all_params, all_params2):
            p.set_value(v.get_value())