import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as \
    RandomStreams
_srng = RandomStreams()
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=4)

import lasagne


class Selector(lasagne.layers.Layer):
    """
    Extracts a portion of the batch.
    """
    def __init__(self, incoming, prop_indices, **kwargs):
        super(Selector, self).__init__(incoming, **kwargs)

        numpy_arr = prop_indices * self.input_shape[0]
        self.slice = slice(* numpy_arr.astype(int))

    def get_output_shape_for(self, input_shape):
        return (self.slice.stop-self.slice.start,) + input_shape[1:]

    def get_output_for(self, input, *args, **kwargs):
        return input[self.slice]


class DropoutBranchLayer(lasagne.layers.Layer):
    """
    Dropout contiguous portions of the features.
    """
    def __init__(self, incoming, n_branches, n_branches_to_cut, batch_size,
                 rescale=True, **kwargs):
        super(DropoutBranchLayer, self).__init__(incoming, **kwargs)
        self.n_branches_to_cut = n_branches_to_cut
        self.rescale = rescale
        self.scale_factor = float(n_branches) / (n_branches - self.n_branches_to_cut)
        self.n_branches = n_branches
        branches = numpy.ones((n_branches,), dtype="int64")
        branches[:self.n_branches_to_cut] = 0
        self.branches = theano.shared(branches)
        self.batch_size = batch_size

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        if deterministic or self.n_branches_to_cut == 0:
            return input
        else:
            if self.rescale:
                input *= self.scale_factor

            branch_perm = srng.permutation(n=self.n_branches)
            branches = self.branches[branch_perm]
            features = theano.tensor.extra_ops.repeat(branches,
                                                      repeats=self.batch_size)

            features = features.dimshuffle(0, 'x', 'x', 'x')

            return input * features