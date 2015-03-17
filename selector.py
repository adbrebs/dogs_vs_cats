import lasagne


class Selector(lasagne.layers.Layer):
    """
    A fully connected layer.

    :parameters:
        - input_layer : `Layer` instance
            The layer from which this layer will obtain its input

        - num_units : int
            The number of units of the layer

        - W : Theano shared variable, numpy array or callable
            An initializer for the weights of the layer. If a Theano shared
            variable is provided, it is used unchanged. If a numpy array is
            provided, a shared variable is created and initialized with the
            array. If a callable is provided, a shared variable is created and
            the callable is called with the desired shape to generate suitable
            initial values. The variable is then initialized with those values.

        - b : Theano shared variable, numpy array, callable or None
            An initializer for the biases of the layer. If a Theano shared
            variable is provided, it is used unchanged. If a numpy array is
            provided, a shared variable is created and initialized with the
            array. If a callable is provided, a shared variable is created and
            the callable is called with the desired shape to generate suitable
            initial values. The variable is then initialized with those values.

            If None is provided, the layer will have no biases.

        - nonlinearity : callable or None
            The nonlinearity that is applied to the layer activations. If None
            is provided, the layer will be linear.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
    """
    def __init__(self, incoming, prop_indices, **kwargs):
        super(Selector, self).__init__(incoming, **kwargs)

        numpy_arr = prop_indices * self.input_shape[0]
        self.slice = slice(* numpy_arr.astype(int))

    def get_output_shape_for(self, input_shape):
        return (self.slice.stop-self.slice.start,) + input_shape[1:]

    def get_output_for(self, input, *args, **kwargs):
        return input[self.slice]