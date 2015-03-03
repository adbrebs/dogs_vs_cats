import Lasagne.lasagne as lasagne


def build_model(input_width, input_height, output_dim, batch_size):

    print("Building model...")

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
        )

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=32,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(0.1),
        )
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1,
        num_filters=32,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(0.1),
        )
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool2,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(0.1),
        )

    # l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    # l_hidden2 = lasagne.layers.DenseLayer(
    #     l_hidden1_dropout,
    #     num_units=256,
    #     nonlinearity=lasagne.nonlinearities.rectify,
    #     )
    # l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.Uniform(),
        )

    return l_out
