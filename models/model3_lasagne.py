import Lasagne.lasagne as lasagne


def build_model(input_width, input_height, output_dim, batch_size):

    print("Building model...")

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
        )

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=16,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        )
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1,
        num_filters=32,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        )
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

    l_conv3 = lasagne.layers.Conv2DLayer(
        l_pool2,
        num_filters=48,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        )
    l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, ds=(2, 2))

    l_conv4 = lasagne.layers.Conv2DLayer(
        l_pool3,
        num_filters=64,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        )
    l_pool4 = lasagne.layers.MaxPool2DLayer(l_conv4, ds=(2, 2))

    l_conv5 = lasagne.layers.Conv2DLayer(
        l_pool4,
        num_filters=128,
        filter_size=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        )
    l_pool5 = lasagne.layers.MaxPool2DLayer(l_conv5, ds=(2, 2))

    l_conv6 = lasagne.layers.Conv2DLayer(
        l_pool5,
        num_filters=256,
        filter_size=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Uniform(),
        )
    l_pool6 = lasagne.layers.MaxPool2DLayer(l_conv6, ds=(2, 2))

    l_out = lasagne.layers.DenseLayer(
        l_pool6,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.Uniform(),
        )

    return l_out
