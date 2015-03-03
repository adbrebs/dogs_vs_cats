__author__ = 'adeb'

import cPickle as pickle
import itertools
import sys
import argparse
import os
import theano
import theano.tensor as T
import numpy

from dataset import *


def evaluate(dataset, batch_size, iter_function, name):

    iterator = dataset.get_iterator(batch_size, None)

    batch_accuracies = []

    for b, (X_batch, y_batch) in enumerate(iterator):
        if len(X_batch) == 1:
            X_batch = X_batch[0]
        print "\r{}/{}".format(b,iterator.n_batches),
        sys.stdout.flush()
        batch_accuracy = iter_function(X_batch, y_batch)
        batch_accuracies.append(batch_accuracy)
    print ""
    avg_accuracy = numpy.mean(batch_accuracies)

    print name + ": " + str(avg_accuracy)


def evaluate2(dataset, batch_size, pred_fun, name):

    iterator = dataset.get_iterator(batch_size, None)

    batch_accuracies = []

    for b, (X_batch, y_batch) in enumerate(iterator):
        print "\r{}/{}".format(b,iterator.n_batches),
        sys.stdout.flush()
        pred = numpy.zeros((batch_size, 2), dtype=theano.config.floatX)
        for bat in X_batch:
            pred += pred_fun(bat)
        pred = numpy.argmax(pred, axis=1)
        batch_accuracy = numpy.mean(pred != y_batch)
        batch_accuracies.append(batch_accuracy)
    print ""
    avg_accuracy = numpy.mean(batch_accuracies)

    print name + ": " + str(avg_accuracy)


def main(path):
    theano.config.floatX = "float32"

    batch_size = 50
    output_layer = pickle.load( open( path[0] + "mod.pkl", "rb" ) )
    d_train, d_test = pickle.load( open( path[0] + "data_agumentation.pkl", "rb" ) )

    dataset_train = Dataset(start=0, stop=20000, data_augmentator=d_test)
    dataset_valid = Dataset(start=20000, stop=22500, data_augmentator=d_test)
    dataset_test = Dataset(start=22500, stop=25000, data_augmentator=d_test)

    X_batch = T.tensor4('x')
    y_batch = T.ivector('y')

    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.neq(pred, y_batch))

    iter_fun = theano.function(
        [X_batch, y_batch], [accuracy]
        )

    evaluate(dataset_train, batch_size, iter_fun, "train")
    evaluate(dataset_valid, batch_size, iter_fun, "valid")
    evaluate(dataset_test, batch_size, iter_fun, "test")


def main2(path):
    theano.config.floatX = "float32"

    batch_size = 50
    output_layer = pickle.load( open( path[0] + "mod.pkl", "rb" ) )
    d_train, d_test = pickle.load( open( path[0] + "data_agumentation.pkl", "rb" ) )

    dat_aug = ParallelDataAugmentator([d_train]*100)

    dataset_test1 = Dataset(start=22500, stop=25000, data_augmentator=d_train)
    dataset_test2 = Dataset(start=22500, stop=25000, data_augmentator=dat_aug)
    # dataset_test = Dataset(start=22500, stop=25000, data_augmentator=d_test)

    X_batch = T.tensor4('x')
    y_batch = T.ivector('y')

    predd = output_layer.get_output(X_batch, deterministic=True)
    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.neq(pred, y_batch))

    iter_pred = theano.function([X_batch], predd)
    iter_fun = theano.function(
        [X_batch, y_batch], [accuracy]
        )

    # evaluate(dataset_train, batch_size, iter_fun, "train")
    # evaluate(dataset_test1, batch_size, iter_fun, "valid")
    evaluate2(dataset_test2, batch_size, iter_pred, "valid")
    # evaluate(dataset_test, batch_size, iter_fun, "test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs='+')
    options = parser.parse_args()
    main2(options.model_path)
    # main("mod_mod_data_augmentation_nothing.pkl")