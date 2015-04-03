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
from models.mod7_smaller2_nomaxpool_3every import Model


# def evaluate_special(dataset, data_augmentator, batch_size, pred_fun, n_avg, name):
#
#     # n_procs = multiprocessing.cpu_count()-3
#     n_procs = 2
#
#     iterator = dataset.get_iterator(data_augmentator, batch_size, n_procs, None)
#
#     predictions = numpy.zeros((dataset.n_data, 2), dtype=int)
#     predictions[:, 0] = 1 + numpy.arange(dataset.n_data)
#     for b, (batches, qsize) in enumerate(iterator.parallel_generation(n_times=n_avg, num_cached=2)):
#         print "\r{}/{}, qsize {}".format(b, iterator.n_batches, qsize),
#         sys.stdout.flush()
#
#         pred = numpy.zeros((batch_size, 2), dtype=theano.config.floatX)
#
#         for batch, idx in batches:
#             X_batch = batch[:-1]
#             pred += pred_fun(*X_batch)
#         predictions[idx, 1] = numpy.argmax(pred, axis=1)
#     print ""
#
#     numpy.savetxt("foo.csv", predictions, delimiter=",")


def evaluate(dataset, data_augmentator, batch_size, pred_fun, n_avg, name):

    n_procs = multiprocessing.cpu_count()-3
    # n_procs = 1

    iterator = dataset.get_iterator(data_augmentator, batch_size, n_procs, None)
    n_batches = iterator.n_batches

    batch_accuracies = []
    misclassified_examples = []

    for b, (batches, qsize) in enumerate(iterator.generate_batches_in_parallel(n_times=n_avg, num_cached=3)):
        print "\r{}/{}, qsize {}".format(b, iterator.n_batches, qsize),
        sys.stdout.flush()

        pred = 3 * numpy.ones((batch_size, 2), dtype=theano.config.floatX)

        for batch, idx in batches:
            y_batch = batch[-1]
            X_batch = batch[:-1]
            pred += pred_fun(*X_batch)
        pred = numpy.argmax(pred, axis=1)
        mis = pred != y_batch
        misclassified_examples += list(idx[mis])
        batch_accuracy = numpy.mean(mis)
        batch_accuracies.append(batch_accuracy)

    print ""
    avg_accuracy = numpy.mean(batch_accuracies)

    print name + ": " + str(avg_accuracy)
    print "misclassified: " + str(misclassified_examples)


def main(path):
    theano.config.floatX = "float32"

    n_avg = 20

    model = pickle.load(open(path[0] + "mod_best.pkl", "rb"))

    l_out = model.l_out
    d_train, d_valid = pickle.load(open(path[0] + "data_agumentation.pkl", "rb"))

    # dataset_test = Dataset(start=0, stop=12500, path="/data/lisa/exp/debrea/transfer/test.h5")

    dataset_train = Dataset(start=0, stop=20000)
    dataset_valid = Dataset(start=20000, stop=22500)
    dataset_test = Dataset(start=22500, stop=25000)

    pred = l_out.get_output(deterministic=True)

    Xs_batch = [l_in.input_var for l_in in model.l_ins]
    pred_fun = theano.function(
        Xs_batch, pred
        )
    # a = 0.1 *numpy.ones((50,3,260,260), dtype=theano.config.floatX)
    # print pred_fun(a)

    # evaluate_special(dataset_test, d_train, model.batch_size, pred_fun, n_avg, "tssss")
    evaluate(dataset_train, d_train, model.batch_size, pred_fun, n_avg, "train")
    evaluate(dataset_valid, d_valid, model.batch_size, pred_fun, n_avg, "valid")
    evaluate(dataset_test, d_valid, model.batch_size, pred_fun, n_avg, "test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs='+')
    options = parser.parse_args()
    main(options.model_path)
    # main("mod_mod_data_augmentation_nothing.pkl")