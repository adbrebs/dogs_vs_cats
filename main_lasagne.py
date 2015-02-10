import cPickle as pickle
import gzip
import itertools
import urllib
import os
import sys
import time
import argparse

import numpy as np

import Lasagne.lasagne as lasagne
import theano
import theano.tensor as T
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop

from model3 import build_model

NUM_EPOCHS = 1000
BATCH_SIZE = 50
LEARNING_RATE = 1e-3
MOMENTUM = 0.1


def create_iter_functions(output_layer,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):

    print("Creating training functions...")

    X_batch = T.tensor4('x')
    # y_batch = T.matrix('y', dtype='int32')
    y_batch = T.ivector('y')

    objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.multinomial_nll)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    # pred = T.cast(T.switch(
    #     output_layer.get_output(X_batch, deterministic=True) < 0.5, 0, 1), 'int32')
    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.neq(pred, y_batch))

    f_pred = theano.function([X_batch], [output_layer.get_output(X_batch, deterministic=True)])

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.momentum(loss_train, all_params, learning_rate, momentum)
    # updates = lasagne.updates.sgd(loss_train, all_params, learning_rate)
    # updates = lasagne.updates.nesterov_momentum(
    #     loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        [X_batch, y_batch], [loss_train, accuracy],
        updates=updates
        )

    iter_valid = theano.function(
        [X_batch, y_batch], [loss_eval, accuracy]
        )

    iter_test = theano.function(
        [X_batch, y_batch], [loss_eval, accuracy]
        )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        f_pred=f_pred
        )


def train(iter_funcs,
          dataset_train, dataset_valid, dataset_test,
          batch_size, num_training_batches_per_epoch=None, num_validation_batches_per_epoch=None,
          write_path="./", name_experiment="mod"):

    # Initialize containers
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in itertools.count(1):

        start = time.time()
        train_iterator = get_data_iterator(dataset_train, batch_size)
        valid_iterator = get_data_iterator(dataset_valid, batch_size)

        if num_training_batches_per_epoch is None:
            num_training_batches_per_epoch = train_iterator.num_batches
        if num_validation_batches_per_epoch is None:
            num_validation_batches_per_epoch = valid_iterator.num_batches

        batch_train_losses = []
        batch_train_accuracies = []

        for b in range(num_training_batches_per_epoch):
            print "\r{}/{}".format(b,num_training_batches_per_epoch),
            sys.stdout.flush()
            X_batch, y_batch = next(train_iterator)
            if X_batch.shape[0] != batch_size:
                X_batch, y_batch = next(train_iterator)
            X_batch = np.swapaxes(X_batch, 3, 2)
            X_batch = np.swapaxes(X_batch, 2, 1)
            y_batch = np.squeeze(y_batch)
            # a = iter_funcs['f_pred'](X_batch)
            y_batch = y_batch.astype('int32')
            batch_train_loss, batch_train_accuracy = iter_funcs['train'](X_batch, y_batch)
            batch_train_losses.append(batch_train_loss)
            batch_train_accuracies.append(batch_train_accuracy)
        print ""
        avg_train_loss = np.mean(batch_train_losses)
        avg_train_accuracy = np.mean(batch_train_accuracies)

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_validation_batches_per_epoch):
            X_batch, y_batch = next(valid_iterator)
            if X_batch.shape[0] != batch_size:
                X_batch, y_batch = next(valid_iterator)
            X_batch = np.swapaxes(X_batch, 3, 2)
            X_batch = np.swapaxes(X_batch, 2, 1)
            y_batch = np.squeeze(y_batch)
            y_batch = y_batch.astype('int32')
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](X_batch, y_batch)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_accuracy)

        losses=(("Train losses", train_losses), ("Validation losses", valid_losses))
        accuracies=(("Train accuracies", train_accuracies), ("Train losses", valid_accuracies))
        pickle.dump((losses, accuracies), open(write_path + name_experiment + "_lasagne.pkl", "wb"))

        end = time.time()

        yield {
            'number': epoch,
            'elapsed_time': end - start,
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            }

def get_data_iterator(dataset, batch_size=BATCH_SIZE):
    iterator = dataset.iterator(
        mode='batchwise_shuffled_sequential',
        batch_size=batch_size)
    return iterator


if __name__ == '__main__':
    theano.config.floatX="float32"

    write_path = "/Tmp/debrea/cat/"
    num_epochs = NUM_EPOCHS
    image_width = 260
    rescale = 280
    transformer1 = RandomCrop(rescale, image_width)
    transformer2 = RandomCrop(rescale, image_width)
    transformer3 = RandomCrop(rescale, image_width)

    dataset_train = DogsVsCats(transformer1, 0, 20000)
    dataset_valid = DogsVsCats(transformer2, 20000, 22500)
    dataset_test = DogsVsCats(transformer3, 22500, 25000)

    output_layer = build_model(
        input_height=image_width,
        input_width=image_width,
        output_dim=2,
        batch_size=BATCH_SIZE
        )

    iter_funcs = create_iter_functions(output_layer)

    print("Starting training...")
    for epoch in train(iter_funcs, dataset_train, dataset_valid, dataset_test, BATCH_SIZE,
                       num_training_batches_per_epoch=None, num_validation_batches_per_epoch=3,
                       write_path=write_path, name_experiment="mod3"):
        print("Epoch %d of %d" % (epoch['number'], num_epochs))
        print("  elapsed time:\t\t%.6f" % epoch['elapsed_time'])
        print("  training loss:\t\t%.6f" % epoch['train_loss'])
        print("  training accuracy:\t\t%.2f %%" %
              (epoch['train_accuracy'] * 100))
        print("  validation loss:\t\t%.6f" % epoch['valid_loss'])
        print("  validation accuracy:\t\t%.2f %%" %
              (epoch['valid_accuracy'] * 100))

        if epoch['number'] >= num_epochs:
            break


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("write_path", type=str)
#     args = parser.parse_args()
