import cPickle as pickle
import gzip
import itertools
import urllib
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import Lasagne.lasagne as lasagne
import theano
import theano.tensor as T
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop

from model1 import build_model

NUM_EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
MOMENTUM = 0.9


def create_iter_functions(output_layer,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):

    print("Creating training functions...")

    X_batch = T.tensor4('x')
    y_batch = T.matrix('y', dtype='int64')

    objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    # aa = output_layer.get_output(X_batch, deterministic=True)
    # f = theano.function([X_batch], aa)
    # print f(np.random.normal(0, 1, (BATCH_SIZE, 3, 28, 28)))

    pred = T.cast(T.switch(
        output_layer.get_output(X_batch, deterministic=True) < 0.5, 0, 1), 'int64')
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.sgd(loss_train, all_params, learning_rate)
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
        )


def train(iter_funcs, dataset_train, dataset_valid, dataset_test, batch_size=BATCH_SIZE):

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in itertools.count(1):

        start = time.time()
        train_iterator = get_data_iterator(dataset_train, batch_size)
        valid_iterator = get_data_iterator(dataset_valid, batch_size)

        batch_train_losses = []
        batch_train_accuracies = []

        for b in range(train_iterator.num_batches):
            print "\r{}/{}".format(b,train_iterator.num_batches),
            sys.stdout.flush()
            X_batch, y_batch = next(train_iterator)
            X_batch = np.swapaxes(X_batch, 3, 2)
            X_batch = np.swapaxes(X_batch, 2, 1)
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
        for b in range(valid_iterator.num_batches):
            X_batch, y_batch = next(valid_iterator)
            X_batch = np.swapaxes(X_batch, 3, 2)
            X_batch = np.swapaxes(X_batch, 2, 1)
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](X_batch, y_batch)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_accuracy)

        np.savez("records.npz", train_losses=train_losses, valid_losses=valid_losses,
                 train_accuracies=train_accuracies,
                 valid_accuracies=valid_accuracies)

        end = time.time()

        yield {
            'number': epoch,
            'elapsed_time': end - start,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            }

def get_data_iterator(dataset, batch_size=BATCH_SIZE):
    iterator = dataset.iterator(
        mode='batchwise_shuffled_sequential',
        batch_size=batch_size)
    return iterator


def main(num_epochs=NUM_EPOCHS):

    image_width = 221
    rescale = 256
    transformer = RandomCrop(rescale, image_width)

    dataset_train = DogsVsCats(transformer, 0, 20000)
    dataset_valid = DogsVsCats(transformer, 20000, 22500)
    dataset_test = DogsVsCats(transformer, 22500, 25000)

    output_layer = build_model(
        input_height=image_width,
        input_width=image_width,
        output_dim=1,
        batch_size=BATCH_SIZE
        )

    iter_funcs = create_iter_functions(output_layer)

    print("Starting training...")
    for epoch in train(iter_funcs, dataset_train, dataset_valid, dataset_test):
        print("Epoch %d of %d" % (epoch['number'], num_epochs))
        print("  elapsed time:\t\t%.6f" % epoch['elapsed_time'])
        print("  training loss:\t\t%.6f" % epoch['train_loss'])
        print("  validation loss:\t\t%.6f" % epoch['valid_loss'])
        print("  validation accuracy:\t\t%.2f %%" %
              (epoch['valid_accuracy'] * 100))

        if epoch['number'] >= num_epochs:
            break

    return output_layer

if __name__ == '__main__':
    records = np.load("records.npz")


    valid_losses = records['valid_losses']

    plt.xlabel("Number of epochs")
    plt.ylabel("acc")
    plt.title("go")

    plt.plot(valid_losses, linestyle="dashed", marker="o", color="green")
    plt.grid()
    plt.show()

    # main()
