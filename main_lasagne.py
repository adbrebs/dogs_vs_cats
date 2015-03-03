import cPickle as pickle
import itertools
import sys
import time
import os
import numpy as np

import theano
import theano.tensor as T
theano.config.floatX = "float32"
import Lasagne.lasagne as lasagne

from dataset import *
from models.lasagne.mod4 import build_model


def create_iter_functions(output_layer, learning_rate, momentum):

    print("Creating training functions...")

    X_batch = T.tensor4('x')
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
    # updates = lasagne.updates.adadelta(loss_train, all_params, learning_rate)
    # updates = lasagne.updates.sgd(loss_train, all_params, learning_rate)

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


def iter_over_epoch(iterator, iter_fun):
    batch_losses = []
    batch_accuracies = []
    for b, (X_batch, y_batch) in enumerate(iterator):
        if len(X_batch) == 1:
            X_batch = X_batch[0]
        print "\r{}/{}".format(b, iterator.n_batches),
        sys.stdout.flush()
        batch_loss, batch_accuracy = iter_fun(X_batch, y_batch)
        batch_losses.append(batch_loss)
        batch_accuracies.append(batch_accuracy)
    print ""
    avg_loss = np.mean(batch_losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy


def train(iter_funcs, output_layer,
          dataset_train, dataset_valid,
          batch_size, learning_rate, num_training_batches_per_epoch=None, num_validation_batches_per_epoch=None,
          write_path="./", name_experiment="mod", save_freq=10):

    # Initialize containers
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    best_valid_accuracy = np.inf


    patience = 0
    patience_max = 5
    epochs_lr_change = [(0, learning_rate.get_value())]

    for epoch in itertools.count(1):

        start = time.time()
        train_iterator = dataset_train.get_iterator(batch_size, num_training_batches_per_epoch)
        valid_iterator = dataset_valid.get_iterator(batch_size, num_validation_batches_per_epoch)

        avg_train_loss, avg_train_accuracy = iter_over_epoch(train_iterator, iter_funcs["train"])
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        avg_valid_loss, avg_valid_accuracy = iter_over_epoch(valid_iterator, iter_funcs["valid"])
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_accuracy)

        channels_train = {"loss": train_losses, "accuracy": train_accuracies}
        channels_valid = {"loss": valid_losses, "accuracy": valid_accuracies}
        channels = {"traing": channels_train, "valid": channels_valid}
        pickle.dump(channels, open(write_path + "channels.pkl", "wb"))

        end = time.time()

        if epoch >= save_freq and epoch%save_freq==0:
            print("Saving model..." + name_experiment)
            pickle.dump(output_layer, open(write_path + "mod.pkl", "wb"))
            pickle.dump(epochs_lr_change, open(write_path + "lr.pkl", "wb"))

        if avg_valid_accuracy < best_valid_accuracy:
            patience = 0
            best_valid_accuracy = avg_valid_accuracy
            print("Saving best model..." + name_experiment)
            pickle.dump(output_layer, open(write_path + "mod_best.pkl", "wb"))
        else:
            if patience == patience_max:
                patience = 0
                new_lr = numpy.float32(learning_rate.get_value()/2.0)
                print "new learning rate: " + str(new_lr)
                learning_rate.set_value(new_lr)
                epochs_lr_change.append((epoch, new_lr))
            else:
                patience += 1

        yield {
            'number': epoch,
            'elapsed_time': end - start,
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            }


if __name__ == '__main__':

    # Setup
    name_experiment="mod_4_dropout_da_lr_decrease"
    write_path = "/Tmp/debrea/cat/" + name_experiment + "/"
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    num_epochs = 1000
    batch_size = 50
    initial_lr = 1e-2
    momentum = 0.9
    image_width = 260
    min_side = 260

    # Data_agumentation
    d1 = ScaleMinSide(min_side)
    d2 = RandomCropping(image_width, child=d1)
    d3 = Randomize(HorizontalFlip(child=d2), 0.5)
    d4 = RandomRotate(30, child=d3)
    d_train = d4
    d_valid = d2
    pickle.dump((d_train, d_valid), open(write_path + "data_agumentation.pkl", "wb"))

    # Creation of the datasets
    dataset_train = Dataset(start=0, stop=20000, data_augmentator=d_train)
    dataset_valid = Dataset(start=20000, stop=22500, data_augmentator=d_valid)

    # Creation of the model
    output_layer = build_model(
        input_height=image_width,
        input_width=image_width,
        output_dim=2,
        batch_size=batch_size
    )

    # Creation of the iterative functions
    learning_rate = theano.shared(numpy.float32(initial_lr), "learning_rate")
    iter_funcs = create_iter_functions(output_layer, learning_rate, momentum)

    epochs = train(iter_funcs, output_layer, dataset_train,
                   dataset_valid, batch_size, learning_rate,
                   num_training_batches_per_epoch=None,
                   num_validation_batches_per_epoch=None,
                   write_path=write_path, name_experiment=name_experiment,
                   save_freq=10)

    print("Starting training...")
    for epoch in epochs:
        print("Epoch {} of {}".format(epoch['number'], num_epochs))
        print("  elapsed time:\t\t\t{}".format(epoch['elapsed_time']))
        print("  training loss:\t\t{}".format(epoch['train_loss']))
        print("  training accuracy:\t\t{} %%".format(epoch['train_accuracy'] * 100))
        print("  validation loss:\t\t{}".format(epoch['valid_loss']))
        print("  validation accuracy:\t\t{} %%".format(epoch['valid_accuracy'] * 100))

        if epoch['number'] >= num_epochs:
            break
