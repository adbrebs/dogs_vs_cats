import cPickle as pickle
import itertools
import sys
import time
import os
import argparse
import numpy as np

import theano
import theano.tensor as T
theano.config.floatX = "float32"
import lasagne

from dataset import *
from models.mod_9 import Network_9
from updates import momentum_bis


def create_iter_functions(model, learning_rate, momentum):

    print("Creating training functions...")

    y_batch = T.ivector('y')
    l_out = model.l_out

    objective = lasagne.objectives.Objective(l_out, loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(target=y_batch)
    loss_eval = objective.get_loss(target=y_batch, deterministic=True)

    # pred = T.cast(T.switch(
    #     l_out.get_output(deterministic=True) < 0.5, 0, 1), 'int32')
    pred = T.argmax(
        l_out.get_output(deterministic=True), axis=1)
    error_rate = T.mean(T.neq(pred, y_batch))

    Xs_batch = [l_in.input_var for l_in in model.l_ins]
    f_pred = theano.function(Xs_batch, [l_out.get_output(deterministic=True)])

    all_params = lasagne.layers.get_all_params(l_out)
    updates, extra_params = momentum_bis(loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        Xs_batch + [y_batch], [loss_train, error_rate],
        updates=updates
    )

    iter_valid = theano.function(
        Xs_batch + [y_batch], [loss_eval, error_rate]
    )

    iter_test = theano.function(
        Xs_batch + [y_batch], [loss_eval, error_rate]
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        f_pred=f_pred,
        params_learning_rule=extra_params
    )


def iter_over_epoch(iterator, iter_fun, losses, error_rates):
    batch_losses = []
    batch_errors = []
    for b, ([(batch, _)], qsize) in enumerate(iterator.parallel_generation()):
        print "\r{}/{}, qsize {}".format(b, iterator.n_batches, qsize),
        sys.stdout.flush()
        batch_loss, batch_err = iter_fun(*batch)
        batch_losses.append(batch_loss)
        batch_errors.append(batch_err)
    print ""
    avg_loss = np.mean(batch_losses)
    avg_err = np.mean(batch_errors)
    losses.append(avg_loss)
    error_rates.append(avg_err)
    return avg_loss, avg_err


def train(iter_funcs, model,
          dataset_train, dataset_valid,
          batch_size, learning_rate, momentum, patience_max,
          data_aug_train, data_aug_valid, n_procs,
          num_training_batches_per_epoch=None,
          num_validation_batches_per_epoch=None,
          write_path="./", name_experiment="mod", save_freq=10):

    name_best_model = "mod_best.pkl"

    # Initialize containers
    channels_train = {"loss": [], "err": []}
    channels_valid = {"loss": [], "err": []}
    channels = {"training": channels_train, "valid": channels_valid}

    lowest_valid_error = np.inf

    patience = 0
    epochs_lr_change = [(0, learning_rate.get_value())]

    for epoch in itertools.count(1):

        start = time.time()
        train_iterator = dataset_train.get_iterator(data_aug_train, batch_size, n_procs,
                                                    num_training_batches_per_epoch)
        valid_iterator = dataset_valid.get_iterator(data_aug_valid, batch_size, n_procs,
                                                    num_validation_batches_per_epoch)

        train_avg_loss, train_avg_err = iter_over_epoch(train_iterator, iter_funcs["train"],
                                                        channels_train["loss"], channels_train["err"])
        valid_avg_loss, valid_avg_err = iter_over_epoch(valid_iterator, iter_funcs["valid"],
                                                        channels_valid["loss"], channels_valid["err"])

        pickle.dump(channels, open(write_path + "channels.pkl", "wb"))

        end = time.time()

        if epoch >= save_freq and epoch%save_freq==0:
            print("Saving model..." + name_experiment)
            pickle.dump(model, open(write_path + "mod.pkl", "wb"))

        if valid_avg_err < lowest_valid_error:
            patience = 0
            lowest_valid_error = valid_avg_err
            print("Saving best model..." + name_experiment)
            pickle.dump(model, open(write_path + name_best_model, "wb"))
        else:
            if patience == patience_max:
                patience = 0
                new_lr = numpy.float32(learning_rate.get_value()/2.0)
                print "new learning rate: " + str(new_lr)
                learning_rate.set_value(new_lr)
                epochs_lr_change.append((epoch, new_lr))
                pickle.dump(epochs_lr_change, open(write_path + "lr.pkl", "wb"))
                # We continue from our best model so far
                model2 = pickle.load(open(write_path + name_best_model, "rb"))
                model.copy_from_other_model(model2)
                del model2
                # Set the variables of the momentum to zero!
                for s_param in iter_funcs["params_learning_rule"]:
                    s_param.set_value(np.zeros_like(s_param.get_value()))
            else:
                patience += 1
        print "best: " + str(lowest_valid_error)
        yield {
            'number': epoch,
            'elapsed_time': end - start,
            'train_loss': train_avg_loss,
            'train_error': train_avg_err,
            'valid_loss': valid_avg_loss,
            'valid_error': valid_avg_err,
            }

import socket
host = socket.gethostname()


def main(batch_size=50):
    # # For pretraining
    # model_to_be_loaded = pickle.load(open("/Tmp/debrea/cat/mod_9_1view/" +
    #                                  "mod_best.pkl", "rb"))

    # Setup
    n_branches = 5
    name_experiment="mod_9_tata"
    write_path = "/Tmp/debrea/cat/" + name_experiment + "/"
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    if host == "adeb.laptop":
        n_procs = 1
    else:
        n_procs = multiprocessing.cpu_count()-4

    num_epochs = 500
    initial_lr = 1e-2
    momentum = 0.9
    patience_max = 10

    # Data_agumentation
    image_width = 70
    min_side = 70

    scale_min = ScaleMinSide(min_side)
    crop = RandomCropping(image_width)
    h_flip = HorizontalFlip()
    rand_h_flip = RandomizeSingle(h_flip, 0.5, child=crop)
    rot_30 = RandomRotate(30, child=rand_h_flip)
    d5 = ParallelDataAugmentator([rot_30]*n_branches, child=scale_min)
    d_train = d5
    d_valid = d5
    pickle.dump((d_train, d_valid), open(write_path + "data_agumentation.pkl", "wb"))

    # Creation of the datasets
    if host == "adeb.laptop":
        dataset_train = Dataset(start=0, stop=2000)
        dataset_valid = Dataset(start=2000, stop=2500)
    else:
        dataset_train = Dataset(start=0, stop=20000)
        dataset_valid = Dataset(start=20000, stop=25000)

    # Creation of the model
    model = Network_9(
        n_chunks=n_branches,
        input_height=image_width,
        input_width=image_width,
        output_dim=2,
        batch_size=batch_size
    )

    # # for pre-training
    # model.copy_from_other_model(model_to_be_loaded)

    # Creation of the iterative functions
    learning_rate = theano.shared(numpy.float32(initial_lr), "learning_rate")
    momentum = theano.shared(numpy.float32(momentum), "momentum")
    iter_funcs = create_iter_functions(model, learning_rate, momentum)

    epochs = train(iter_funcs, model, dataset_train,
                   dataset_valid, batch_size, learning_rate, momentum, patience_max,
                   d_train, d_valid, n_procs,
                   num_training_batches_per_epoch=None,
                   num_validation_batches_per_epoch=None,
                   write_path=write_path, name_experiment=name_experiment,
                   save_freq=10)

    print("Starting training...")
    for epoch in epochs:
        print("Epoch {} of {}".format(epoch['number'], num_epochs))
        print("  elapsed time:\t\t\t{}".format(epoch['elapsed_time']))
        print("  training loss:\t\t{}".format(epoch['train_loss']))
        print("  training error:\t\t{} %%".format(epoch['train_error'] * 100))
        print("  validation loss:\t\t{}".format(epoch['valid_loss']))
        print("  validation error:\t\t{} %%".format(epoch['valid_error'] * 100))

        if epoch['number'] >= num_epochs:
            break


if __name__ == '__main__':
    main()