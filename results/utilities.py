import numpy as np
import matplotlib.pyplot as plt
import cPickle


def plot_statistics(statistics, legends, title="", ylabel="", xlim=None, ylim=None, writeto="default.jpeg"):
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel("Number of epochs")
    plt.ylabel(ylabel)
    plt.title(title)

    for stat in statistics:
        plt.plot(stat, linestyle="solid", marker=".")
    plt.grid()
    plt.legend(legends, loc='upper left')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.savefig("./results/" + writeto)


def analyse_pylearn2_channels(path, xlim=None, ylim_nll=None, ylim_err=None):
    channels = cPickle.load(open(path, "rb" ))

    train_misclass = channels['train_y_misclass'].val_record
    val_misclass = channels['valid_y_misclass'].val_record
    train_nll = channels['train_y_nll'].val_record
    val_nll = channels['valid_y_nll'].val_record
    legends = ['training', 'validation']

    plot_statistics([train_misclass, val_misclass], legends,
                    ylabel="Error rate", xlim=xlim, ylim=ylim_err, writeto="error_rate_pylearn2.jpeg")
    plot_statistics([train_nll, val_nll], legends,
                    ylabel="Negative Log Likelihood", xlim=xlim, ylim=ylim_nll, writeto="nll_pylearn2.jpeg")


def analyse_lasagne_records(path, xlim=None, ylim_nll=None, ylim_err=None):
    losses, accuracies = cPickle.load(open(path, "rb" ))
    train_losses, valid_losses = losses
    train_accuracies, valid_accuracies = accuracies
    train_accuracies = [1-a for a in train_accuracies[1]]
    valid_accuracies = [1-a for a in valid_accuracies[1]]

    legends = ['training', 'validation']
    plot_statistics([train_accuracies, valid_accuracies], legends,
                    ylabel="Error rate", xlim=xlim, ylim=ylim_err, writeto="error_rate_lasagne.jpeg")
    plot_statistics([train_losses[1], valid_losses[1]], legends,
                    ylabel="Negative Log Likelihood", xlim=xlim, ylim=ylim_nll, writeto="nll_lasagne.jpeg")


if __name__ == '__main__':
    xlim = (0, 170)
    ylim_nll = (0.1, 0.75)
    ylim_err = (0, 0.5)
    analyse_pylearn2_channels("./results/mod4_pylearn_channels.pkl", xlim=xlim, ylim_nll=ylim_nll, ylim_err=ylim_err)
    # analyse_lasagne_records("./results/mod1_lasagne.pkl", xlim=xlim, ylim_nll=ylim_nll, ylim_err=ylim_err)