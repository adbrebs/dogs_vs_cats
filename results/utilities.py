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
    plt.legend(legends, loc='upper right')

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


def analyse_lasagne_records(path, write_to, measure="loss", xlim=None, ylim=None):
    train_records = extract_lasagne_records(path)["traing"][measure]
    valid_records = extract_lasagne_records(path)["valid"][measure]

    legends = ['training', 'validation']
    plot_statistics([train_records, valid_records], legends,
                    ylabel="Error rate", xlim=xlim, ylim=ylim, writeto=write_to)


def extract_lasagne_records(path):
    channels = cPickle.load(open(path, "rb"))
    return channels


def compare_lasagne_records(ls_paths, writeto, xlim, ylim, dataset="traing", measure="loss"):
    records = []
    legends = []
    for (path, descript) in ls_paths:
        if not isinstance(dataset, list):
            channels = extract_lasagne_records(path)[dataset][measure]
            records.append(channels)
            legends.append(descript + "_" + measure + "_" + dataset)
        else:
            for ds in dataset:
                channels = extract_lasagne_records(path)[ds][measure]
                records.append(channels)
                legends.append(descript + "_" + measure + "_" + ds)

    plot_statistics(records, legends=legends,
                    ylabel=measure, xlim=xlim, ylim=ylim, writeto=writeto)


if __name__ == '__main__':

    # paths = [("./results/lasagne/mod_data_augmentation_nothing/channels.pkl", "nothing"),
    #           ("./results/lasagne/mod_data_augmentation_flip/channels.pkl", "flip 0.5"),
    #           ("./results/lasagne/mod_data_augmentation_rotation_30/channels.pkl", "rotation 30")]
    # compare_lasagne_records(paths, "acc_data_augmentation.jpeg", xlim=(0,400), ylim=(0,0.5), dataset=["traing",
    #                                                                                                    "valid"],
    #                         measure="accuracy",)

    # paths = [("./results/lasagne/mod_4_no_dropout/channels.pkl", "no dropout, mom 0.9"),
    #           ("./results/lasagne/mod_4_no_dropout_low_momentum/channels.pkl", "no dropout, mom 0.1"),
    #           ("./results/lasagne/mod_4_dropout/channels.pkl", "dropout, mom 0.9")]
    # compare_lasagne_records(paths, "loss.jpeg", xlim=(0,100), ylim=(0,0.7), dataset=["traing", "valid"],
    #                         measure="loss",)
    # analyse_pylearn2_channels("./results/channels_mod_gui_pylearn.pkl", xlim=xlim, ylim_nll=ylim_nll, ylim_err=ylim_err)

    # analyse_lasagne_records("./results/lasagne/mod_data_augmentation_nothing/channels.pkl",
    #                         write_to="eee.jpeg", measure="accuracy", xlim=None, ylim=None)

    # paths = [("./results/lasagne/mod_4_dropout/channels.pkl", "no data augmentation"),
    #          ("./results/lasagne/mod_4_dropout_data_augmentation/channels.pkl", "data augmentation")]
    # compare_lasagne_records(paths, "loss_comparison_data_augmentation.jpeg", xlim=(0,150), ylim=(0,0.7),
    #                         dataset=["traing", "valid"],
    #                         measure="loss",)

    paths = [("./results/lasagne/mod_4_dropout/channels.pkl", "fixed lr 1e-3"),
             ("./results/lasagne/mod_4_dropout_lr_decrease/channels.pkl",
              "decreasing lr")]
    compare_lasagne_records(paths, "acc_comparison_lr.jpeg", xlim=(0,80),
                            ylim=(0,0.4),
                            dataset=["traing", "valid"],
                            measure="accuracy",)