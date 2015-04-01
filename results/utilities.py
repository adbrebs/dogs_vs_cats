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

    plt.savefig("./" + writeto)


def extract_records(path):
    channels = cPickle.load(open(path, "rb"))
    return channels


def compare_records(ls_files, writeto, xlim, ylim, dataset, measure):
    """
    ls_files is a list of (path, description)
    dataset can be a list, measure can't be a list
    """
    measure_lgd = {"loss": "Negative Log Likelihodd", "err": "Error rate"}

    writeto = writeto + "_" + measure + ".jpeg"
    if not isinstance(dataset, list):
        dataset = [dataset]
    records = []
    legends = []
    for (path, descript) in ls_files:
        for ds in dataset:
            channels = extract_records(path)[ds][measure]
            records.append(channels)
            legends.append(descript + " (" + measure + "_" + ds + ")")

    plot_statistics(records, legends=legends,
                    ylabel=measure_lgd[measure], xlim=xlim, ylim=ylim, writeto=writeto)


if __name__ == '__main__':

    name = "multi_view/comp_pretrain_valid"
    # ls_files = [
    #             # ("./results/lasagne/mod_7_1/", ""),
    #             # ("./results/lasagne/mod_7_smaller1/", "smaller"),
    #             # ("./results/lasagne/mod_7_bigger1/", "bigger"),
    #             ("./results/lasagne/mod_7_smaller21/", "smaller with 3x3"),
    #             # ("./results/lasagne/mod_7_smaller31/", "3x3 and less neurons"),
    #             ("./results/lasagne/mod_7_smaller2_nomaxpool1/", "no maxpool at the end"),
    #             ("./results/lasagne/mod_7_smaller2_nomaxpool_3every1/", "only 3x3"),
    #             ("./results/lasagne/mod_7_top1/", "only 3x3 top")]

    ls_files = [
        ("./multi_view/c_1view.pkl", "1 view"),
                ("./multi_view/c_5views_mean.pkl", "5 views mean"),
                # ("./multi_view/c_5views_dropout_branches.pkl", "5 views mean "
                #                                               "dropout "
                #                                               "branches"),
                # ("./multi_view/c_5views_max.pkl", "5 views max"),
                # ("./multi_view/c_5views_l2.pkl", "5 views l2"),
                ("./multi_view/c_5views_pretrained.pkl", "5 views mean "
                                                          "pretrained")
                ]
    compare_records(ls_files, name, xlim=(0,200),
                            ylim=(0.06,0.15),
                            dataset=["valid"],
                            measure="err",)
