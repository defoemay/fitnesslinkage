import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

from preprocess_ds1 import preprocess_ds1
from preprocess_ds2 import preprocess_ds2
from knn_method import kNN_method_N
from svm_method import SVM_method_N
from kde_method import KDE_method_N
import sim_params as pm

if __name__ == "__main__":

    DATASET = pm.DATASET
    N = pm.N
    n_iters = pm.n_iters
    rng_seed = pm.rng_seed
    attributes = pm.attributes
    need_plot = True

    if DATASET in ["ds2", "DS2", "pmd", "PMD"]:
        preprocess = preprocess_ds2
    else:
        preprocess = preprocess_ds1

    varsplit = pm.varsplit

    p_range = np.arange(pm.p_min, pm.p_max+0.001, pm.p_step)
    accuracy_p = np.zeros((3, p_range.shape[0]))

    to_write = {"DATASET": DATASET, "seed": rng_seed, "N": N, "scope":varsplit, "q": pm.q,
            "kNN - n_neighbors": pm.n_neighbors, "SVM - kernel":pm.kernel,
            "SVM - C":pm.C, "SVM - gamma":pm.gamma, "KDE - bandwidth":pm.bandwidth}

    for nn, p in enumerate(p_range):
        #x_data, y_data = preprocess_ds1(attributes=attributes, p=p, q=0.1)
        if varsplit == "test":
            x_data, y_data = preprocess(attributes=attributes, p=pm.q, q=p)
            spq = "q"
        else:
            x_data, y_data = preprocess(attributes=attributes, p=p, q=pm.q)
            spq = "p"
        print("Testing accuracy for {0}={1:.2f}".format(spq, p))
        accuracy_p[0, nn] = kNN_method_N(x_data, y_data, N, n_neighbors=pm.n_neighbors, n_iters=n_iters)
        accuracy_p[1, nn] = SVM_method_N(x_data, y_data, N, kernel=pm.kernel, C=pm.C, gamma=pm.gamma, n_iters=n_iters)
        accuracy_p[2, nn] = KDE_method_N(x_data, y_data, N, bandwidth=pm.bandwidth, n_iters=n_iters)

    results_path = os.path.join(pm.results_dir, "split_pq", str(int(time())))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, 'params.txt'), 'w+') as f:
        for key, value in to_write.items():
            s = "{0}: {1}\n".format(key, value)
            f.write(s)

    if need_plot:
        plt.plot(p_range, np.ones(p_range.shape)/N, '.-', label='naive')
        methods = ['kNN', 'SVM', 'KDE']
        for ii, acc in enumerate(accuracy_p):
            plt.plot(p_range, acc, '.-', label=methods[ii])
        plt.xticks([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        plt.yticks([.2, .4, .6, .8, 1])
        plt.xlim((min(p_range), max(p_range)))
        plt.xlabel('Fraction of samples used for training')
        plt.ylabel('P[success]')
        plt.legend()
        plt.savefig(os.path.join(results_path, "plot.png"), dpi=100)
        plt.show();

    out_array = np.transpose(np.insert(accuracy_p, 0, p_range, 0))
    np.savetxt(os.path.join(results_path, "accuracy.csv"), out_array, delimiter=',')
