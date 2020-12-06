import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from time import time

from preprocess_ds1 import preprocess_ds1
from preprocess_pmd import preprocess_pmd
from svm_method import run_SVM_method
from knn_method import run_kNN_method
from kde_method import run_KDE_method
import sim_params as pm


def test_methods(x_data, y_data, N_max):
    accuracy = {}
    accuracy['naive'] = np.array([1/n for n in range(1, N_max+1)])
    accuracy['kNN'] = run_kNN_method(x_data, y_data, n_neighbors=pm.n_neighbors, N_max=N_max)
    accuracy['SVM'] = run_SVM_method(x_data, y_data, kernel=pm.kernel, C=pm.C, gamma=pm.gamma, N_max=N_max)
    accuracy['KDE'] = run_KDE_method(x_data, y_data, bandwidth=pm.bandwidth, N_max=N_max)

    return accuracy

if __name__ == "__main__":

    DATASET = pm.DATASET
    N_max = pm.N_max
    rng_seed = pm.rng_seed
    #p = 0.5
    need_plot = True

    np.random.seed(rng_seed) # set random seed
    pd.options.mode.chained_assignment = None  # remove warnings for pandas
    attributes = pm.attributes

    attr_str = '_'.join(attributes)

    if DATASET == "pmdata":
        x_data, y_data = preprocess_pmd(attributes=attributes)
    else:
        x_data, y_data = preprocess_ds1(attributes=attributes)
    #print(x_data)
    accuracy = test_methods(x_data, y_data, N_max=N_max)
    #print(accuracy)

    results_path = os.path.join(pm.results_dir, attr_str, str(int(time())))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    to_write = {"DATASET": DATASET, "seed": rng_seed, "N_max": N_max,
            "kNN - n_neighbors": pm.n_neighbors, "SVM - kernel":pm.kernel,
            "SVM - C":pm.C, "SVM - gamma":pm.gamma, "KDE - bandwidth":pm.bandwidth}

    with open(os.path.join(results_path, 'params.txt'), 'w+') as f:
        for key, value in to_write.items():
            s = "{0}: {1}\n".format(key, value)
            f.write(s)


    if need_plot:
        N_range = np.array(range(1, N_max+1))
        for method, acc in accuracy.items():
            plt.plot(N_range, acc, '.-', label=method)
        #plt.xticks([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        plt.yticks([.2, .4, .6, .8, 1])
        plt.xlim((1, N_max))
        plt.xlabel('Number of users')
        plt.ylabel('P[success]')
        plt.legend()
        plt.savefig(os.path.join(results_path, "plot.png"), dpi=100)
        plt.show();


    for method, acc in accuracy.items():
        np.savetxt(os.path.join(results_path, "accuracy_"+method+".csv"), acc, delimiter=',')
