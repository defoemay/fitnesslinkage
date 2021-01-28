import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from time import time

from preprocess_ds1 import preprocess_ds1
from preprocess_ds2 import preprocess_ds2
from all_methods import majority_method
import sim_params as pm


def test_methods(x_data, y_data, hparams_list, N_max):
    accuracy = {}
    accuracy['naive'] = np.array([1/n for n in range(1, N_max+1)])
    for hh, hparams in enumerate(hparams_list):
        accuracy[hparams['method']] = majority_method(x_data, y_data,
            hparams, N_max=N_max, n_iters=pm.n_iters)

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

    if DATASET in ["ds2", "DS2", "pmd", "PMD"]:
        preprocess = preprocess_ds2
    else:
        preprocess = preprocess_ds1
    x_data, y_data = preprocess(attributes=attributes)

    hparams_list = [{'method':'kNN', 'n_neighbors': pm.n_neighbors},
                    {'method':'RF', 'n_estimators': pm.n_estimators},
                    {'method':'SVM', 'C':pm.gamma, 'gamma':pm.gamma},
                    {'method':'KDE', 'bandwidth':pm.bandwidth}]


    to_write = {"DATASET": DATASET, "seed": rng_seed, "N_max": N_max,
            "kNN - n_neighbors": pm.n_neighbors, "RF - n_estimators": pm.n_estimators,
            "SVM - kernel":pm.kernel, "SVM - C":pm.C, "SVM - gamma":pm.gamma,
            "KDE - bandwidth":pm.bandwidth}

    accuracy = test_methods(x_data, y_data, hparams_list, N_max=N_max)
    #print(accuracy)

    results_path = os.path.join(pm.results_dir, attr_str, str(int(time())))
    if not os.path.exists(results_path):
        os.makedirs(results_path)



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
