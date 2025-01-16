import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
from time import time

# Local
from src.preprocess_ds1 import preprocess_ds1
from src.preprocess_ds2 import preprocess_ds2
import src.sim_params as pm
from src.scores_threshold import scores_threshold_N
from src.utils import normalize_std


if __name__ == '__main__':

    DATASET = pm.DATASET
    N = pm.N
    rng_seed = pm.rng_seed
    need_plot = True

    exponent = pm.exponent

    np.random.seed(rng_seed) # set random seed
    attributes = pm.attributes

    if DATASET in ["ds2", "DS2", "pmd", "PMD"]:
        preprocess = preprocess_ds2
    else:
        preprocess = preprocess_ds1
    x_data, y_data = preprocess(attributes=attributes)

    to_write = {"DATASET": DATASET, "seed": rng_seed, 'N': N,
            "kNN - n_neighbors": pm.n_neighbors,
            'exponent': exponent}

    metrics = {'accuracy':[], 'precision':[], 'recall':[],
        'specificity':[], 'balanced_accuracy':[]}

    th_range = np.arange(2, 21, 2)

    for key in metrics:
        metrics[key] = np.zeros(len(th_range))

    for nn, threshold in enumerate(th_range):
        print("Testing threshold {0}.".format(threshold))
        results = scores_threshold_N(x_data, y_data, threshold, N, exponent=exponent, n_iters=pm.n_iters)
        for key in metrics:
            metrics[key][nn] = results[key]

    results_path = os.path.join(pm.results_dir, 'var_threshold', str(int(time())))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, 'params.txt'), 'w+') as f:
        for key, value in to_write.items():
            s = "{0}: {1}\n".format(key, value)
            f.write(s)


    if need_plot:
        for key in metrics:
            plt.plot(th_range, metrics[key], '.-')
            #plt.xticks(list(N_range))
            plt.yticks([.2, .4, .6, .8, 1])
            plt.xlim((min(th_range), max(th_range)))
            plt.xlabel('threshold')
            plt.ylabel(np.char.capitalize(key.replace("_", " ")))
            #plt.legend()
            plt.savefig(os.path.join(results_path, "{0}.png".format(key)), dpi=100)
            plt.show();

    for key in metrics:
        out_array = np.transpose(np.vstack((th_range, metrics[key])))
        np.savetxt(os.path.join(results_path, "{0}.csv".format(key)), out_array, delimiter=',')
