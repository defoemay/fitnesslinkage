import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

from preprocess_ds1 import preprocess_ds1
from preprocess_ds2 import preprocess_ds2
from all_methods import majority_stats_N
import sim_params as pm

if __name__ == "__main__":

    DATASET = pm.DATASET
    N = pm.N
    n_iters = pm.n_iters
    rng_seed = pm.rng_seed
    attributes = pm.attributes
    need_plot = True

    np.random.seed(rng_seed)

    if DATASET in ["ds2", "DS2", "pmd", "PMD"]:
        preprocess = preprocess_ds2
    else:
        preprocess = preprocess_ds1

    x_data, y_data = preprocess(attributes=attributes)

    hparams_list = [{'method':'kNN', 'n_neighbors': pm.n_neighbors},
                    {'method':'RF', 'n_estimators': pm.n_estimators},
                    {'method':'SVM', 'C':pm.gamma, 'gamma':pm.gamma},
                    {'method':'KDE', 'bandwidth':pm.bandwidth}]

    stats_list = [{} for hparams in hparams_list]

    to_write = {"DATASET": DATASET, "seed": rng_seed, "N": N,
            "kNN - n_neighbors": pm.n_neighbors, "RF - n_estimators": pm.n_estimators,
            "SVM - kernel":pm.kernel, "SVM - C":pm.C, "SVM - gamma":pm.gamma,
            "KDE - bandwidth":pm.bandwidth}


    bins=np.arange(0, 1.01, 0.1).tolist()

    for hh, hparams in enumerate(hparams_list):
        print("Computing stats for {0} method.".format(hparams['method']))
        stats_list[hh] = majority_stats_N(x_data, y_data, N, hparams, n_iters=n_iters, normalize=True, eps=0.1)

    results_path = os.path.join(pm.results_dir, "majority_stats", str(int(time())))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, 'params.txt'), 'w+') as f:
        for key, value in to_write.items():
            s = "{0}: {1}\n".format(key, value)
            f.write(s)

    if need_plot:
        title_list = {'p_T':'Votes to the true user', 'p_F':'Votes to the first candidate',
            'p_S':'Votes to the second candidate', 'p_diff_TF':'Vote difference between true and first',
            'p_diff_TS':'Vote difference between true and second', 'p_diff_FS':'Vote difference between first and second'}
        for hh, hparams in enumerate(hparams_list):
            stats = stats_list[hh]
            for key in stats:
                histc = stats[key]
                plt.hist(histc, 11, weights=np.ones_like(histc)/len(histc))
                plt.title(title_list[key]+' - '+hparams['method'])
                plt.savefig(os.path.join(results_path, "{0}_{1}.png".format(key, hparams['method'])), dpi=100)
                plt.show();
