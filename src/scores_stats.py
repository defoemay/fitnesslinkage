import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
from time import time

# Local
from src.preprocess_ds1 import preprocess_ds1
from src.preprocess_ds2 import preprocess_ds2
from src.utils import normalize_std
import src.sim_params as pm


if __name__ == "__main__":

    DATASET = pm.DATASET
    N_max = pm.N_max
    n_iters = pm.n_iters
    rng_seed = pm.rng_seed

    np.random.seed(rng_seed) # set random seed
    attributes = pm.attributes

    if DATASET in ["ds2", "DS2", "pmd", "PMD"]:
        preprocess = preprocess_ds2
    else:
        preprocess = preprocess_ds1
    x_data, y_data = preprocess(attributes=attributes)

    results = {'in':{}, 'out':{}}
    results['in'] = {'ss_min':np.zeros(N_max), 'ss_25q':np.zeros(N_max), 'ss_median':np.zeros(N_max), 'ss_75q':np.zeros(N_max), 'ss_max':np.zeros(N_max)}
    results['out'] = {'ss_min':np.zeros(N_max), 'ss_25q':np.zeros(N_max), 'ss_median':np.zeros(N_max), 'ss_75q':np.zeros(N_max), 'ss_max':np.zeros(N_max)}

    to_write = {"DATASET": DATASET, "seed": rng_seed, "N_max": N_max,
            "kNN - n_neighbors": pm.n_neighbors}

    results_path = os.path.join(pm.results_dir, 'scores_stats', str(int(time())))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, 'params.txt'), 'w+') as f:
        for key, value in to_write.items():
            s = "{0}: {1}\n".format(key, value)
            f.write(s)

    for nn, N in enumerate(range(1,N_max+1)):

        print("Evaluating stats for {0} user(s).".format(N))
        sum_scores = {'in':[], 'out':[]}

        for iter in range(n_iters):
            idx = np.random.choice(len(x_data), N, replace=False) # choose N users at random
            i_c = np.random.choice(len(x_data)) # among those N users, choose 1 at random

            x_batch = [x for i, x in enumerate(x_data) if i in idx]
            y = y_data[i_c]

            x_batch, y = normalize_std(x_batch, y, eps=0.1)

            X = np.vstack(x_batch)
            i_X = np.hstack([np.repeat(i, s) for i, s in enumerate(np.array([x.shape[0] for x in x_batch]))])

            model =  KNeighborsClassifier(n_neighbors=pm.n_neighbors, algorithm='auto')
            model.fit(X, i_X)

            votes = model.predict(y)
            ii_hat = np.argmax(np.bincount(votes))
            i_hat = idx[ii_hat]

            scores, _ = model.kneighbors(y, 1)

            #ss = 100*np.sum(scores[votes == ii_hat])/len(y)#/N
            ss = 100*np.sum(scores)/len(y)#/N # count all the votes

            if i_c in idx:
                sum_scores['in'].append( ss )
            else:
                sum_scores['out'].append( ss )

        fig, ax = plt.subplots()
        ax.boxplot([sum_scores['in'], sum_scores['out']], sym='')
        ax.set_xticklabels(['In', 'Out']);
        plt.savefig(os.path.join(results_path, "boxplot_{0}.png".format(N)), dpi=100)
        plt.close(fig=fig)

        for key in results:
            results[key]['ss_min'][nn] = np.min(sum_scores[key])
            results[key]['ss_25q'][nn] = np.quantile(sum_scores[key], 0.25)
            results[key]['ss_median'][nn] = np.quantile(sum_scores[key], 0.5)
            results[key]['ss_75q'][nn] = np.quantile(sum_scores[key], 0.75)
            results[key]['ss_max'][nn] = np.max(sum_scores[key])

    for key in results:
        out_array = np.array([results[key][metric] for metric in results[key]])
        out_array = np.insert(out_array, 0, np.arange(1, N_max+1), 0)
        out_array = np.transpose(out_array)
        np.savetxt(os.path.join(results_path, "stats_{0}.csv".format(key)), out_array, delimiter=',')
