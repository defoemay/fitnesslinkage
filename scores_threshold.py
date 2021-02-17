import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from preprocess_ds1 import preprocess_ds1
from preprocess_ds2 import preprocess_ds2
from sklearn.neighbors import KNeighborsClassifier
from utils import normalize_std
import sim_params as pm
import os
from time import time

def scores_threshold_N(x_data, y_data, threshold, N, n_iters=1000, exponent=1, normalize=True, eps=0.1):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for iter in range(n_iters):
        idx = np.random.choice(len(x_data), N, replace=False) # choose N users at random
        i_c = np.random.choice(len(x_data)) # among those N users, choose 1 at random

        x_batch = x_data[idx]
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

        #ss = 100*np.sum(scores[votes == ii_hat])/len(y)
        ss = 100*np.sum(scores)/len(y) # count all scores

        cond1 = ss*(N**exponent) < threshold
        cond2 = i_c in idx
        cond3 = i_c == i_hat

        H_pred = cond1
        H_true = cond2

        #H_pred = H_pred and i_hat == i_c # comment this for the reduced rule

        # if H_pred and H_true:
        #     tp += 1
        # elif H_pred and not H_true:
        #     fp += 1
        # elif not H_pred and H_true:
        #     fn += 1
        # elif not H_pred and not H_true:
        #     tn += 1
        # else:
        #     print("This line should not be reached.")

        if cond1 and cond3:
            tp += 1
        elif cond1 and not cond3:
            fp += 1
        elif not cond1 and cond2:
            fn += 1
        elif not cond1 and not cond2:
            tn += 1
        else:
            print("This point should not be reached.")

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    if (tp + fp) != 0:
        precision = tp/(tp + fp)
    else:
        precision = 1
    if (tp + fn) != 0:
        recall = tp/(tp + fn) # true positive rate
    else:
        recall = 0
    if (tn + fp) != 0:
        specificity = tn/(tn + fp) # true negative rate
    else:
        specificity = 0
    balanced_accuracy = (recall + specificity)/2

    results = {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "specificity": specificity,
        "balanced_accuracy":balanced_accuracy
    }

    return results

if __name__ == '__main__':

    DATASET = pm.DATASET
    N_max = pm.N_max
    rng_seed = pm.rng_seed
    need_plot = True

    threshold = pm.threshold
    exponent = pm.exponent

    np.random.seed(rng_seed) # set random seed
    attributes = pm.attributes

    if DATASET in ["ds2", "DS2", "pmd", "PMD"]:
        preprocess = preprocess_ds2
    else:
        preprocess = preprocess_ds1
    x_data, y_data = preprocess(attributes=attributes)

    to_write = {"DATASET": DATASET, "seed": rng_seed, "N_max": N_max,
            "kNN - n_neighbors": pm.n_neighbors, 'threshold': threshold,
            'exponent': exponent}

    metrics = {'accuracy':[], 'precision':[], 'recall':[],
        'specificity':[], 'balanced_accuracy':[]}

    N_range = range(1, N_max+1)

    for key in metrics:
        metrics[key] = np.zeros(len(N_range))

    for nn, N in enumerate(N_range):
        print("Testing threshold for {0} user(s).".format(N))
        results = scores_threshold_N(x_data, y_data, threshold, N, exponent=exponent, n_iters=pm.n_iters)
        for key in metrics:
            metrics[key][nn] = results[key]

    results_path = os.path.join(pm.results_dir, 'scores_threshold', str(int(time())))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, 'params.txt'), 'w+') as f:
        for key, value in to_write.items():
            s = "{0}: {1}\n".format(key, value)
            f.write(s)

    if need_plot:
        for key in metrics:
            plt.plot(N_range, metrics[key], '.-')
            #plt.xticks(list(N_range))
            plt.yticks([.2, .4, .6, .8, 1])
            plt.xlim((min(N_range), max(N_range)))
            plt.xlabel('N')
            plt.ylabel(np.char.capitalize(key.replace("_", " ")))
            #plt.legend()
            plt.savefig(os.path.join(results_path, "{0}.png".format(key)), dpi=100)
            plt.show();

    for key in metrics:
        out_array = np.transpose(np.insert(metrics[key], 0, N_range, 0))
        np.savetxt(os.path.join(results_path, "{0}.csv".format(key)), out_array, delimiter=',')
