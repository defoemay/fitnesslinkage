import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from preprocess import preprocess_dfs
from knn_method import kNN_method_N
from svm_method import SVM_method_N
from kde_method import KDE_method_N
import sim_params as pm

filepath1 = pm.filepath1
filepath2 = pm.filepath2


if __name__ == "__main__":

    N = pm.N
    n_iters = pm.n_iters
    rng_seed = pm.rng_seed
    attributes = pm.attributes
    need_plot = True

    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)

    p_range = np.arange(0.1, 0.91, 0.05)
    accuracy_p = np.zeros((3, p_range.shape[0]))
    for nn, p in enumerate(p_range):
        print("Testing accuracy for p={0:.2f}".format(p))
        x_data, y_data = preprocess_dfs(df1, df2, p=p)
        accuracy_p[0, nn] = kNN_method_N(x_data, y_data, N, n_neighbors=pm.n_neighbors, n_iters=n_iters)
        accuracy_p[1, nn] = SVM_method_N(x_data, y_data, N, kernel=pm.kernel, C=pm.C, gamma=pm.gamma, n_iters=n_iters)
        accuracy_p[2, nn] = KDE_method_N(x_data, y_data, N, bandwidth=pm.bandwidth, n_iters=n_iters)

    if need_plot:
        plt.plot(p_range, np.ones(p_range.shape)/N, '.-', label='naive')
        methods = ['kNN', 'SVM', 'KDE']
        for ii, acc in enumerate(accuracy_p):
            plt.plot(p_range, acc, '.-', label=methods[ii])
        plt.xticks([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        plt.yticks([.2, .4, .6, .8, 1])
        plt.xlim((.1, .9))
        plt.xlabel('Fraction of samples used for training')
        plt.ylabel('P[success]')
        plt.legend()
        plt.show();

    results_path = os.path.join(pm.results_dir, "split_p")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    out_array = np.transpose(np.insert(accuracy_p, 0, p_range, 0))
    np.savetxt(os.path.join(results_path, "accuracy_N"+str(int(N))+".csv"), out_array, delimiter=',')
