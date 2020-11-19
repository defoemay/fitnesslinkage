import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from time import time
from preprocess import preprocess_dfs
from svm_method import run_SVM_method
from knn_method import run_kNN_method
from kde_method import run_KDE_method


datapath = "data"
filepath1 = os.path.join(datapath, "month1.csv")
filepath2 = os.path.join(datapath, "month2.csv")

def test_methods(x_data, y_data, N_max):
    accuracy = {}
    accuracy['naive'] = np.array([1/n for n in range(1, N_max+1)])
    accuracy['svm'] = run_SVM_method(x_data, y_data, kernel='rbf', C=1, gamma=0.5, N_max=N_max)
    accuracy['knn'] = run_kNN_method(x_data, y_data, n_neighbors=4, N_max=N_max)
    accuracy['kde'] = run_KDE_method(x_data, y_data, bandwidth=0.5, N_max=N_max)

    return accuracy

if __name__ == "__main__":

    N_max = 5
    rng_seed = 1164

    np.random.seed(rng_seed) # set random seed
    pd.options.mode.chained_assignment = None  # remove warnings for pandas

    results_path = os.path.join("results", str(int(time())))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    x_data, y_data = preprocess_dfs(df1, df2)
    accuracy = test_methods(x_data, y_data, N_max=N_max)
    #print(accuracy)

    for method, acc in accuracy.items():
        np.savetxt(os.path.join(results_path, "accuracy_"+method+".csv"), acc, delimiter=',')
