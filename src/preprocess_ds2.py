import os
import pandas as pd
import numpy as np

import src.sim_params as pm


def split_pq(df, p, q):
    T = df.shape[0] # number of samples per user
    if p < 0 or p > 1:
        p  = 0.5
    if q < 0 or q > 1-p:
        q = 1-p

    i1 = int(p*T)
    i2 = max(i1, int((1-q)*T))

    df1 = df.iloc[:i1, :]
    df2 = df.iloc[i2:, :]
    return df1, df2


def remove_rows_zero(df):
    return df[(df.T != 0).all()]


def find_idx_bad(x_data):
    Tmax = max([len(x) for x in x_data])
    return [i for i,x in enumerate(x_data) if len(x)<0.5*Tmax]

def remove_bad_data(x_data, y_data):
    idx_badx = find_idx_bad(x_data)
    idx_bady = find_idx_bad(y_data)

    idx_good = list(set(range(len(x_data)))-set(idx_badx)-set(idx_bady))
    x_data = [x for i, x in enumerate(x_data) if i in idx_good]
    y_data = [y for i, y in enumerate(y_data) if i in idx_good]
    return x_data, y_data

def preprocess_ds2(attributes=['steps', 'calories'], p=0.5, q=-1):

    datapath = pm.datapath_ds2

    data_train = []
    data_test = []

    subfolders = [dd[1] for dd in os.walk(datapath)][0]

    for subdir in subfolders:
        file = os.path.join(datapath, subdir, "daily.csv")

        df = pd.read_csv(file)
        df1, df2 = split_pq(df, p, q)
        df1 = remove_rows_zero(df1)
        df2 = remove_rows_zero(df2)
        data_train.append(df1[attributes].values)
        data_test.append(df2[attributes].values)

    x_data = data_train
    y_data = data_test

    idx_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    x_data = [x for i, x in enumerate(x_data) if i in idx_keep]
    y_data = [y for i, y in enumerate(y_data) if i in idx_keep]
    #x_data, y_data = remove_bad_data(x_data, y_data)

    return x_data, y_data
