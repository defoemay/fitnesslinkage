import pandas as pd
import numpy as np
import os
import sim_params as pm

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

def preprocess_pmd(attributes=['steps', 'calories'], p=0.5, q=-1):


    pmdata_dir = pm.pmdata_dir

    data_train = []
    data_test = []

    subfolders = [dd[1] for dd in os.walk(pmdata_dir)][0]
    #N = len([subdir in subfolders])

    for subdir in subfolders:
        file = os.path.join(pmdata_dir, subdir, "daily.csv")

        df = pd.read_csv(file)
        df1, df2 = split_pq(df, p, q)
        df1 = remove_rows_zero(df1)
        df2 = remove_rows_zero(df2)
        data_train.append(df1[attributes].values)
        data_test.append(df2[attributes].values)

    return np.array(data_train), np.array(data_test)
