import os

import pandas as pd
import numpy as np

import src.sim_params as pm

def preprocess_dfs(df1, df2, attributes=['steps', 'calories'], p=0.5, q=-1):

    def keep_common_ids(df1, df2):
        ids1 = df1['id'].unique()
        ids2 = df2['id'].unique()
        ids = [idn for idn in ids1 if idn in ids2]
        df1 = df1[df1['id'].isin(ids)]
        df2 = df2[df2['id'].isin(ids)]
        return df1, df2

    def get_samples(df, attributes=['steps', 'calories']):
        df['samples'] = df[attributes].values.tolist()
        x_data = df.groupby("id")['samples'].apply(np.array).to_numpy()
        return np.array([np.stack(x) for x in x_data])

    def split_pq(df1, df2, p, q=-1):
        if p < 0 or p > 1: # should never happen
            p  = 0.5
        if q < 0 or q > 1-p:
            q = 1-p
        df2.index = df2.index + df1.shape[0]
        df = pd.concat([df1, df2]).sort_values(by=['date', 'id'])
        T = df.shape[0] # number of samples per user
        i1 = int(p*T)
        i2 = max(i1, int((1-q)*T))
        df1 = df.iloc[:i1, :]
        df2 = df.iloc[i2:, :]
        return keep_common_ids(df1, df2)

    def find_idx_bad(x_data):
        Tmax = max([len(x) for x in x_data])
        return [i for i,x in enumerate(x_data) if len(x)<0.5*Tmax]

    def remove_bad_data(x_data, y_data):
        idx_badx = find_idx_bad(x_data)
        idx_bady = find_idx_bad(y_data)

        idx_good = list(set(range(len(x_data)))-set(idx_badx)-set(idx_bady))
        x_data = x_data[idx_good]
        y_data = y_data[idx_good]
        return x_data, y_data

    # rename the columns
    df1.columns = ['id', 'date', 'steps', 'distance', 'td', 'ld', 'distance3', 'distance2', 'distance1', 'distance0',
              'minutes3', 'minutes2', 'minutes1', 'minutes0', 'calories']
    df2.columns = df1.columns

    # keep only the users which are recorded on both months
    df1, df2 = keep_common_ids(df1, df2)

    # if p is in [0,1], split the dataset
    if p > 0 and p < 1:
        df1, df2 = split_pq(df1, df2, p, q)

    x_data = get_samples(df1, attributes) # data from the first month
    y_data = get_samples(df2, attributes) # data from the second month

    x_data = x_data[:29]
    y_data = y_data[:29]

    #x_data, y_data = remove_bad_data(x_data, y_data)

    return x_data, y_data

def preprocess_ds1(attributes=['steps', 'calories'], p=0.5, q=-1):

    filepath1 = os.path.join(pm.datapath_ds1, "month1.csv")
    filepath2 = os.path.join(pm.datapath_ds1, "month2.csv")

    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    return preprocess_dfs(df1, df2, attributes=attributes, p=p, q=q)
