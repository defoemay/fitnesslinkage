import pandas as pd
import numpy as np

def preprocess_dfs(df1, df2):

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

    # rename the columns
    df1.columns = ['id', 'date', 'steps', 'distance', 'td', 'ld', 'distance3', 'distance2', 'distance1', 'distance0',
              'minutes3', 'minutes2', 'minutes1', 'minutes0', 'calories']
    df2.columns = df1.columns

    # keep only the users which are recorded on both months
    df1, df2 = keep_common_ids(df1, df2)

    x_data = get_samples(df1) # data from the first month
    y_data = get_samples(df2) # data from the second month
    return x_data, y_data
