import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

def kNN_method_N(x_data, y_data, N, n_neighbors=4, n_iters=1000, eps=0.1):
    n_correct = 0
    for i in range(n_iters):
        idx = np.random.choice(len(x_data), N, replace=False) # choose N users at random
        j_c = np.random.randint(N) # among those N users, choose 1 at random

        x_batch = x_data[idx]
        y_batch = y_data[idx]
        y = y_batch[j_c]

        # normalize the features
        #print(x_batch[0].shape)
        if len(x_batch[0].shape) > 1 and x_batch[0].shape[0] != 1 and x_batch[0].shape[1] != 1:
            stddev = np.sqrt(np.diagonal(np.cov(np.transpose(np.concatenate(x_batch)))))
        else:
            stddev = np.std(np.concatenate(x_batch))
        stddev += eps
        x_batch = [x/stddev for x in x_batch]
        y = y/stddev

        X = np.vstack(x_batch)

        i_X = np.hstack([np.repeat(i, s) for i, s in enumerate(np.array([x.shape[0] for x in x_batch]))])

        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
        model.fit(X, i_X)
        j_preds = model.predict(y)

        # majority rule
        if np.argmax(np.bincount(j_preds)) == j_c:
            n_correct += 1

    return n_correct/n_iters

def run_kNN_method(x_data, y_data, n_neighbors=4, N_max=20, n_iters=1000):

    N_range = range(1, N_max+1)

    accuracy = np.zeros(len(N_range))

    for nn, N in enumerate(N_range):
        print("Testing kNN method for {0} user(s).".format(N))
        accuracy[nn] = kNN_method_N(x_data, y_data, N, n_neighbors=n_neighbors, n_iters=n_iters)

    return accuracy

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    pd.options.mode.chained_assignment = None  # default='warn'
    from preprocess import preprocess_dfs

    n_neighbors = 2
    N_max = 20
    n_iters = 1000

    datapath = "data"
    filepath1 = os.path.join(datapath, "month1.csv")
    filepath2 = os.path.join(datapath, "month2.csv")

    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    x_data, y_data = preprocess_dfs(df1, df2)

    accuracy = run_kNN_method(x_data, y_data, n_neighbors=n_neighbors, N_max=N_max, n_iters=n_iters)

    np.savetxt('results/accuracy_kNN_{0}.csv'.format(n_neighbors), accuracy, delimiter=',')

    N_range = range(1, N_max+1)
    #plt.plot(N_range, np.array([1/N for N in N_range]),'--', color='gray', label='Naive')
    plt.plot(N_range, accuracy,'.-', label='kNN')
    plt.xticks([5, 10, 15, 20])
    plt.yticks([.2, .4, .6, .8, 1])
    plt.xlabel('Number of users')
    plt.ylabel('P[success]')
    plt.legend()
    plt.savefig('figures/plot_kNN.png', dpi=100)
    plt.show();
