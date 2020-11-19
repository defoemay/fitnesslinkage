import numpy as np
from sklearn.neighbors import KernelDensity
import os

def KDE_method_N(x_data, y_data, N, bandwidth=0.5, n_iters=1000):
    n_correct = 0
    for i in range(n_iters):
        idx = np.random.choice(len(x_data), N, replace=False) # choose N users at random
        j_c = np.random.randint(N) # among those N users, choose 1 at random

        x_batch = x_data[idx]
        y_batch = y_data[idx]
        y = y_batch[j_c]

        #stddev = np.sqrt(np.diagonal(np.cov(np.transpose(np.concatenate(x_data)))))
        #x_batch = [x/stddev for x in x_batch]
        #y = y/stddev

        X = np.vstack(x_batch)

        i_X = np.hstack([np.repeat(i, s) for i, s in enumerate(np.array([x.shape[0] for x in x_batch]))])

        scores = np.zeros((N, len(y)))
        #print("bandwidth: ", bandwidth)
        model = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        for j in range(N):
            model.fit(X[i_X == j])
            scores[j, :] = model.score_samples(y)
        j_preds = np.argmax(scores, axis=0)
        #print(j_preds)

        # majority rule
        if np.argmax(np.bincount(j_preds)) == j_c:
            n_correct += 1

    return n_correct/n_iters

def run_KDE_method(x_data, y_data, bandwidth=0.5, N_max=20, n_iters=1000):

    N_range = range(1, N_max+1)

    accuracy = np.zeros(len(N_range))

    for nn, N in enumerate(N_range):
        print("Testing KDE method for {0} user(s).".format(N))
        accuracy[nn] = KDE_method_N(x_data, y_data, N, bandwidth=bandwidth, n_iters=n_iters)

    return accuracy

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    pd.options.mode.chained_assignment = None  # default='warn'
    from preprocess import preprocess_dfs

    bandwidth=10
    N_max = 20
    n_iters = 1000

    datapath = "data"
    filepath1 = os.path.join(datapath, "month1.csv")
    filepath2 = os.path.join(datapath, "month2.csv")

    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    x_data, y_data = preprocess_dfs(df1, df2)

    accuracy = run_KDE_method(x_data, y_data, bandwidth=bandwidth, N_max=N_max, n_iters=n_iters)

    np.savetxt('results/accuracy_KDE.csv', accuracy, delimiter=',')

    N_range = range(1, N_max+1)
    #plt.plot(N_range, np.array([1/N for N in N_range]),'--', color='gray', label='Naive')
    plt.plot(N_range, accuracy,'.-', label='KDE')
    plt.xticks([5, 10, 15, 20])
    plt.yticks([.2, .4, .6, .8, 1])
    plt.xlabel('Number of users')
    plt.ylabel('P[success]')
    plt.legend()
    plt.savefig('figures/plot_KDE.png', dpi=100)
    plt.show();
