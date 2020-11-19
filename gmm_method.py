import numpy as np
import os

def GMM_score(x, y, iSigma):
    return np.array([np.sum(np.exp(-np.diagonal(np.dot(yl-x, np.dot(iSigma, np.transpose(yl-x)))) / 2 )) for yl in y])

def GMM_method_N(x_data, y_data, keep_diag, N, w=0.03, n_iters=1000, rule='majority'):
    n_correct = 0
    for i in range(n_iters):
        idx = np.random.choice(len(x_data), N, replace=False) # choose N users at random
        j_c = np.random.randint(N) # among those N users, choose 1 at random

        x_batch = x_data[idx]
        y = y_data[idx[j_c]]

        Sigma = w*np.eye(y.shape[1])
        if N > 1:
            Sigma = np.cov(np.transpose(np.concatenate(x_batch)))
            if keep_diag:
                Sigma = np.diag(np.diagonal(Sigma))
            Sigma = w*Sigma
        iSigma = np.linalg.inv(Sigma)

        #stddev = np.sqrt(np.diagonal(Sigma))
        #x_batch = [x/stddev for x in x_batch]
        #y = y/stddev

        scores = np.array([GMM_score(x, y, iSigma) for x in x_batch]) # compute the scores

        if rule == 'majority':
            # select the most likely user according to majority rule
            i_bests = np.argmax(scores, axis=0)
            if np.argmax(np.bincount(i_bests)) == j_c:
                n_correct += 1
        elif rule == 'sum':
            # select the most likely user summing the scores and choosing the argmax
            s = np.sum(scores, axis=1)
            if np.argmax(s) == j_c:
                n_correct += 1
        else:
            print('Error: rule does not exist.')

    return n_correct/n_iters

def run_GMM_method(x_data, y_data, keep_diag=False, w=0.03, N_max=20, n_iters=1000, rule = 'majority'):

    N_range = range(1, N_max+1)

    accuracy = np.zeros(len(N_range))
    for nn, N in enumerate(N_range):
        print("Testing GMM method for {0} user(s).".format(N))
        accuracy[nn] = GMM_method_N(x_data, y_data, keep_diag, N, w=0.03, n_iters=n_iters, rule=rule)

    return accuracy

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    pd.options.mode.chained_assignment = None  # default='warn'
    from preprocess import preprocess_dfs
    np.random.seed(1164)

    w=3e-5
    keep_diag = True
    N_max = 20
    n_iters = 1000

    datapath = "data"
    filepath1 = os.path.join(datapath, "month1.csv")
    filepath2 = os.path.join(datapath, "month2.csv")

    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    x_data, y_data = preprocess_dfs(df1, df2)

    accuracy = run_GMM_method(x_data, y_data, keep_diag=keep_diag, w=w, N_max=20, n_iters=n_iters, rule = 'majority')

    np.savetxt('results/accuracy_GMM.csv', accuracy, delimiter=',')

    N_range = range(1, N_max+1)
    #plt.plot(N_range, np.array([1/N for N in N_range]),'--', color='gray', label='Naive')
    plt.plot(N_range, accuracy,'.-', label='GMM')
    plt.xticks([5, 10, 15, 20])
    plt.yticks([.2, .4, .6, .8, 1])
    plt.xlabel('Number of users')
    plt.ylabel('P[success]')
    plt.legend()
    plt.savefig('figures/plot_GMM.png', dpi=100)
    plt.show();
