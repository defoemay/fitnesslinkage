import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import normalize_std
import os

def model_fn(hparams):
    if hparams['method'] == 'kNN':
        return KNeighborsClassifier(n_neighbors=hparams['n_neighbors'], algorithm='auto')
    elif hparams['method'] == 'RF':
        return RandomForestClassifier(n_estimators=hparams['n_estimators'], criterion='gini')
    elif hparams['method'] == 'SVM':
        return SVC(kernel='rbf', C=hparams['C'], gamma=hparams['gamma'])
    # default is Nearest Neighbors
    return KNeighborsClassifier(n_neighbors=1, algorithm='auto')

def majority_method_N(x_data, y_data, hparams, N, n_iters=1000, normalize=True, eps=0.1):

    assert len(x_data) == len(y_data)

    n_correct = 0
    for iter in range(n_iters):
        idx = np.random.choice(len(x_data), N, replace=False) # choose N users at random
        j_c = np.random.randint(N) # among those N users, choose 1 at random

        x_batch = x_data[idx]
        y_batch = y_data[idx]
        y = y_batch[j_c]

        if normalize:
            x_batch, y = normalize_std(x_batch, y, eps=eps)

        X = np.vstack(x_batch)

        i_X = np.hstack([np.repeat(i, s) for i, s in enumerate(np.array([x.shape[0] for x in x_batch]))])

        model = model_fn(hparams)
        model.fit(X, i_X)
        j_preds = model.predict(y)

        # majority rule
        if np.argmax(np.bincount(j_preds)) == j_c:
            n_correct += 1
    return n_correct/n_iters

def majority_method(x_data, y_data, hparams, N_max=20, n_iters=1000, normalize=True, eps=0.1):

    N_range = range(1, N_max+1)

    accuracy = np.zeros(len(N_range))
    print("Testing {0} method for {1} user.".format(hparams['method'], 1))
    accuracy[0] = 1
    for nn, N in enumerate(N_range[1:len(N_range)]):
        print("Testing {0} method for {1} users.".format(hparams['method'], N))
        accuracy[nn+1] = majority_method_N(x_data, y_data, hparams, N, n_iters=n_iters, normalize=normalize, eps=eps)

    return accuracy

def majority_stats_N(x_data, y_data, hparams, N, n_iters=1000, normalize=True, eps=0.1, in_set=False):

    assert len(x_data) == len(y_data)

    p_T = np.zeros(n_iters) # stats of actual target
    p_F = np.zeros(n_iters) # stats of first candidate
    p_S = np.zeros(n_iters) # stats of second candidate

    for iter in range(n_iters):
        idx = np.random.choice(len(x_data), N, replace=False) # choose N users at random
        x_batch = x_data[idx]

        if in_set:
            j_c = np.random.randint(N) # among those N users, choose 1 at random
            y_batch = y_data[idx]
            y = y_batch[j_c]
        else:
            idxc = np.array(list(set(range(len(x_data))) - set(idx)))
            j_c = np.random.choice(idxc, 1)[0]
            y = y_data[j_c]

        if normalize:
            x_batch, y = normalize_std(x_batch, y, eps=eps)

        X = np.vstack(x_batch)

        i_X = np.hstack([np.repeat(i, s) for i, s in enumerate(np.array([x.shape[0] for x in x_batch]))])

        model = model_fn(hparams)
        model.fit(X, i_X)
        j_preds = model.predict(y)

        bincounts = np.bincount(j_preds)

        bincounts_sorted = np.flip(np.sort(bincounts))
        best_candidates = np.flip(np.argsort(bincounts))

        if in_set:
            if len(bincounts) > j_c:
                p_T[iter] = bincounts[j_c]/len(y)
            else:
                p_T[iter] = 0
        p_F[iter] = bincounts_sorted[0]/len(y)
        if len(bincounts_sorted) > 1:
            p_S[iter] = bincounts_sorted[1]/len(y)
        else:
            p_S[iter] = 0

    if in_set:
        p_diff_TF = np.abs(p_T - p_F) # difference between actual target and first
        p_diff_TS = np.abs(p_T - p_S) # difference between actual target and second
    p_diff_FS = np.abs(p_F - p_S) # difference between actual target and second

    if in_set:
        stats = {'p_T':p_T, 'p_F':p_F, 'p_S':p_S,
            'p_diff_TF':p_diff_TF, 'p_diff_TS':p_diff_TS, 'p_diff_FS':p_diff_FS}
    else:
        stats = {'p_F':p_F, 'p_S':p_S, 'p_diff_FS':p_diff_FS}

    return stats

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    pd.options.mode.chained_assignment = None  # default='warn'
    from preprocess_ds1 import preprocess_ds1

    N_max = 5
    n_iters = 1000
    normalize = True
    eps = 0.1

    hparams = {
        'method': 'kNN',
        'n_neighbors': 1
    }


    x_data, y_data = preprocess_ds1()

    accuracy = majority_method(x_data, y_data, hparams, N_max=N_max, n_iters=n_iters, normalize=normalize, eps=eps)

    np.savetxt('results/accuracy_{0}.csv'.format(hparams['method']), accuracy, delimiter=',')

    N_range = range(1, N_max+1)
    #plt.plot(N_range, np.array([1/N for N in N_range]),'--', color='gray', label='Naive')
    plt.plot(N_range, accuracy,'.-', label=hparams['method'])
    plt.xticks(list(range(0, N_max+1, 5)))
    plt.yticks([.2, .4, .6, .8, 1])
    plt.xlim([0, N_max])
    plt.xlabel('Number of users')
    plt.ylabel('P[success]')
    plt.legend()
    plt.savefig('figures/plot_{0}.png'.format(hparams['method']), dpi=100)
    plt.show();
