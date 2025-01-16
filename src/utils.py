import numpy as np

def normalize_std(x_batch, y, eps=0.1):
    if len(x_batch[0].shape) > 1 and x_batch[0].shape[0] != 1 and x_batch[0].shape[1] != 1:
        stddev = np.sqrt(np.diagonal(np.cov(np.transpose(np.concatenate(x_batch)))))
    else:
        stddev = np.std(np.concatenate(x_batch))
    stddev += eps
    x_batch = [x/stddev for x in x_batch]
    y = y/stddev
    return x_batch, y

def normalize_bound(x_batch, y):
    xmax = np.max(np.concatenate(x_batch), axis=0)
    xmin = np.min(np.concatenate(x_batch), axis=0)
    x_batch = [(x-xmin)/(xmax-xmin) for x in x_batch]
    y = (y-xmin)/(xmax-xmin)
    return x_batch, y
