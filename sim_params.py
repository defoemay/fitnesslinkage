import os

data_dir = "data"
results_dir = "results"

datapath_ds1 = os.path.join(data_dir, "ds1")
datapath_ds2 = os.path.join(data_dir, "ds2")

rng_seed = 1164
n_iters = 1000

# -------------------------------------------------------- #

DATASET = "ds2"

# -------------------------------------------------------- #

N_max = 9

# -------------------------------------------------------- #

N = 5

# -------------------------------------------------------- #

varsplit = "train"

# defining p_range
p_min = 0.1
p_max = 0.9
p_step = 0.05

q = 0.1

# -------------------------------------------------------- #

threshold_min = 0
threshold_max = 1
threshold_step = 0.1

# -------------------------------------------------------- #

#attributes = ['steps']
#attributes = ['calories']
attributes = ['steps', 'calories']

# -------------------------------------------------------- #
# kNN

n_neighbors = 1

# -------------------------------------------------------- #
# RF

n_estimators = 10

# -------------------------------------------------------- #
# SVM

kernel = 'rbf'
C = 1e3
gamma = 0.5

# -------------------------------------------------------- #
# KDE

bandwidth = 0.5
