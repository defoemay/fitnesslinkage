import os

data_dir = "data"
results_dir = "results"

pmdata_dir = os.path.join(data_dir, 'pmdata')

filepath1_ds1 = os.path.join(data_dir, "month1.csv")
filepath2_ds1 = os.path.join(data_dir, "month2.csv")


rng_seed = 1164
n_iters = 1000

# -------------------------------------------------------- #

DATASET = "ds1"

# -------------------------------------------------------- #

N_max = 29

# -------------------------------------------------------- #

N = 10

varsplit = "test"

# defining p_range
p_min = 0.1
p_max = 0.9
p_step = 0.05

q = 0.1

# -------------------------------------------------------- #

#attributes = ['steps']
#attributes = ['calories']
#attributes = ['steps', 'calories']

# -------------------------------------------------------- #
# kNN

n_neighbors = 1

# -------------------------------------------------------- #
# SVM

kernel = 'rbf'
C = 1e3
gamma = 0.5

# -------------------------------------------------------- #
# KDE

bandwidth = 0.5
