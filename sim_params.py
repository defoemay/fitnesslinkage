import os

data_dir = "data"
results_dir = "results"

pmdata_dir = os.path.join(data_dir, 'pmdata')

filepath1_ds1 = os.path.join(data_dir, "month1.csv")
filepath2_ds1 = os.path.join(data_dir, "month2.csv")


rng_seed = 1164
n_iters = 1000

# -------------------------------------------------------- #

N_max = 16

#attributes = ['steps']
attributes = ['calories']
#attributes = ['steps', 'calories']

# -------------------------------------------------------- #

N = 10

# -------------------------------------------------------- #
# kNN

n_neighbors = 1

# -------------------------------------------------------- #
# SVM

kernel = 'rbf'
C = 1e4
gamma = 0.5

# -------------------------------------------------------- #
# KDE

bandwidth = 0.5
