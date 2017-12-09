import numpy as np
from sklearn.model_selection import KFold

# Subsample dataframe
def subsample(data, ratio, seed=42):
    np.random.seed(seed)
    n = len(data)
    idxs = np.random.permutation(range(n))[:round(n*ratio)]
    return data.iloc[idxs]


# Get train and validation indexes
def train_val_idx(data, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    n = len(data)
    idxs = np.random.permutation(range(n))
    m = round(val_ratio*n)
    val_idxs = idxs[:m]
    trn_idxs = idxs[m:]
    return [(trn_idxs, val_idxs)]

# Get kfold cross validation indexes
def kfold_cv_idx(data, k, seed=42):
    np.random.seed(seed)
    kf = KFold(k)
    return list(kf.split(data))
