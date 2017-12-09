import numpy as np
import pandas as pd
from scipy.special import erfinv


##############################
# STANDARDIZE/NORMALIZE DATA
##############################


# Rankgauss normalization, seen in Porto-Seguro Kaggle winning solution
def to_gauss(x): return np.sqrt(2)*erfinv(x)

def rankgauss(data, exclude=None):
    norm_cols = [n for n, c in data.drop(exclude, 1).items() if len(np.unique(c)) > 2]
    n = data.shape[0]
    for col in norm_cols:
        sorted_idx = data[col].sort_values().index.tolist()
        uniform = np.linspace(start=-0.99, stop=0.99, num=n)
        normal = to_gauss(uniform)
        normalized_col = pd.Series(index=sorted_idx, data=normal)
        data[col] = normalized_col
    return data


##############################
# DATA PREPROCESS
##############################

# For now it creates one-hot encodes
# cats: variables to ohe
# keep: whether to keep original cats

def num_df(data, cats, keep=True):
    dummies = pd.get_dummies(data[cats], columns=cats)
    if keep: res = pd.concat([data, dummies], 1)
    else: res = pd.concat([data.drop(cats, 1), dummies], 1)
    return res


