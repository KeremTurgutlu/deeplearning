import pandas as pd
import numpy as np
from scipy.special import erfinv

import torch
from torch import nn
from torch.utils.data import *
from torch.optim import *
from fastai.model import *
from fastai.column_data import *
from fastai.dataloader import *
from torch.utils.data import DataLoader as torch_dl

# Normalization

def to_gauss(x): return np.sqrt(2)*erfinv(x)

def normalize(data, exclude=None):
    norm_cols = [n for n, c in data.drop(exclude, 1).items() if len(np.unique(c)) > 2]
    n = data.shape[0]
    for col in norm_cols:
        sorted_idx = data[col].sort_values().index.tolist()
        uniform = np.linspace(start=-0.99, stop=0.99, num=n)
        normal = to_gauss(uniform)
        normalized_col = pd.Series(index=sorted_idx, data=normal)
        data[col] = normalized_col
    return data


# Denoising Functions, more can be added

def inputSwap(x, arr, p):
    ### Gives augmented version of the row
    ### Takes x and swaps each element with a probability of p
    ### With an element from arr that corresponds to the same column
    np.random.seed()
    x_tilde = x.copy()
    swap_idx = np.where(np.random.rand(len(x)) < p)[0]
    x_tilde[swap_idx] = np.array([np.random.choice(arr[:,c], 1)[0] for c in swap_idx])
    return (x, x_tilde)


# Data classes

class DAEDataset(Dataset):
    ### AutoEncoder dataset class
    ### Takes denoising function
    ### Some kind of denoising function is recommended
    ### if hidden layer sizes are smaller then input dimension
    ### since there is risk of finding identity
    ### x_tilde is (denoised) input x is output
    def __init__(self, arr, denoise=None, p=None):
        self.arr = arr
        self.denoise = denoise
        self.p = p

    def __len__(self): return self.arr.shape[0]

    def __getitem__(self, idx):
        x = self.arr[idx]
        x, x_tilde = self.get(x)
        return torch.from_numpy(x_tilde).float(), torch.from_numpy(x).float()  # x,y

    def get(self, row):
        return (x, row) if self.denoise is None else self.denoise(row, self.arr, self.p)


class DAEModelData(ModelData):
    def __init__(self, path, trn_ds, val_ds, bs):
        super().__init__(path, torch_dl(trn_ds, batch_size=bs, shuffle=False, num_workers=1)
                         , torch_dl(val_ds, batch_size=bs, shuffle=False, num_workers=1))

    @classmethod
    def from_array(cls, path, arr, val_idxs, trn_idxs, denoise_func, p, bs):
        arr_trn = arr[np.array(trn_idxs)]
        arr_val = arr[np.array(val_idxs)]
        return cls(path, DAEDataset(arr_trn, denoise_func, p),
                   DAEDataset(arr_val, denoise_func, p), bs)


# Model class
# After training get_features can be called with any tensor

class AutoEncoder(nn.Module):

    def __init__(self, layers, compute_activations=False):
        super().__init__()
        self.layers = layers
        self.compute_activations = compute_activations
        for i in range(len(layers) - 1):
            setattr(self, f"fc{i}", nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        self.activations = []
        for i in range(len(self.layers) - 1):
            dotprod = getattr(self, f"fc{i}")
            x = dotprod(x)
            if self.compute_activations:
                self.activations += [x]
        return x

    def get_activations(self, x):
        self.forward(x)
        return self.activations

    # TODO add param to get_faetures to specify which activations to use as new features ('bottleneck', 'stack')
    def get_features(self, x, type = 'stack'):
        self.compute_activations = True
        self.get_activations(x)
        features = torch.cat([act.data for act in self.activations[:-1]], 1)
        return features





