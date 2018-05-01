# IMPORTS
import torch
import pandas as pd
import numpy as np
import sys
import argparse
from models import *

np.random.seed(7)

# IMPORT THESE FROM LOCAL - SCHOOL GPU PROBLEM
sys.path.append('../../../fastai/')
from fastai.column_data import *
from fastai.structured import *

# SET ARGUMENTS
parser = argparse.ArgumentParser(description="Talking Data")
parser.add_argument("-g" ,"--gpu", default=0)
parser.add_argument("-s", "--sampleratio", type=int, default=None)
parser.add_argument("train_path") # feather
parser.add_argument("val_path") # feather
parser.add_argument("test_path") # feather

parser.add_argument("emb_drop", default=0.5, type=float)
parser.add_argument("szs", default=[500,500], type=list)
parser.add_argument("drops", default=0.5, type=float)
parser.add_argument("cross_depth", default=4, type=int)

# PARSE ARGUMENTS
args=parser.parse_args()

gpu = args.gpu
sample_ratio = args.sampleratio
train_path = args.train_path
val_path = args.val_path
test_path = args.test_path
emb_drop = args.emb_drop
szs = args.szs
drops = [args.drops]*len(szs)
cross_depth = args.cross_depth

# SET CUDA
torch.cuda.set_device(gpu)


# DATA PREP

"""
train_data and val_data should have all features + target
test data should have all features 
all data needs to be encoded and generated together without any leakage
"""

train_data = pd.read_feather(train_path)
val_data = pd.read_feather(val_path)
test = pd.read_feather(test_path)


# trains with samples if sample_ratio is given
if sample_ratio is not None:
    train_idx = np.random.choice(train_data.index,
                                 size=int(len(train_data)*sample_ratio),
                                 replace=False)
    val_idx = np.random.choice(val_data.index,
                               size=int(len(val_data)*sample_ratio),
                               replace=False)
    # create sample data
    train_sample = train_data.iloc[train_idx].reset_index(drop=True, inplace=False)
    val_sample = val_data.iloc[val_idx].reset_index(drop=True, inplace=False)
else:
    train_sample = train_data.reset_index(drop=True, inplace=False)
    val_sample = val_data.reset_index(drop=True, inplace=False)


# create dataframes
trn_df = train_sample.drop("is_attributed",1)
trn_y = train_sample.is_attributed.reset_index(drop=True, inplace=False)
val_df = val_sample.drop("is_attributed",1)
val_y = val_sample.is_attributed.reset_index(drop=True, inplace=False)
test_df = test


# get cat sizes
cats = ['ip', 'app', 'device', 'os', 'channel','click_timeHour']
cat_sz = [(c, len(pd.concat([trn_df, val_df, test_df])[c].unique())) for c in cats]
# create embedding sizes
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
#emb_szs = [(c, min(50, int(6*((c)^1/4)))) for _,c in cat_sz]
# nconts
n_conts = len(trn_df.columns) - len(cats)


# initialize model
model = CrossDenseNN(emb_szs,
                    n_cont=n_conts,
                    emb_drop=emb_drop,
                    out_sz=2,
                    szs=szs,
                    drops=drops,
                    cross_depth = cross_depth,
                    is_reg=False,
                    is_multi=False).cuda(gpu)


bm = BasicModel(model, 'binary_classifier')

# initialize model data
md = ColumnarModelData.from_data_frames('/tmp',
                                        trn_df,
                                        val_df,
                                        trn_y,
                                        val_y,
                                        cats,
                                        512, False, False, test_df=test_df)

# initialize learner
learn = StructuredLearner(md, bm)


# Good learning rate add schedular if needed
lr = 4e-3
n_epochs = 100
learn.fit(lr, n_epochs)
learn.save("model")


if __name__ == '__main__':

    print('==============================')
    print('=======    Starting    =======')
    print('==============================')
    print()
    print(f"Running on GPU {gpu}")
    print(f"Sample Ratio {sample_ratio}")
    print(f"Reading train from {train_path}")
    print(f"Reading val from {val_path}")
    print(f"Reading test from {test_path}")
    print(f"Using embedding dropout {emb_drop}")
    print(f"Using hidden layer sizes {szs}")
    print(f"Using hidden layer dropout {drops}")
    print(f"Using cross depth {cross_depth}")
    print()





