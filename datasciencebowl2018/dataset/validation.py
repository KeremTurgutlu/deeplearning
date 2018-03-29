import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F

def get_stratified_valid_dirs(classes_df, seed=0):
    """
    Inputs:
        classes_df (dataframe) : Must have columns: filename, foreground, background, is_train
        seed (int) : random seed
    Return:
        train_dirs (list) : training sample dir names
        valid_dirs (list) : validation sample dir names
    """
    test_groups = classes_df[~classes_df.is_train].groupby(['foreground', 'background']).count().reset_index()
    np.random.seed(seed)
    valid_files = []
    train_classes = classes_df[classes_df.is_train]
    for _,row in test_groups.iterrows():
        fg = row['foreground']
        bg = row['background']
        count = row['is_train']
        # filter by group
        sub_data = train_classes[(train_classes.foreground == fg) & (train_classes.background == bg)]
        if len(sub_data) != 0:
            # sample randomly
            sample = list(np.random.choice(sub_data.filename, size=count, replace=False))
            valid_files += sample
        else:
            valid_files += list(np.random.choice(train_classes[~train_classes.filename.isin(valid_files)]['filename'], size=count, replace=False))

    valid_dirs = [f.split('.')[0] for f in valid_files]
    train_dirs = [f.split('.')[0] for f in list(train_classes[~train_classes.filename.isin(valid_files)]['filename'])]
    valid_dirs = list(np.unique(valid_dirs))
    return train_dirs, valid_dirs


def create_validation_dirs(dst, src, train_ratio, train_dirs=None, valid_dirs=None, seed=None):
    """
    Creates train and valid folders from data in a data_path folder. If train_dirs and
    valid_dirs is not provided sampling will be done randomly.
    Inputs:
        dst (str): destination path where 'train' and 'valid' folders will be created
        src (str): source path where full data is available
        train_ratio (float): ratio of training data to create
        train_dirs (list) : list of directory that have training set
        valid_dirs (list) : list of directory that have validation set
        seed (int): random seed
    """
    np.random.seed(seed)
    image_list = os.listdir(src)

    # remove previous train and valid dirs
    shutil.rmtree(dst + 'train/', True)
    shutil.rmtree(dst + 'valid/', True)

    # create new empty train and valid dirs
    os.makedirs(dst + 'train', exist_ok=True)
    os.makedirs(dst + 'valid', exist_ok=True)

    # if train and valid dirs are not given
    if (train_dirs is None) or (valid_dirs is None):
        # find n
        n = int(len(image_list) * train_ratio)
        # shuffle list inplace
        np.random.shuffle(image_list)
        # get training and valid data
        train_dirs = image_list[:n]
        valid_dirs = image_list[n:]
    # copy image dirs to train and valid
    for trn_dir in train_dirs:
        shutil.copytree(src + trn_dir, dst + 'train/' + trn_dir)
    for val_dir in valid_dirs:
        shutil.copytree(src + val_dir, dst + 'valid/' + val_dir)

    print(f"Copied {len(train_dirs)} training and {len(valid_dirs)} validation data")
