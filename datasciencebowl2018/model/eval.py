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
    return train_dirs, list(np.unique(valid_dirs))


def create_validation_dirs(main_path, data_path, train_ratio, train_dirs=None, valid_dirs=None, seed=None):
    """
    Creates train and valid folders from data in a data_path folder.
    Inputs:
        main_path (str): path to create train and valid folders
        data_path (str): path of the directory with data that will be copied
        train_ratio (float): ratio of training data to create
        train_dirs (list) : list of directory that have training set
        valid_dirs (list) : list of directory that have validation set
    """
    np.random.seed(seed)
    image_list = os.listdir(data_path)

    # remove previous train and valid dirs
    shutil.rmtree(main_path + 'train/', True)
    shutil.rmtree(main_path + 'valid/', True)
    # create new empty train and valid dirs
    os.makedirs(main_path + 'train', exist_ok=True)
    os.makedirs(main_path + 'valid', exist_ok=True)
    # if train and valid dirs are not given
    if train_dirs is None:
        # find n
        n = int(len(image_list) * train_ratio)
        # shuffle list inplace
        np.random.shuffle(image_list)
        # get training and valid data
        train_dirs = image_list[:n]
        valid_dirs = image_list[n:]
    # copy image dirs to train and valid
    for _dir in train_dirs:
        shutil.copytree(data_path + _dir, main_path + 'train/' + _dir)
    for _dir in valid_dirs:
        shutil.copytree(data_path+ _dir, main_path + 'valid/' + _dir)

    print(f"Copied {len(train_dirs)} training and {len(valid_dirs)} validation data")



def show_predictions(dataloader, classifier, threshold=0.5, n=None):
    print('\t\t Image \t\t\t\t\t Mask \t\t\t\t Predicted Mask')
    for i, (img, msk, _ )in enumerate(iter(dataloader)):
        classifier.net.eval()
        plt.figure(figsize=(20, 20))
        if torch.cuda.is_available():
            img = img.cuda()
        out = classifier.net(V(img))
        plt.subplot(1,3,1)
        plt.imshow(img.cpu().numpy()[0].transpose(1,2,0))
        plt.subplot(1,3,2)
        plt.imshow(msk.cpu().numpy()[0, 0])
        plt.subplot(1,3,3)
        plt.imshow((F.sigmoid(out).cpu().data.numpy()[0, 0] > threshold)*1)
        plt.show()
        i += 1
        if n:
            if n == i:
                break