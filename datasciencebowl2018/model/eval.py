import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F


def create_validation_dirs(main_path, data_path, train_ratio):
    """
    Creates train and valid folders from data in a data_path folder.
    Inputs:
        main_path (str): path to create train and valid folders
        data_path (str): path of the directory with data that will be copied
        train_ratio (float): ratio of training data to create
    """
    image_list = os.listdir(data_path)

    # remove previous train and valid dirs
    shutil.rmtree(main_path + 'train/', True)
    shutil.rmtree(main_path + 'valid/', True)
    # create new empty train and valid dirs
    os.makedirs(main_path + 'train', exist_ok=True)
    os.makedirs(main_path + 'valid', exist_ok=True)
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

    print(f"Copied {n} training and {len(image_list)-n} validation data")


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