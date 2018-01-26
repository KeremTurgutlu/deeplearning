### IMPORTS
print('IMPORT\n')
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import IPython.display as ipd
import librosa
import librosa.display
import pandas as pd


### CREATE TRAIN AND VAL DATA
print('CREATE TRAIN AND VAL DATA\n')
full_train_path = '../data/train/audio/'
val_list = open('../data/train/validation_list.txt').read().split('\n')
train_path = '../data/train/train'
val_path = '../data/train/val'

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

classes = os.listdir(full_train_path)

for c in classes:
    os.makedirs(train_path + '/' + c, exist_ok=True)
    os.makedirs(val_path + '/' + c, exist_ok=True)

main_train_path = '../data/train/audio/'
for r, d, f in os.walk(main_train_path):
    for file in f:
        fpath = os.path.join(r, file)
        cls = fpath.split('/')[4]
        cls_fname = '/'.join(fpath.split('/')[4:])
        # copy files to necessary directories
        # is validation
        if cls_fname in val_list:
            shutil.copy2(fpath, val_path + '/' + cls_fname)
        # is train
        else:
            shutil.copy2(fpath, train_path + '/' + cls_fname)

def class_counts(path, classes):
    total_count = 0
    d = {}
    for c in classes:
        i = len(os.listdir(path + c))
        print(c, ':', i)
        d[c] = i
        total_count += i
    print('\n')
    print(f'total counts: {total_count}')
    return d

print('Created training and validation data\n')
path = '../data/train/train/'
class_counts(path, classes)
path = '../data/train/val/'
class_counts(path, classes)

### GENERATE UNKNOWN AND SILENCE
print('Start generating unknown and silence clips\n')
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_background_noise_']
train_path = '../data/train/train/'
val_path = '../data/train/val/'

#create unknown directory for both train and val
os.makedirs(train_path + 'unknown/', exist_ok=True)
os.makedirs(val_path + 'unknown/', exist_ok=True)

def moveToUnknown(path):
    dirs = os.listdir(path)
    for c in dirs:
        # moving if not in labels list
        if c not in labels:
            fnames = os.listdir(path + c)
            for fname in fnames:
                src = path + c + '/' + fname
                dst = path + 'unknown' + '/' + c + '_' + fname
                shutil.move(src, dst)

moveToUnknown(train_path)
moveToUnknown(val_path)

### SILENCE
train_path = '../data/train/train/'
val_path = '../data/train/val/'

os.makedirs(train_path+'silence', exist_ok=True)
os.makedirs(val_path+'silence', exist_ok=True)

background_path = '../data/train/train/_background_noise_/'

background_fnames = [f for f in os.listdir(background_path) if 'wav' in f]

sample_rate = 16000
n_train = 2200
n_val = 220

for back_fname in background_fnames:
    sample_rate, samples = wavfile.read(background_path + back_fname)
    # split into train and val 80-20
    train_background_arr = samples[:int(len(samples) * 0.8)]
    val_background_arr = samples[int(len(samples) * 0.8):]
    train_len, val_len = train_background_arr.shape[0], val_background_arr.shape[0]

    # create clips for training n_train times
    for i in range(n_train):
        start_idx = np.random.choice(range(train_len - sample_rate))
        seq_idx = range(start_idx, start_idx + sample_rate)
        new_clip = train_background_arr[seq_idx]
        wavfile.write('../data/train/train/silence/' + back_fname + f'_{i}', sample_rate, new_clip)

    for i in range(n_val):
        start_idx = np.random.choice(range(val_len - sample_rate))
        seq_idx = range(start_idx, start_idx + sample_rate)
        new_clip = val_background_arr[seq_idx]
        wavfile.write('../data/train/val/silence/' + back_fname + f'_{i}', sample_rate, new_clip)



### GET FINAL DISTRIBUTION OF FILES IN TRAIN AND VALID
train_path = '../data/train/train/'
val_path = '../data/train/val/'
classes = os.listdir(train_path)

print('Datasets Preparation Done\n')
train_counts=class_counts(train_path, classes)
print('\n')
val_counts=class_counts(val_path, classes)
