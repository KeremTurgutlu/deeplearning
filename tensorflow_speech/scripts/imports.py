# PYTHON
import os
import random
import librosa
from scipy.io import wavfile
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import confusion_matrix

#PYTORCH
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable as V
from torch import optim
from torch.utils.data.sampler import *