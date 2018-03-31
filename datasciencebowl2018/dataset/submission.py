import numpy as np
import pandas as pd
from skimage.morphology import label

def get_test_sz(test_ds):
    """
    Getting original test images szs
    in same order they were predicted
    Input:
        test_ds (Dataset):test dataset
    Return:
        test_sz (list): list containing tuple for W, H
    """
    test_sz = []
    for fpath in test_ds.image_dirs:
        fname = fpath + 'images/' + fpath.split('/')[-2] + '.png'
        h, w = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).shape
        test_sz.append((w, h))
    return test_sz


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    """takes binary mask and yields for generator"""
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def get_submission_df(preds, test_ds, rle_func=prob_to_rles):
    """
    Takes resized preds and test dataset
    to return rle df
    Inputs:
        preds (list): list of np.arrays which has 2d binary mask predictions
        test_ds (Dataset): test dataset
        rle_func (function): function to encode each binary mask prediction with run length encoding
    Return:
        sub (pd.dataframe): pandas dataframe for submission
    """
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ds.image_dirs):
        id_ = id_.split('/')[-2]
        rle = list(rle_func(preds[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    return sub


def predict_and_submit(classifier, testloader):
    preds = classifier.predict(test_dl)

    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ds.image_dirs):
        id_ = id_.split('/')[-2]
        rle = list(prob_to_rles(preds[n][0, 0]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    return sub