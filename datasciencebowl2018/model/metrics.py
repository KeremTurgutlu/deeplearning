import numpy as np
import torch.nn.functional as F

def dice_score(logits, targets, thresh):
    hard_preds = (F.sigmoid(logits) > thresh).float()
    m1 = hard_preds.view(-1)  # Flatten
    m2 = targets.view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection) / (m1.sum() + m2.sum() + 1e-6)

def dice_coeff(pred, target, smooth=0):
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth + 1e-10)

#######
# RLE #
######
def run_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b>prev+1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    #https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
        rle[-2] = rle[-2] -1  #print('xxx')

    rle = ' '.join([str(r) for r in rle])
    return rle

def run_length_decode(rle, H, W, fill_value=255):

    mask = np.zeros((H * W), np.uint8)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T # H, W need to swap as transposing.
    return mask



#https://www.kaggle.com/wcukierski/example-metric-implementation
def compute_precision(threshold, iou):
    matches = iou > threshold
    true_positives  = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

def print_precision(precision):

    print('thresh   prec    TP    FP    FN')
    print('---------------------------------')
    for (t, p, tp, fp, fn) in precision:
        print('%0.2f     %0.2f   %3d   %3d   %3d'%(t, p, tp, fp, fn))



def compute_average_precision_for_mask(predict, truth, t_range=np.arange(0.5, 1.0, 0.05)):

    num_truth   = len(np.unique(truth  ))
    num_predict = len(np.unique(predict))

    # Compute intersection between all objects
    intersection = np.histogram2d(truth.flatten(), predict.flatten(), bins=(num_truth, num_predict))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(truth,   bins = num_truth  )[0]
    area_pred = np.histogram(predict, bins = num_predict)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred,  0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    precision = []
    average_precision = 0
    for t in t_range:
        tp, fp, fn = compute_precision(t, iou)
        p = tp / (tp + fp + fn)
        precision.append((t, p, tp, fp, fn))
        average_precision += p

    average_precision /= len(precision)
    return average_precision, precision
