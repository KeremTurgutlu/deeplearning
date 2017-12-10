import numpy as np


##############################
# NORMALIZED GINI
##############################
def get_gini_data(actuals, preds):
    actual = [arr[0] for arr in actuals]
    pred = [np.exp(arr[1]) for arr in preds]
    return actual, pred


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    actual, pred = get_gini_data(actual, pred)
    return gini(actual, pred) / gini(actual, actual)