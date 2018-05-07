from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

#########################
###   TARGET ENCODING ###
#########################

def kfold_mean_encoding(train, col, new_col, target, splits=5, seed=42, alpha=None, dtype="float32"):
    """ Computes regularize mean encoding.
        Use this to create mean encoding for training data

    Inputs:
        train: training dataframe
        col: a single column as string or list of columns to groupby
        during mean target encoding
        new_col: name of new created column
        splits: splits to use for cv
        alpha: smoothing parameter, increase to regularize fewer groups
    Returns:
        train: dataframe with new column added
    """
    # single column to groupby
    train[new_col] = 0
    if isinstance(col, str):
        for split, (trn_idx, val_idx) in enumerate(KFold(splits, shuffle=True, random_state=seed).split(train)):

            groups = train.iloc[trn_idx].groupby(col)[target].mean()
            train.loc[val_idx, new_col] = train.loc[val_idx, col].map(groups)

    # multiple columns to groupby
    elif isinstance(col, list):
        for split, (trn_idx, val_idx) in enumerate(KFold(splits, shuffle=True, random_state=seed).split(train)):
            stats = train.iloc[trn_idx].groupby(col)[target].mean().reset_index()
            vals = pd.merge(train.iloc[val_idx], stats, "left", on=col, suffixes=["_", ""])[target]
            vals.index = val_idx
            train.loc[val_idx, new_col] = vals

    train[new_col].fillna(train[new_col].mean(), inplace=True)
    train[new_col] = train[new_col].astype(dtype)
    
    if alpha is not None:
        """smooths calculated mean encoding"""
        global_mean = train[target].mean()
        nrows = train.groupby(col)[target].transform('count')
        train[new_col] = (train[new_col]*nrows + global_mean*alpha) / (nrows + alpha) 
    
    
    return train

def loo_mean_encoding(train, col, new_col, target, alpha=None, dtype="float32"):
    """ Computes regularize mean encoding.
        Use this to create mean encoding for training data

    Inputs:
        train: training dataframe
        col: a single column as string or list of columns to groupby
        during mean target encoding
        new_col: name of new created column
        target: target column
        alpha: smoothing parameter, increase to regularize fewer groups
    Returns:
        train: dataframe with new column added
    """
    # single column to groupby
    train[new_col] = (train.groupby(col)[target].transform('sum') - train[target]) / \
                     (train.groupby(col)[target].transform('count') - 1)
    train[new_col].fillna(train[new_col].mean(), inplace=True)
    train[new_col] = train[new_col].astype(dtype)
    
    if alpha is not None:
        """smooths calculated mean encoding"""
        global_mean = train[target].mean()
        nrows = train.groupby(col)[target].transform('count')
        train[new_col] = (train[new_col]*nrows + global_mean*alpha) / (nrows + alpha) 
    return train

def expanding_mean_encoding(train, col, new_col, target, dtype="float32"):
    """ Computes regularize mean encoding.
        Use this to create mean encoding for training data.
        Do with different permutations of data and average results.

    Inputs:
        train: training dataframe
        col: a single column as string or list of columns to groupby
        during mean target encoding
        new_col: name of new created column
        target: target column
    Returns:
        train: dataframe with new column added
    """
    cumsum = train.groupby(col)[target].cumsum() - train[target]
    cumcount = train.groupby(col)[target].cumcount()
    train[new_col] = (cumsum/cumcount)
    train[new_col].fillna(train[new_col].mean(), inplace=True)
    train[new_col] = train[new_col].astype(dtype)
    return train


def mean_encoding_test(test, train, col, new_col, target, dtype="float32"):
    """ Computes target enconding for test data.
        Use this to create mean encoding for valdiation and test data
        Inputs:
            train: training dataframe to compute means
            test: training dataframe to create new column
            col: a single column as string or list of columns
            new_col: name of new created column
        Returns:
            test: dataframe with new column added
    This is similar to how we do validation
    """
    # single column to groupby
    test[new_col] = 0
    if isinstance(col, str):
        test[new_col] = test[col].map(train.groupby(col)[target].mean())
        test[new_col].fillna(train[target].mean(), inplace=True)
    # multiple columns to groupby
    elif isinstance(col, list):
        stats = train.groupby(col)[target].mean().reset_index()
        vals = pd.merge(test, stats, "left", on=col, suffixes=["_", ""])[target]
        test[new_col] = vals

    test[new_col].fillna(train[target].mean(), inplace=True)
    test[new_col] = test[new_col].astype(dtype)
    return test


#########################
# NOT TARGET ENCODING ###
#########################

def kfold_count_encoding(train, col, new_col, target, splits=5, seed=42, dtype="int16"):
    """ Computes regularize count encoding.
        Use this to create count encoding for training data

    Inputs:
        train: training dataframe
        col: a single column as string or list of columns to groupby
        during count target encoding
        new_col: name of new created column
        splits: splits to use for cv
    Returns:
        train: dataframe with new column added
    """
    # single column to groupby
    train[new_col] = 0
    if isinstance(col, str):
        for split, (trn_idx, val_idx) in enumerate(KFold(splits, shuffle=True, random_state=seed).split(train)):
            groups = train.iloc[trn_idx].groupby(col)[target].count()
            train.loc[val_idx, new_col] = train.loc[val_idx, col].map(groups)

    # multiple columns to groupby
    elif isinstance(col, list):
        for split, (trn_idx, val_idx) in enumerate(KFold(splits, shuffle=True, random_state=seed).split(train)):
            stats = train.iloc[trn_idx].groupby(col)[target].count().reset_index()
            vals = pd.merge(train.iloc[val_idx], stats, "left", on=col, suffixes=["_", ""])[target]
            vals.index = val_idx
            train.loc[val_idx, new_col] = vals

    train[new_col].fillna(train[new_col].mean(), inplace=True)
    train[new_col] = train[new_col].astype(dtype)
    return train


def count_encoding_test(test, train, col, new_col, target, dtype="int16"):
    """ Computes target enconding for test data.
        Use this to create count encoding for valdiation and test data
        Inputs:
            train: training dataframe to compute counts
            test: training dataframe to create new column
            col: a single column as string or list of columns
            new_col: name of new created column
        Returns:
            test: dataframe with new column added
    This is similar to how we do validation
    """
    # single column to groupby
    test[new_col] = 0
    if isinstance(col, str):
        test[new_col] = test[col].map(train.groupby(col)[target].count())
        test[new_col].fillna(train[target].count(), inplace=True)
    # multiple columns to groupby
    elif isinstance(col, list):
        stats = train.groupby(col)[target].count().reset_index()
        vals = pd.merge(test, stats, "left", on=col, suffixes=["_", ""])[target]
        test[new_col] = vals

    test[new_col].fillna(train[target].mean(), inplace=True)
    test[new_col] = test[new_col].astype(dtype)
    return test


##########################
# FEATURE INTERACTIONS ###
##########################


def create_feat_interaction(data, columns):
    """
    Create Featuer Interactions Column
    
    data (pd.DataFrame): Input dataframe to create the new column
    columns (list): column names in list to combine
    """
    
    new_column = '_'.join(columns)
    data[new_column] = ""

    for c in columns:
        data[new_column] += data[c].astype(str) + "_"
    
    val2idx = {val:i for i,val in enumerate(data[new_column].unique())}
    data[new_column] = data[new_column].map(val2idx)
    return data


##########################
####  TIME COMPONENT   ###
##########################


def time_since_flag(data, col, time, target, flag):
    """
    Find time since an event happening for a 
    group. Ex. time since a customer clicked an ad.
        
    Inputs:
        data (pd.DataFrame): main data
        col (str): single column to use as group values
        time (str): time column to calculate seconds diff
        target (str): target column to look for an event
        flag (...): flag to check inside target column - event flag
    Returns:
        data (pd.DataFrame): data with the new column
    """
    new_col = []

    # update this every iterations
    prev_col_val = None
    # update this if a desired event happens
    prev_time = None
    # sort data 
    data = data.sort_values([col, time], ascending=True)

    for idx, row in data[[col, time, target]].iterrows():
        col_val = row[col]
        col_time = row[time]
        col_target = row[target]

        # new group
        if col_val != prev_col_val:
            prev_time = None
            feature = np.nan
        # same group
        else:    
            # desired event happened in the past
            if prev_time is not None:
                feature = (col_time - prev_time).seconds
            # desired event didn't happened in the past
            else:
                feature = np.nan

        prev_col_val = col_val
        # update prev time if col target is flag
        if col_target == flag: prev_time = col_time
        new_col.append(feature)
    
    data[f"{col}_time_since_{target}_{flag}"] = new_col
    return data
