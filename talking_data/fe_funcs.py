from sklearn.model_selection import KFold
import pandas as pd

def reg_mean_encoding(train, col, new_col, target, splits=10, seed=42, dtype="float32"):
    """ Computes regularize mean encoding.
        Use this to create mean encoding for training data

    Inputs:
        train: training dataframe
        col: a single column as string or list of columns to groupby
        during mean target encoding
        new_col: name of new created column
        splits: splits to use for cv
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
    return train


def reg_mean_encoding_test(test, train, col, new_col, target, dtype="float32"):
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


def reg_count_encoding(train, col, new_col, target, splits=10, seed=42, dtype="int16"):
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

    train[new_col].fillna(train[new_col].count(), inplace=True)
    train[new_col] = train[new_col].astype(dtype)
    return train


def reg_count_encoding_test(test, train, col, new_col, target, dtype="int16"):
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

    test[new_col].fillna(train[target].count(), inplace=True)
    test[new_col] = test[new_col].astype(dtype)
    return test
