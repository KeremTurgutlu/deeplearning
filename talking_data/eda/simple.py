import operator

def sort_dict(d, reverse=True):
    return sorted(d.items(), key=operator.itemgetter(1), reverse=reverse)

def col_nunique(df):
    """get number of unique elements"""
    d = {c: df[c].nunique() for c in df.columns}
    return sort_dict(d)

def col_nas(df):
    """get number of NAs"""
    d = {c :df[c].isnull().sum() for c in df.columns}
    return sort_dict(d)