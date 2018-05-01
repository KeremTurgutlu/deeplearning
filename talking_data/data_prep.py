import pandas as pd
import sys
import gc
from fe_funcs import *

train_dtype_dict={"ip":"int32",
      "app":"int16",
      "device":"int16",
      "os":"int16",
      "channel":"int16",
      "is_attributed":"int8"}

test_dtype_dict={"ip":"int32",
      "app":"int16",
      "device":"int16",
      "os":"int16",
      "channel":"int16",
      "click_id":"int32"}

train_path =  "../../../data/talking/train_sample.csv"
test_path = "../../../data/talking/test.csv"

train = pd.read_csv(train_path,
                    parse_dates=["click_time", "attributed_time"],
                    skiprows=range(1,18700000),
                    dtype=train_dtype_dict)

test = pd.read_csv(test_path,
                   parse_dates=["click_time"],
                   #skiprows=range(1,18700000),
                   dtype=test_dtype_dict)

common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
merged = pd.concat([train[common_cols], test[common_cols]])

train_len = len(train)
is_attributed = train.is_attributed
del train
del test
gc.collect()

# ADD DAY HOUR
def add_dayhour(data):
    # extract time information
    data['click_time'] = pd.to_datetime(data['click_time'])
    data['click_timeHour'] = data.click_time.dt.hour.astype("int8")
    data['click_timeDay'] = data.click_time.dt.day.astype("int8")
    data.drop('click_time', 1, inplace=True)
    return data
merged = add_dayhour(merged)

# NORMALIZE CATS
cats = ['ip', 'app', 'device', 'os', 'channel',
'click_timeDay', 'click_timeHour']

def norm_cats(data, cats, dtype="int16"):
    for c in cats:
        cat2emb = {v:k for k, v in enumerate(data[c].unique())}
        data[c] = data[c].map(cat2emb).astype(dtype)
    return data

merged = norm_cats(merged, cats)
test = merged[train_len:].reset_index(drop=True)
rest = merged[:train_len].reset_index(drop=True)
del merged
gc.collect()

# CREATE TRAIN-VAL
train_lower_limits = (8, 4) # DAY 8 HOUR 04
train_upper_limits = (9, 3) # DAY 9 HOUR 03
val_lower_limits = (9, 4) # DAY 9 HOUR 04
val_upper_limits = (9, 15) # DAY 9 HOUR 15

train_msk = (((rest.click_timeDay >= train_lower_limits[0])
             & (rest.click_timeHour >= train_lower_limits[1]))
&
((rest.click_timeDay <= train_upper_limits[0])
 & (rest.click_timeHour <= train_upper_limits[1])))

val_msk = (((rest.click_timeDay >= val_lower_limits[0]) &
           (rest.click_timeHour >= val_lower_limits[1]))
&
((rest.click_timeDay <= val_upper_limits[0]) &
 (rest.click_timeHour <= val_upper_limits[1])))

# get test, train, val
train = rest[train_msk]
val = rest[val_msk]
del rest
gc.collect()

train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
train_val = pd.concat([train, val]).reset_index(drop=True)


####################################
###### FEATURE ENGINEERING    ######
####################################

# may add more random columns or column combinations
random_encoding_cols = \
    ['ip',
     'app',
     'device',
     'os',
     'channel',
     'click_timeHour',
     'click_timeDay',
     ['ip', 'click_timeDay', 'click_timeHour'],
     ['ip', 'app'],
     ['ip', 'app', 'os'],
     ['ip', 'app', 'click_timeHour']]

for c in random_encoding_cols:
    if isinstance(c, list):
        name = "_".join(c)
    else:
        name = c
    # regularized mean encoding for train
    train = reg_mean_encoding(train,
                              c,
                              f'random_mean_encode_{name}',
                              'is_attributed')
    # print("done1")
    # regularized mean encoding fo validation
    val = reg_mean_encoding_test(val,
                                 train,
                                 c,
                                 f'random_mean_encode_{name}',
                                 'is_attributed')
    # print("done2")
    # regularized mean encoding fo test
    test = reg_mean_encoding_test(test,
                                  train_val,
                                  c,
                                  f'random_mean_encode_{name}',
                                  'is_attributed')
    # print("done3")
    # encodings for full train: train + val
    train_val = reg_mean_encoding(train_val,
                                  c,
                                  f'random_mean_encode_{name}',
                                  'is_attributed')
    # print("done4")

for c in random_encoding_cols:
    if isinstance(c, list):
        name = "_".join(c)
    else:
        name = c
    # regularized count encoding for train
    train = reg_count_encoding(train,
                               c,
                               f'random_count_encode_{name}',
                               'is_attributed')
    # print("done")
    # regularized count encoding for validation
    val = reg_count_encoding_test(val,
                                  train,
                                  c,
                                  f'random_count_encode_{name}',
                                  'is_attributed')
    # print("done")count encoding for test
    test = reg_count_encoding_test(test,
                                   train_val,
                                   c,
                                   f'random_count_encode_{name}',
                                   'is_attributed')

    # encodings for full train: train + val
    train_val = reg_count_encoding(train_val,
                                   c,
                                   f'random_count_encode_{name}',
                                   'is_attributed')
    # print("done")


# SAVE DATA FOR MODELING
dst = "../../../data/talking/"
train.to_feather(dst+"train_prepd.feather")
val.to_feather(dst+"val_prepd.feather")
train_val.to_feather(dst+"train_val_prepd.feather")
test.to_feather(dst+"test_prepd.feather")











