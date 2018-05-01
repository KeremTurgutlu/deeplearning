"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""

import pandas as pd
import numpy as np
import gc
import os


def DO(frm, to, fileno):
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
    }

    print('loading train data...', frm, to)
    train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1, frm), nrows=to - frm,
                           dtype=dtypes,
                           usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    if debug:
        test_df = pd.read_csv("../input/test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], dtype=dtypes,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df = train_df.append(test_df)

    del test_df
    gc.collect()

    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

    gc.collect()

    naddfeat = 9
    for i in range(0, naddfeat):
        if i == 0: selcols = ['ip', 'channel']; QQ = 4;
        if i == 1: selcols = ['ip', 'device', 'os', 'app']; QQ = 5;
        if i == 2: selcols = ['ip', 'day', 'hour']; QQ = 4;
        if i == 3: selcols = ['ip', 'app']; QQ = 4;
        if i == 4: selcols = ['ip', 'app', 'os']; QQ = 4;
        if i == 5: selcols = ['ip', 'device']; QQ = 4;
        if i == 6: selcols = ['app', 'channel']; QQ = 4;
        if i == 7: selcols = ['ip', 'os']; QQ = 5;
        if i == 8: selcols = ['ip', 'device', 'os', 'app']; QQ = 4;
        print('selcols', selcols, 'QQ', QQ)

        filename = 'X%d_%d_%d.csv' % (i, frm, to)

        if os.path.exists(filename):
            if QQ == 5:
                gp = pd.read_csv(filename, header=None)
                train_df['X' + str(i)] = gp
            else:
                gp = pd.read_csv(filename)
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
        else:
            if QQ == 0:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].count().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 1:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].mean().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 2:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].var().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 3:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].skew().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 4:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].nunique().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 5:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[selcols[len(selcols) - 1]].cumcount()
                train_df['X' + str(i)] = gp.values

            if not debug:
                gp.to_csv(filename, index=False)

        del gp
        gc.collect()

    print('doing nextClick')
    predictors = []

    new_feature = 'nextClick'
    filename = 'nextClick_%d_%d.csv' % (frm, to)

    if os.path.exists(filename):
        print('loading from save file')
        QQ = pd.read_csv(filename).values
    else:
        D = 2 ** 26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df[
            'device'].astype(str) \
                                + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer = np.full(D, 3000000000, dtype=np.uint32)

        train_df['epochtime'] = train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks = []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category] - t)
            click_buffer[category] = t
        del (click_buffer)
        QQ = list(reversed(next_clicks))

        if not debug:
            print('saving')
            pd.DataFrame(QQ).to_csv(filename, index=False)

    train_df[new_feature] = QQ
    predictors.append(new_feature)

    train_df[new_feature + '_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature + '_shift')

    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip', 'day', 'hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(
        index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip', 'app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'channel'])[
        ['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_df = train_df.merge(gp, on=['ip', 'day', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = train_df[['ip', 'app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(
        index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = train_df[['ip', 'app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[
        ['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = train_df[['ip', 'app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])[
        ['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    train_df = train_df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    target = 'is_attributed'
    predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'day',
                       'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                       'ip_app_os_count', 'ip_app_os_var',
                       'ip_app_channel_var_day', 'ip_app_channel_mean_hour'])
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    for i in range(0, naddfeat):
        predictors.append('X' + str(i))

    print('predictors', predictors)

    test_df = train_df[len_train:]
    # val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:len_train]

    # print("train size: ", len(train_df))
    # print("valid size: ", len(val_df))
    # print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    train_df.to_csv("train_baris.csv")
    test_df.to_csv("test_baris.csv")



debug = 0
nrows = 184903891 - 1
if debug:
    val_size = 10000
    frm = 0
    nchunk = 100000
else:
    val_size = 2500000
    # nchunk = 40000000
    nchunk = 154903890
    frm = nrows - nchunk

to = frm + nchunk
to = frm + nchunk

sub = DO(frm, to, 0)