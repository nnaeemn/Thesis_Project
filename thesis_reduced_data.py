#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:33:35 2018

@author: naeemnowrouzi
"""
import os

import numpy as np
import pandas as pd
import datetime as dt

# PATH_TO_ORIGINAL_DATA = '/path/to/clicks/dat/file/'
# PATH_TO_PROCESSED_DATA = '/path/to/store/processed/data/'

PATH_TO_ORIGINAL_DATA ='./yoochoose-data/'
PATH_TO_PROCESSED_DATA = PATH_TO_ORIGINAL_DATA

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'rsc15_reduced_train_2.txt', sep='\t', dtype={'ItemId': np.int64})
#test_data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'yoochoose-test.dat', sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
len(data['ItemId'].unique())
test_data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'rsc15_test.txt', sep='\t', dtype={'ItemId': np.int64})
test_data.shape

#data.columns = ['SessionId', 'TimeStr', 'ItemId']
#data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
#del(data['TimeStr'])
#data.head(20)

#session_lengths = data.groupby('SessionId').size()
#session_lengths.shape

#data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
#data.head(20)
#item_supports = data.groupby('ItemId').size()
#item_supports.head(20)
#data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]

#session_lengths = data.groupby('SessionId').size()
#data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]

#data.groupby('ItemId').sum()
offset_sessions = np.zeros(1000, dtype=np.int32)
offset_sessions[1:] = data.groupby('SessionId').size().cumsum()[0:999]
((-offset_sessions[0:999]+offset_sessions[1:1000])-(data.groupby('SessionId').size().cumsum()[0:999])).sum()
data.sort_values(['SessionId', 'Time'], inplace=True)
data.sort_values('ItemId', inplace=True)
data['ItemId'].unique()[900:925]
data.ItemId
#data.head(20)
tmax0 = data.Time.max()
#tmin = data.Time.min()
#(tmax-tmin)/86400
session_max_times0 = data.groupby('SessionId').Time.max()
session_reduced = session_max_times0[session_max_times0 < tmax0-150*86400].index
data_reduced = data[np.in1d(data.SessionId, session_reduced)]
data_reduced.shape
#data.shape

tmax1 = data_reduced.Time.max()
session_max_times1 = data_reduced.groupby('SessionId').Time.max()
session_train = session_max_times1[session_max_times1 < tmax1-86400/2].index
session_test = session_max_times1[session_max_times1 >= tmax1-86400/2].index

train = data_reduced[np.in1d(data_reduced.SessionId, session_train)]
test = data_reduced[np.in1d(data_reduced.SessionId, session_test)]

test.ItemId.nunique()
train.ItemId.nunique()
len(data['ItemId'])
test = test[np.in1d(test.ItemId, train.ItemId)]
test.shape
tslength = test.groupby('SessionId').size()
test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
test.shape

# Write filtered data
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_reduced_train_2.txt', sep='\t', index=False)

print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_reduced_test_2.txt', sep='\t', index=False)

# Now split TRAIN into: TRAIN and VALID
tmax = train.Time.max()  # maximum recorded timestamp in the data
session_max_times = train.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < tmax-86400].index   # 86400s = 1 day
session_valid = session_max_times[session_max_times >= tmax-86400].index  # Select only the last day from a SessionID

train_tr = train[np.in1d(train.SessionId, session_train)]
valid = train[np.in1d(train.SessionId, session_valid)]
valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
tslength = valid.groupby('SessionId').size()
valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
train_tr.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_tr.txt', sep='\t', index=False)
print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
valid.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_valid.txt', sep='\t', index=False)