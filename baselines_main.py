#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:50:20 2019

@author: naeemnowrouzi
"""

# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import pandas as pd
import numpy as np
import argparse
import baselines
import evaluation

PATH_TO_TRAIN ='./yoochoose-data/rsc15_reduced_train_2.txt'
PATH_TO_TEST = './yoochoose-data/rsc15_reduced_test_2.txt'
PATH_TO_PROCESSED_DATA = './yoochoose-data/'

train_data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
train_data.shape
test_data = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})
test_data.shape

# RandomPred (Random Predictor)
rand = baselines.RandomPred()
rand.fit(train_data)
res0 = evaluation.evaluate_sessions(pr=rand, test_data=test_data, train_data=train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time')
print('Recall@{}: {}\tMRR@{}: {}'.format(20, res0[0], 20, res0[1])) 


# POP (Popularity Predictor)
pop = baselines.Pop(top_n = 100, item_key = 'ItemId', support_by_key = None)
pop.fit(train_data)
res1 = evaluation.evaluate_sessions(pr=pop, test_data=test_data, train_data=train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time')
print('Recall@{}: {}\tMRR@{}: {}'.format(20, res1[0], 20, res1[1])) 


# SessionPOP (Session Popularity Predictor)
sesspop = baselines.SessionPop(top_n = 100, item_key = 'ItemId', support_by_key = None)
sesspop.fit(train_data)
res2 = evaluation.evaluate_sessions(pr=sesspop, test_data=test_data, train_data=train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time')
print('Recall@{}: {}\tMRR@{}: {}'.format(20, res2[0], 20, res2[1])) 


# ItemKNN
itemknn = baselines.ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')
itemknn.fit(train_data)
res3 = evaluation.evaluate_sessions(pr=itemknn, test_data=test_data, train_data=train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time')
print('Recall@{}: {}\tMRR@{}: {}'.format(20, res3[0], 20, res3[1])) 


# BPR (Bayesian Personalized Ranking Matrix Factorization)
bpr = baselines.BPR(n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId')
bpr.fit(train_data)
res4 = evaluation.evaluate_sessions(pr=bpr, test_data=test_data, train_data=train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time')
print('Recall@{}: {}\tMRR@{}: {}'.format(20, res4[0], 20, res4[1])) 



























 gru = model_RBP.GRU4Rec(sess, args)
        if args.is_training:
            gru.fit(data)
        else:
            #res = evaluation.evaluate_sessions_batch(gru, data, valid)
            #print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))            
            for c in [3,15,20]:
                res = evaluation.evaluate_sessions_batch(gru, data, valid, cut_off=c) 
                print('Recall@{}: {}\tMRR@{}: {}'.format(c, res[0], c, res[1])) 
            ### Export ratings
            # preds = res[2]
            # preds.to_csv(PATH_TO_PROCESSED_DATA + 'eikon_pred_test.csv', sep=',', index=False)
            #ids = res[3]
            #ids.to_csv(PATH_TO_PROCESSED_DATA + 'eikon_pred_test_ids.csv', sep=',', index=False)