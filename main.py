# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

import model
import evaluation

#PATH_TO_TRAIN = '/PATH/TO/rsc15_train_full.txt'
#PATH_TO_TEST = '/PATH/TO/rsc15_test.txt'

#PATH_TO_TRAIN ='/home/iordan.slavov/MyPython/packages/GRU4Rec_TF/data/eikon_train_full.txt'
#PATH_TO_TEST = '/home/iordan.slavov/MyPython/packages/GRU4Rec_TF/data/eikon_test.txt'
#PATH_TO_PROCESSED_DATA = '/home/iordan.slavov/MyPython/packages/GRU4Rec_TF/data/'

PATH_TO_TRAIN ='./yoochoose-data/rsc15_reduced_train_2.txt'
PATH_TO_TEST = './yoochoose-data/rsc15_reduced_test_2.txt'
PATH_TO_PROCESSED_DATA = './yoochoose-data/'

class Args():
    #is_training = 1
    layers = 1
    rnn_size = 100
    n_epochs = 2
    batch_size = 50
    dropout_p_hidden = 1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1.5e4
    sigma = 0
    init_as_normal = False
    reset_after_session = False
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 1.
    test_model = 2
    checkpoint_dir = '/Users/naeemnowrouzi/Desktop/Thesis/checkpoint_4'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=0, type=int)
    parser.add_argument('--test', default=1, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)        ## Dropout rate (default) = 0.5
    
    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})
    
    # Read parameters from command line
    args = Args()
    args.n_items = len(data['ItemId'].unique())
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    print(args.dropout_p_hidden)
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    
    ## GPU configurations could be updated as in Zoltan's code
    #gpu_config = tf.ConfigProto()
    #gpu_config.gpu_options.allow_growth = True
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        gru = model.GRU4Rec(sess, args)
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