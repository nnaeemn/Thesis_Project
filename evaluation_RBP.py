# -*- coding: utf-8 -*-
"""
Created on Feb 27 2017
Author: Weiping Song
"""
import numpy as np
import pandas as pd


PATH_TO_PROCESSED_DATA = './yoochoose-data/'


def evaluate_sessions_batch(model_RBP, train_data, test_data, cut_off=20, batch_size=50, session_key='SessionId', item_key='ItemId', time_key='Time'):
    

    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
    
    Parameters
    --------
    model : A trained GRU4Rec model.
    
    train_data : It contains the transactions of the train set. In evaluation phase, this is used to build item-to-id map.
    test_data  : It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the 
    timestamp of the events (unix timestamps).
    
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
        
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. 
        Default value is 100.
        
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)f
   
    '''
    model_RBP.predict = False
    
    # Build itemidmap from TRAIN data!
    itemids = train_data[item_key].unique()
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
    
    #test_data.sort([session_key, time_key], inplace=True)
    test_data.sort_values([session_key, time_key], inplace=True)
    
    # Set a 1-dim array for event count in each test session. 
    # Then filled it - these values are like pointers, they indicate beginning and end of the session events 
    offset_sessions = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    
    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0
    # Set number of "iters" - they are set to session lentgh for sessions shorter than 50 
    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
    
    iters = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()
    
    start = offset_sessions[iters]
    end = offset_sessions[iters+1]
    
    in_idx = np.zeros(batch_size, dtype=np.int32)     # initialized to 0, positions in data will be placed here
    np.random.seed(42)
    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        minlen = (end[valid_mask]-start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]
        
        #<IS>: for i in xrange(minlen-1):
        #<IS>: Add a counter of runs
        numruns = 0
        for i in range(minlen-1):
            out_idx = test_data[item_key].values[start_valid+i+1]  #<IS>: "i" is the time step (n_epoch) -> moves "t" inside a batch
            
            ### <IS>:
            ### Main call to model - creates prediction
            preds = model_RBP.predict_next_batch(iters, in_idx, itemidmap, batch_size)
            preds.fillna(0, inplace=True)
            
            in_idx[valid_mask] = out_idx
            
            # <IS>: Calculate fit stats at current time step (over batches)
            ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
            rank_ok = ranks < cut_off
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evalutation_point_count += len(ranks)
            # <IS>:
            numruns = numruns + 1
        
        # <IS> iterations
        #iters2 = iters > -1 
        #iters2 = iters2.sum()
        #print('Iterations: {}\t minlen: {}'.format(iters2, minlen))
        
        start = start+minlen-1
        mask = np.arange(len(iters))[(valid_mask) & (end-start<=1)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions)-1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter+1]
    
        # export ratings (from 1st run, not the last as it was before)
        if numruns==1:
            preds.to_csv(PATH_TO_PROCESSED_DATA + 'preds_test.csv', sep=',', index=False)
                       
    print('Number of runs: {}'.format(numruns))
        
    return recall/evalutation_point_count, mrr/evalutation_point_count, preds, in_idx

### From Hidasi
def evaluate_sessions(pr, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'):    
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation. Breaks up ties.
    
    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the 
        timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. 
        Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
    
    '''
    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    evalutation_point_count = 0
    prev_iid, prev_sid = -1, -1
    mrr, recall = 0.0, 0.0
    
    for i in range(len(test_data)):
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        if prev_sid != sid:
            prev_sid = sid
        else:
            if items is not None:
                if np.in1d(iid, items): items_to_predict = items
                else: items_to_predict = np.hstack(([iid], items))      
            preds = pr.predict_next(sid, prev_iid, items_to_predict)
            preds[np.isnan(preds)] = 0
            preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties111
            rank = (preds > preds[iid]).sum()+1
            assert rank > 0
            if rank < cut_off:
                recall += 1
                mrr += 1.0/rank
            evalutation_point_count += 1
        prev_iid = iid
        
    return recall/evalutation_point_count, mrr/evalutation_point_count