# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import os
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell
import pandas as pd
import numpy as np
import RBP
import torch

class GRU4Rec:

    def __init__(self, sess, args):
        self.sess = sess
        self.is_training = args.is_training

        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.sigma = args.sigma
        self.init_as_normal = args.init_as_normal
        self.reset_after_session = args.reset_after_session
        self.session_key = args.session_key
        self.item_key = args.item_key
        self.time_key = args.time_key
        self.grad_cap = args.grad_cap
        self.n_items = args.n_items
        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        self.checkpoint_dir = args.checkpoint_dir
        print(self.checkpoint_dir)
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Checkpoint Dir not found")

        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if self.is_training:
            return

        ## use self.predict_state to hold hidden states during prediction - 3D tensor, 3rd dim is the layers
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            #saver = tf.train.import_meta_graph('{}/gru-model-{}.ckpt.meta'.format(self.checkpoint_dir, args.test_model))
            self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, args.test_model))  #<IS>: Model to use!!!
#tf.train.latest_checkpoint

    ########################ACTIVATION FUNCTIONS#########################
    def linear(self, X):
        return X
    def tanh(self, X):
        return tf.nn.tanh(X)
    def softmax(self, X):
        return tf.nn.softmax(X)
    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))
    def relu(self, X):
        return tf.nn.relu(X)
    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    ############################LOSS FUNCTIONS######################
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat)+1e-24))
    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat)-yhatT)))
    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.batch_size
        return tf.reduce_mean(term1 - term2)
    
    ##################################################################
    def build_model(self):

        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        ## <IS>: Num of layers is used here
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in range(self.layers)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):  
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)  
            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
            #softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))
            
            cell = GRUCell(self.rnn_size, activation=self.hidden_act)
            drop_cell = DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)
            stacked_cell = MultiRNNCell([drop_cell] * self.layers)   ##<IS>: Num of layers is used here, too!

            inputs = tf.nn.embedding_lookup(embedding, self.X)
            output, state = stacked_cell(inputs, tuple(self.state))
            self.final_state = state
        if self.is_training:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            sampled_W = tf.nn.embedding_lookup(embedding, self.Y) ####(UNIFIED EMBEDDING)CHHANGED TO EMBEDDING FROM SOFTMAX_W
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)
        else:
            logits = tf.matmul(output, embedding, transpose_b=True) + softmax_b ####(UNIFIED EMBEDDING)CHHANGED TO EMBEDDING FROM SOFTMAX_W
            self.yhat = self.final_activation(logits)

        if not self.is_training:
            return

        self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))

#        '''
 #       Try different optimizers.
  #      '''
   #     #optimizer = tf.train.AdagradOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
     #   #optimizer = tf.train.AdadeltaOptimizer(self.lr)
      #  #optimizer = tf.train.RMSPropOptimizer(self.lr)

        
       # gvs = optimizer.compute_gradients(self.cost, tvars)
       # if self.grad_cap > 0:
       #     capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
       # else:
       #     capped_gvs = gvs
       # self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
       
       
  # Implementing RBP
  
        params_dynamic = tf.trainable_variables()
        #print(params_dynamic)
        last_state = tf.contrib.graph_editor.detach(self.final_state)

        grad_state_last = torch.autograd.grad(self.cost, last_state, retain_graph=True)
        #print(grad_state_last)
        state_2nd_last = state[49]
       #update_forward_diff = ?
        gvs = RBP(params_dynamic,
                  self.final_state,
                  state_2nd_last,
                  grad_state_last,
                  update_forward_diff=None,
                  eta=1.0e-5,
                  truncate_iter=50,
                  rbp_method='Neumann_RBP')
        
        self.train_op = optimizer.apply_gradients(gvs, global_step=self.global_step)

    
    
    def init(self, data):
        #data.sort([self.session_key, self.time_key], inplace=True)
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return offset_sessions
    
    def fit(self, data):
        self.error_during_train = False
        itemids = data[self.item_key].unique() # Not sorted
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)# Each item assigned an integer, randomly it would seem. 

        ### <IS>: Merge "data" with the "itemidmap" - has the decoding here
        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':self.itemidmap[itemids].values}), on=self.item_key, how='inner')

        offset_sessions = self.init(data)

        print('fitting model...')

        for epoch in range(self.n_epochs):
            epoch_cost = []
            # <IS>:state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in xrange(self.layers)]
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
            session_idx_arr = np.arange(len(offset_sessions)-1)
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
        
            start = offset_sessions[session_idx_arr[iters]]
            end   = offset_sessions[session_idx_arr[iters]+1]
            finished = False
            
# First event of each session. 
            while not finished:
                minlen = (end-start).min()
                out_idx = data.ItemIdx.values[start]#[0,4,...]# The first items/events
                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx = data.ItemIdx.values[start+i+1]

                    ###---------------  X, Y  set up --------------------###
                    # prepare inputs, targeted outputs and hidden states - list which tensors to return
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    # input/output
                    feed_dict = {self.X: in_idx, self.Y: out_idx}

                    #<IS>: self.layers
                    for j in range(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    # Execute graph, collect error (cost)
                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))
                start = start+minlen-1
                mask = np.arange(len(iters))[(end-start)<=1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions)-1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter]+1]
                if len(mask) and self.reset_after_session:
                    #<IS>: self.layers
                    for i in range(self.layers):
                        state[i][mask] = 0

            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            #<IS>: "epoch" numbers the checkpoint (model)
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)

    def predict_next_batch(self, session_ids, input_item_ids, itemidmap, batch=50):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e.
        events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the
        corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length MUST equal to the prediction batch size (batch param).

        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be in the training data of the network. Its length must
            equal to the prediction batch size (batch param).

        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. ROWS are indexed by the item IDs.
        '''
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1     #<IS>: [-1, -1, ...]
            self.predict = True

        session_change = np.arange(batch)[session_ids != self.current_session]  # <IS>: np.arange(3) --> array([0,1,2])
        if len(session_change) > 0: # change internal states with session changes
            #<IS>: self.layers
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session=session_ids.copy()

        in_idxs = itemidmap[input_item_ids]
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: in_idxs}

        #<IS>: self.layers
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
            
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T

        # export ratings
        # preds.to_csv(PATH_TO_PROCESSED_DATA + 'eikon_pred_test.csv', sep=',', index=False)
        return pd.DataFrame(data=preds, index=itemidmap.index)
