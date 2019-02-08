#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:03:49 2019

@author: naeemnowrouzi
"""
import torch
from CG import conjugate_gradient

def RBP(params_dynamic,
                final_state,
                state_2nd_last,
                grad_state_last,
                update_forward_diff=None,
                eta=1.0e-5,
                truncate_iter=50,
                rbp_method='Neumann_RBP'):
            
            assert rbp_method in ['Neumann_RBP', 'CG_RBP',
                        'RBP'], "Nonsupported RBP method {}".format(rbp_method)
            

            # gradient of loss w.r.t. dynamic system parameters
            if rbp_method == 'Neumann_RBP':
              neumann_g = None
              neumann_v = None
              neumann_g_prev = grad_state_last
              neumann_v_prev = grad_state_last
              
              for ii in range(truncate_iter):
                  neumann_v = torch.autograd.grad(
                          self.final_state,
                          state_2nd_last,
                          grad_outputs=neumann_v_prev,
                          retain_graph=True,
                          allow_unused=True)
                  neumann_g = [x + y for x, y in zip(neumann_g_prev, neumann_v)]
                  neumann_v_prev = neumann_v
                  neumann_g_prev = neumann_g

              z_star = neumann_g
            
            elif rbp_method == 'CG_RBP':
                # here A = I - J^T
              
              def _Ax_closure(x):
                  JTx = torch.autograd.grad(
                          state,
                          state_2nd_last,
                          grad_outputs=x,
                          retain_graph=True,
                          allow_unused=True)
                  Ax = [m - n for m, n in zip(x, JTx)]
                  JAx = update_forward_diff(Ax, state)
                  ATAx = [m - n + eta * p for m, n, p in zip(Ax, JAx, x)]
                  return ATAx
        
              Jb = update_forward_diff(grad_state_last, state)
              ATb = [m - n for m, n in zip(grad_state_last, Jb)]
              z_star = conjugate_gradient(_Ax_closure, ATb, max_iter=truncate_iter)
            
            elif rbp_method == 'RBP':
                z_T = [torch.zeros_like(pp).uniform_(0, 1) for pp in state]
                
                for ii in range(truncate_iter):
                  z_T = torch.autograd.grad(
                          state,
                          state_2nd_last,
                          grad_outputs=z_T,
                          retain_graph=True,
                          allow_unused=True)
                  z_T = [x + y for x, y in zip(z_T, grad_state_last)]
                  
                z_star = z_T
            
            return torch.autograd.grad(
                    state,
                    params_dynamic,
                    grad_outputs=z_star,
                    retain_graph=True,
                    allow_unused=True)