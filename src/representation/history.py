from __future__ import annotations
import numpy as np

def one_hot_action(action:int, action_dim:int)->np.ndarray:
    if action<0 or action>=action_dim: raise ValueError('action out of range')
    x=np.zeros(action_dim,dtype=np.float32); x[int(action)]=1.0; return x

def transition_feature(state, action:int, next_state, action_dim:int)->np.ndarray:
    s=np.asarray(state,dtype=np.float32); ns=np.asarray(next_state,dtype=np.float32)
    if s.shape!=ns.shape: raise ValueError('state shape mismatch')
    return np.concatenate([s,one_hot_action(action,action_dim),ns-s]).astype(np.float32)

def dynamics_transition_feature(state, action:int, next_state, action_dim:int)->np.ndarray:
    s=np.asarray(state,dtype=np.float32); ns=np.asarray(next_state,dtype=np.float32)
    if s.shape!=ns.shape: raise ValueError('state shape mismatch')
    return np.concatenate([one_hot_action(action,action_dim),ns-s]).astype(np.float32)

def current_feature(state, action:int, action_dim:int)->np.ndarray:
    s=np.asarray(state,dtype=np.float32)
    return np.concatenate([s,one_hot_action(action,action_dim)]).astype(np.float32)
