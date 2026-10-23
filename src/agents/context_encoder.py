from __future__ import annotations
import torch
from torch import nn

class TransitionHistoryEncoder(nn.Module):
    def __init__(self,input_dim:int,gru_hidden_dim:int=32,latent_dim:int=16):
        super().__init__(); self.gru=nn.GRU(input_dim,gru_hidden_dim,batch_first=True); self.proj=nn.Sequential(nn.Linear(gru_hidden_dim,latent_dim),nn.Tanh())
    def forward(self,h):
        if h.ndim!=3: raise ValueError('history must be [batch,time,feature]')
        _,hidden=self.gru(h); return self.proj(hidden[-1])

class HistoryDynamicsPredictor(nn.Module):
    def __init__(self,history_input_dim,current_input_dim,target_dim,gru_hidden_dim=32,latent_dim=16,predictor_hidden_dim=64):
        super().__init__(); self.encoder=TransitionHistoryEncoder(history_input_dim,gru_hidden_dim,latent_dim)
        self.head=nn.Sequential(nn.Linear(latent_dim+current_input_dim,predictor_hidden_dim),nn.ReLU(),nn.Linear(predictor_hidden_dim,predictor_hidden_dim),nn.ReLU(),nn.Linear(predictor_hidden_dim,target_dim))
    def forward(self,h,current):
        z=self.encoder(h); return self.head(torch.cat([z,current],dim=-1)),z

class NoHistoryDynamicsPredictor(nn.Module):
    def __init__(self,current_input_dim,target_dim,hidden_dim=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(current_input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,target_dim))
    def forward(self,current): return self.net(current)
