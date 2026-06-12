import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from collections import deque
from sentence_transformers import SentenceTransformer



class stateSpaceModel(nn.Module):
    def __init__(self,state_dim, input_dim, hidden_dim1,hidden_dim2, output_dim):
        super(stateSpaceModel, self).__init__()
        # F(x_{t-1},u_t)=>x_t (Transition model)
        self.Fxu = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, state_dim)
        )

        # G(x_t,u_t)=>z_t (Observation model)
        self.Gxu = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
        
    def forward(self, x_prev, u):
        # gets xt
        ux_prev = torch.cat([u,x_prev], dim=-1)
        x_curr = self.Fxu(ux_prev)

        # gets zt
        ux_curr = torch.cat([u,x_curr], dim=-1)
        zt = self.Gxu(ux_curr)
        return x_curr, zt
    

class ssm_feature_extractor:
    def __init__(self):
        self.state_dim = 768    # x_t
        self.input_dim = 768    # u_t
        self.output_dim = 768   # z_t
        self.hidden_dim_ssm1 = 1200  
        self.hidden_dim_ssm2 = 900  

        self.embedding_model = SentenceTransformer('all-mpnet-base-v2') 


        self.model = stateSpaceModel(self.state_dim, self.input_dim, self.hidden_dim_ssm1, self.hidden_dim_ssm2, self.output_dim)
        self.model.eval() 

        self.ssm.load_state_dict(torch.load("./../..//NBF/ssm_models/ssm_model_exp5/models_best_ssm.pth")['ssm']) 
        self.ssm.eval()

    
        self.x_prev=torch.zeros(self.state_dim)
        self.x_prev_4step_back = None
        self.q=deque(maxlen=4)

    def feature_engineering(self, x_t, u_t, x_prev, x_prev_4step_back=None):
        features = []
    
        # drift features
        state_drift = torch.norm(x_t -x_prev).item()
        features.append(state_drift)

        state_input_distance = torch.norm(x_t -u_t).item()
        features.append(state_input_distance)
            
        long_term_state_drift = torch.norm(
            x_t - x_prev_4step_back
        ).item()
        features.append(long_term_state_drift)

        # similarity features
        state_similarity = F.cosine_similarity(
            x_t.unsqueeze(0),
            x_prev.unsqueeze(0)
        ).item()
        features.append(state_similarity)

        state_input_similarity = F.cosine_similarity(
            x_t.unsqueeze(0),
            u_t.unsqueeze(0)
        ).item()
        features.append(state_input_similarity)


        long_term_state_similarity = F.cosine_similarity(
            x_t.unsqueeze(0),
            x_prev_4step_back.unsqueeze(0)
        ).item()
        features.append(long_term_state_similarity)


        return torch.tensor(features, dtype=torch.float32)
    
    def extract_features(self, prompt):
        u = self.embedding_model.encode(prompt)
        u = torch.tensor(u, dtype=torch.float32)
        x_curr, zt = self.ssm(self.x_prev, u)
        
        if self.x_prev_4step_back is None:
            self.x_prev_4step_back = x_curr

        if len(self.q) == 4:
            self.x_prev_4step_back = self.q[0]
        
        self.q.append(x_curr)
        features = self.feature_engineering(x_curr, u, self.x_prev, self.x_prev_4step_back)
        self.x_prev = x_curr
        vectors=torch.concat([x_curr, u], dim=0)

        # TODO: add PCA here

        return features, vectors
        
