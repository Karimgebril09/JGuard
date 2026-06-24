import torch
import torch.nn as nn
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "..", "..","integrated", "models")

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
    

class ConversationDynamicsModel:
    def __init__(self):
        self.state_dim = 768    
        self.input_dim = 768    
        self.output_dim = 768   
        self.hidden_dim_ssm1 = 1200  
        self.hidden_dim_ssm2 = 900  

        self.ssm = stateSpaceModel(self.state_dim, self.input_dim, self.hidden_dim_ssm1, self.hidden_dim_ssm2, self.output_dim)
        checkpoint = torch.load(os.path.join(_MODELS_DIR, "models_best_ssm_new_data.pth"), map_location=torch.device("cpu"))
        self.ssm.load_state_dict(checkpoint["ssm"])
        self.ssm.eval()
        self.x_previous = torch.zeros(1, self.state_dim)

    
    def reset_state(self):
        self.x_previous = torch.zeros(1, self.state_dim)

    def get_state_dim(self):
        return self.state_dim


    def get_next_state(self, u):
        with torch.no_grad():
            u_tensor = torch.as_tensor(u, dtype=torch.float32)
            if u_tensor.dim() == 1:
                u_tensor = u_tensor.unsqueeze(0)
            x_curr, _ = self.ssm(self.x_previous, u_tensor)

        self.x_previous = x_curr
        return x_curr

