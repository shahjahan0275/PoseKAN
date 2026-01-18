import torch
import torch.nn as nn
import torch.nn.functional as F
#from einops import rearrange
from torch import einsum, nn

class SwiGLU(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Configuration information
        # self.config=config
        # Linear transformation for the gating mechanism
        self.linear_gate=nn.Linear(output_dim, output_dim)
        # Linear transformation for the main branch
        self.linear=nn.Linear(output_dim, output_dim)
        # Random initialization of the beta parameter
        self.beta=torch.randn(1, requires_grad=True)
        
        # Using nn.Parameter for beta to ensure it's recognized as a learnable parameter
        self.beta=nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta) 
    
    def forward(self, x):
        # Swish-Gated Linear Unit computation
        swish_gate=self.linear_gate(x)* torch.sigmoid(self.beta*self.linear_gate(x))
        # Element -wise multiplication of the gate and main branch
        out=swish_gate*self.linear(x)
        return out








'''class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features, drop=0.):
        super(SwiGLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w1 = nn.Linear(in_features * 16, out_features)
        self.w2 = nn.Linear(in_features * 16, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = x.view(-1, self.in_features * 16)
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = self.dropout(x1) * F.sigmoid(x2)
        return x'''

