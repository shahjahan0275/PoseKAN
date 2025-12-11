from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
from models.ekan import KAN


class GSNetGraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj, beta=0.2, bias=True):
        super(GSNetGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta

        self.W = nn.Parameter(
            torch.zeros(size=(2, in_features, out_features), dtype=torch.float)
        )
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Ensure adj is a tensor
        '''
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float)
        elif adj.ndim == 1:
            adj = adj.unsqueeze(0)
        '''
        self.M = nn.Parameter(
            torch.zeros(size=(adj.size(1), out_features), dtype=torch.float))

        
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.A = adj

        self.Q = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.Q, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1.0 / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)
        
        self.kan = KAN([in_features, out_features])

    def forward(self, input, X):
        # Feature Transformation
        
        H = input

        


        # Adjacency Modulation
        A = self.beta * self.A.to(input.device) + self.beta * self.Q.to(input.device)



        # A = self.A.to(input.device) 
        A = (A + A.T) / 2


        I = torch.eye(A.size(0), dtype=torch.float).to(input.device)

        #Original
        first_term_temp = torch.matmul(((1 - self.beta) * I + self.beta * A), A)
        
        
        first_term = torch.matmul(first_term_temp, H) # KAGNN

        
        
        XW = X # for result 46.77 & 38.36

        # Without Weight Modulation
        output = self.kan(first_term) + self.kan(XW)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
                  
        else:
            return output
        

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
