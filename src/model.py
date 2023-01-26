import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import scipy as sp


class UMI(nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.L1 = nn.Linear(in_size,out_size,bias=False)
        self.optim = optim.Adam(params=self.parameters(),lr=1e-2) # Weight decay implements L2 reg.
        self.L = nn.CrossEntropyLoss()

    def forward(self,x):
        x = self.L1(x)
        return F.softmax(x,dim=-1)

    def loss_fn(self,x,y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        return loss

    def train_step(self,x,y_true):
        self.optim.zero_grad()
        loss = self.loss_fn(x,y_true)
        loss.backward()
        self.optim.step()
        return loss.item()


class UMI_L1(UMI):
    def __init__(self,in_size,out_size,decay=0.01):
        super().__init__(in_size,out_size)
        self.decay = decay

    def loss_fn(self,x,y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        loss += self.decay*self.L1.weight.abs().sum()
        return loss


class UMI_L2(UMI):
    def __init__(self,in_size,out_size,decay=0.01):
        super().__init__(in_size,out_size)
        self.optim = optim.Adam(params=self.parameters(),lr=1e-2,weight_decay=decay)
        

class UMI_SVB(UMI):
    def __init__(self,in_size,out_size):
        super().__init__(in_size,out_size)
        torch.nn.init.orthogonal_(self.L1.weight, gain=1)

    def SVB(self,eps=0.001):
        old_weights = self.L1.weight.data.clone().numpy()
        w_svd = sp.linalg.svd(old_weights,full_matrices=False)
        U,s,V = w_svd[0],w_svd[1],w_svd[2]
        for i in range(len(s)):
            if s[i] > 1 + eps:
                s[i] = 1 + eps
            elif s[i] < 1/(1+eps):
                s[i] = 1/(1+eps)
            else:
                pass
        new_weights = U @ np.diag(s) @ V
        new_weights = torch.tensor(new_weights,dtype=torch.float32)
        self.L1.weight.data = new_weights