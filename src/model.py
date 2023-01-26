import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


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
        

