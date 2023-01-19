import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim



class UMI(nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.L = nn.CrossEntropyLoss()
        self.optim = optim.Adam(params=self.parameters(),lr=1e-3)

        self.L1 = nn.Linear(in_size,out_size)

    def forward(self,x):
        x = self.L1(x)
        return F.softmax(x,dim=-1)

    def loss_fn(self,x,y_true):
        y_hat = self(x)
        loss = self.L(y_hat,y_true)
        return loss.item()

    def train_step(self,x,y_true):
        self.optim.zero_grad()
        loss = self.loss_fn(x,y_true)
        loss.backward()
        self.optim.step()
        return loss.item()