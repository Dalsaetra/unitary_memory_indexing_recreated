import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class UMI(nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.L1 = nn.Linear(in_size,out_size,bias=False)
        self.optim = optim.Adam(params=self.parameters(),lr=1e-2)
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
        loss += self.decay*self.L1.weight.abs().sum() # L1 reg.
        return loss


class UMI_L2(UMI):
    def __init__(self,in_size,out_size,decay=0.01):
        super().__init__(in_size,out_size)
        self.optim = optim.Adam(params=self.parameters(),lr=1e-2,weight_decay=decay) # Weight decay implements L2 reg.
        

class UMI_SVB(UMI):
    def __init__(self,in_size,out_size):
        super().__init__(in_size,out_size)
        torch.nn.init.orthogonal_(self.L1.weight, gain=1)

    def SVB(self,eps=0.001):
        """Implements hard singular value bounding as described in Jia et al. 2019.

        Keyword Arguments:
            eps -- Small constant that sets the weights a small interval around 1 (default: {0.001})
        """
        old_weights = self.L1.weight.data.clone()
        w_svd = torch.linalg.svd(old_weights,full_matrices=False) # Singular value decomposition
        U,s,V = w_svd[0],w_svd[1],w_svd[2]
        for i in range(len(s)): # Main algorithm from Jia et al. 2019
            if s[i] > 1 + eps:
                s[i] = 1 + eps
            elif s[i] < 1/(1+eps):
                s[i] = 1/(1+eps)
            else:
                pass
        new_weights = U @ torch.diag(s) @ V # Reconstruct the matrix with the new singular values
        new_weights = torch.tensor(new_weights,dtype=torch.float32)
        self.L1.weight.data = new_weights # Update the weights for the model
        

class UMI_SVB_soft(UMI):
    def __init__(self,in_size,out_size,decay=0.01):
        super().__init__(in_size,out_size)
        torch.nn.init.orthogonal_(self.L1.weight, gain=1)
        self.decay = decay

    def loss_fn(self, x, y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        W = self.L1.weight
        # Main loss function term for soft SVB from Jia et al. 2019:
        W_orth = W.transpose(0,1) @ W # W^T * W
        W_orth = W_orth - torch.eye(W_orth.shape[0]) # W^T * W - I
        loss += self.decay*torch.linalg.norm(W_orth,ord="fro")**2 # Note that since we only have one layer we need not do this over weights from more layers
        return loss

    