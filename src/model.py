import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class UMI(nn.Module):
    def __init__(self,in_size,out_size,lr=1e-2,bias=False):
        super().__init__()
        self.L1 = nn.Linear(in_size,out_size,bias=bias)
        self.optim = optim.Adam(params=self.parameters(),lr=lr)
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
    def __init__(self,in_size,out_size,decay=0.01,lr=1e-2,bias=False):
        super().__init__(in_size,out_size,lr=lr,bias=bias)
        self.decay = decay

    def loss_fn(self,x,y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        loss += self.decay*self.L1.weight.abs().sum() # L1 reg.
        return loss


class UMI_L2(UMI):
    def __init__(self,in_size,out_size,decay=0.01,built_in=True,lr=1e-2,bias=False):
        """Initializes the model.

        Arguments:
            in_size -- number of input features
            out_size -- number of output features (labels/number of clusters)

        Keyword Arguments:
            decay -- L2 weight decay (default: {0.01})
            built_in -- If to use built in Adam weight decay (default: {True})
            lr -- Learning rate (default: {1e-2})
            bias -- If to use biases in addition to weights (default: {False})
        """
        super().__init__(in_size,out_size,lr=lr,bias=bias)
        if built_in:
            self.optim = optim.Adam(params=self.parameters(),lr=lr,weight_decay=decay) # Weight decay implements L2 reg.
            self.built_in = built_in
        else:
            self.decay = decay
            self.built_in = built_in

    def loss_fn(self,x,y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        if not self.built_in:
            loss += (self.decay*self.L1.weight**2).sum() # L2 reg.
        return loss
        

class UMI_SVB(UMI):
    def __init__(self,in_size,out_size,lr=1e-2,bias=False):
        super().__init__(in_size,out_size,lr=lr,bias=bias)
        torch.nn.init.orthogonal_(self.L1.weight, gain=1)

    def SVB(self,eps=0.001):
        """Implements hard singular value bounding as described in Jia et al. 2019.

        Keyword Arguments:
            eps -- Small constant that sets the weights a small interval around 1 (default: {0.001})
        """
        old_weights = self.L1.weight.data.clone().detach()
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
        # new_weights = torch.tensor(new_weights,dtype=torch.float32)
        self.L1.weight.data = new_weights # Update the weights for the model
        # self.L1.weight.data = torch.nn.parameter.Parameter()
        

class UMI_SVB_soft(UMI):
    def __init__(self,in_size,out_size,decay=0.01,lr=1e-2,bias=False):
        super().__init__(in_size,out_size,lr=lr,bias=bias)
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

    
class UMI_big(nn.Module):
    def __init__(self,in_size,out_size,hidden_size=100):
        super().__init__()
        self.L1 = nn.Linear(in_size,hidden_size,bias=False)
        self.L2 = nn.Linear(hidden_size,out_size,bias=False)
        self.optim = optim.Adam(params=self.parameters(),lr=1e-2)
        self.L = nn.CrossEntropyLoss()

    def forward(self,x):
        x = F.relu(self.L1(x))
        x = self.L2(x)
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