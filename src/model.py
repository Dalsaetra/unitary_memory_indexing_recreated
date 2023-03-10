import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, functional
import torch.optim as optim
import functorch as ft
import numpy as np
from tqdm import tqdm

class UMI(nn.Module):
    def __init__(self,in_size,out_size,lr=1e-2,bias=False):
        super().__init__()
        self.L1 = nn.Linear(in_size,out_size,bias=bias)
        self.lr = lr
        self.optim = optim.Adam(params=self.parameters(),lr=self.lr)
        self.L = nn.CrossEntropyLoss()
        self.in_size = in_size
        self.out_size = out_size

        self.losses = []
        self.epochs = []
        self.areas = []
        self.areas_mD = []
        self.w_his = []

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

    def train(self,loader,n_epochs=5,area=False):
        """Network training function

        Arguments:
            loader -- pytorch dataloader object of training data
            model -- pytorch nn object, neural network model

        Keyword Arguments:
            n_epochs -- number of epochs for training (default: {5})

        Returns:
            epochs, losses -- numpy arrays of epoch numbers (with decimals representing each batch) and loss values for each batch
        """
        
        N = len(loader)
        for epoch in tqdm(range(n_epochs)):
            for param in self.parameters():
                w = param
                w_np = w.detach().numpy().copy()
                self.w_his.append(w_np)
            for i, (inputs,labels) in tqdm(enumerate(loader)):
                loss = self.train_step(inputs,labels)
                self.losses.append(loss)
                self.epochs.append(epoch+i/N)
                if area:
                    self.areas.append(self.area_test())
                    self.areas_mD.append(self.area_test_multiD())
        return np.array(self.epochs), np.array(self.losses), np.array(self.areas), np.array(self.areas_mD)

    def area_test(self):
        weights = self.L1.weight.data.clone().detach()
        # Add a third dimension to the weights with zeros
        weights = torch.cat((weights,torch.zeros(weights.shape[0],1)),dim=1)
        area = torch.linalg.cross(weights[0],weights[1])
        for i in range(2,len(weights)):
            area = torch.linalg.cross(area,weights[i])
        return torch.linalg.norm(area).item()

    def area_test_multiD(self):
        weights = self.L1.weight.data.clone().detach()
        # Add a third dimension to the weights with zeros if the weights are 2D
        if len(weights) == 2:
            weights = torch.cat((weights,torch.zeros(weights.shape[0],1)),dim=1)

        # print(weights.T)
        area = torch.linalg.cross(weights.T[0],weights.T[1])
        return torch.linalg.norm(area).item()


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

    def train(self,loader,n_epochs=5):
        """Network training function for hard SVB
        """
        N = len(loader)
        for epoch in tqdm(range(n_epochs)):
            self.SVB(eps=0.001)
            for param in self.parameters():
                w = param
                w_np = w.detach().numpy().copy()
                self.w_his.append(w_np)
            for i, (inputs,labels) in tqdm(enumerate(loader)):
                loss = self.train_step(inputs,labels)
                self.losses.append(loss)
                self.epochs.append(epoch+i/N)
        return np.array(self.epochs), np.array(self.losses)
        

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


class UMI_Jacobi(UMI):
    def __init__(self,in_size,out_size,lr=1e-2,bias=False,decay=0.01):
        super().__init__(in_size,out_size,lr=lr,bias=bias)
        torch.nn.init.orthogonal_(self.L1.weight, gain=1)
        self.decay = decay

    def loss_fn(self, x, y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        for i in x:
            # jacobi = functional.jacobian(self.L1,i.float(),create_graph=True,strict=False,vectorize=True) # Jacobian of the first layer
            jacobi = functional.jacobian(self.L1,i.float(),create_graph=True,strict=True,vectorize=False) # Jacobian of the first layer
            loss += self.decay*(torch.linalg.det(jacobi.transpose(0,1)@jacobi) - 1) # Determinant of the Jacobian (Jacobian is a square matrix
        return loss


class UMI_Jacobi_2(UMI):
    def __init__(self,in_size,out_size,lr=1e-2,bias=False,decay=0.01, orthogonal=False):
        super().__init__(in_size,out_size,lr=lr,bias=bias)
        torch.nn.init.orthogonal_(self.L1.weight, gain=1)
        self.decay = decay
        self.cross_losses = []
        self.jacobi_losses = []
        if orthogonal:
            torch.nn.init.orthogonal_(self.L1.weight, gain=1)

    def loss_fn(self, x, y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        self.cross_losses.append(loss.item())

        jacobi = ft.vmap(ft.jacfwd(self.L1))(x)
        square_jacobi = (jacobi.transpose(1,2)@jacobi)
        # square_jacobi = (jacobi@jacobi.transpose(1,2))
    
        jacobi_loss = self.decay*((torch.linalg.det(square_jacobi) - 1)**2).sum()
        self.jacobi_losses.append(jacobi_loss.item())
        loss += jacobi_loss
        return loss

    def train(self,loader,n_epochs=5,area=False):
        """Network training function for Jacobi
        """
        N = len(loader)
        for epoch in tqdm(range(n_epochs)):
            for param in self.parameters():
                w = param
                w_np = w.detach().numpy().copy()
                self.w_his.append(w_np)
            for i, (inputs,labels) in tqdm(enumerate(loader)):
                loss = self.train_step(inputs,labels)
                self.losses.append(loss)
                self.epochs.append(epoch+i/N)
                if area:
                    self.areas.append(self.area_test())
                    # self.areas_mD.append(self.area_test_multiD())
        return np.array(self.epochs), np.array(self.losses), np.array(self.cross_losses), np.array(self.jacobi_losses), np.array(self.areas), np.array(self.areas_mD)


class UMI_Jacobi_2_savegrad(UMI):
    def __init__(self,in_size,out_size,lr=1e-2,bias=False,decay=0.01, orthogonal=False):
        super().__init__(in_size,out_size,lr=lr,bias=bias)
        torch.nn.init.orthogonal_(self.L1.weight, gain=1)
        self.lr_original = lr
        self.decay = decay
        self.cross_losses = []
        self.jacobi_losses = []
        self.cross_grad = []
        self.jacobi_grad = []
        if orthogonal:
            torch.nn.init.orthogonal_(self.L1.weight, gain=1)

    def loss_fn(self, x, y_true):
        y_hat = self(x.float())
        cross_loss = self.L(y_hat,y_true)
        self.cross_losses.append(cross_loss.item())
        cross_loss.backward(retain_graph=True)
        for param in self.parameters():
        #     print("Cross loss grad")
        #     print(param.grad)
            self.cross_grad.append(param.grad.detach().numpy().copy())
        self.cross_loss = cross_loss

        self.optim.zero_grad()
        for param in self.parameters():
            param.grad = None

        jacobi = ft.vmap(ft.jacfwd(self.L1))(x)
        square_jacobi = (jacobi.transpose(1,2)@jacobi)
    
        # jacobi_loss = self.decay*((torch.linalg.det(square_jacobi) - 1)**2).sum()
        jacobi_loss = self.decay*((torch.abs(torch.linalg.det(square_jacobi)) - 1)**2).sum()

        self.jacobi_loss = jacobi_loss
        self.jacobi_losses.append(jacobi_loss.item())
        jacobi_loss.backward(retain_graph=True)
        for param in self.parameters():
        #     print("Jacobi loss grad")
        #     print(param.grad)
            self.jacobi_grad.append(param.grad.detach().numpy().copy())
        
        self.optim.zero_grad()
        for param in self.parameters():
            param.grad = None

        loss = jacobi_loss + cross_loss
        return loss

    def train_step(self,x,y_true):
        
        self.optim.zero_grad()
        loss = self.loss_fn(x,y_true)
        loss.backward()
        self.optim.step()
        # for param in self.parameters():
        #     print("Full loss grad")
        #     print("Params grad:", param.grad)
        return loss.item()

    def check_stopped(self,threshold,lr_scaling):
        test_jacobi = torch.tensor(self.jacobi_grad[-1])
        test_cross = torch.tensor(self.cross_grad[-1])
        # Check if all the jacobi and cross gradients are pointing in the opposite direction with a value of 1 persent of the decay
        # If so, increase the learning rate by 100 for the next epoch
        test = torch.sum(test_jacobi*test_cross,dim=-1) < -threshold*self.decay*torch.ones(test_jacobi.shape[0])
        if test.equal(torch.ones(test_jacobi.shape[0])):
            print(f"Temporarily increasing learning rate by {lr_scaling}")
            self.lr = self.lr*lr_scaling
        self.optim = torch.optim.Adam(self.parameters(),lr=self.lr)


    def train(self,loader,n_epochs=5,area=False):
        """Network training function for Jacobi
        """
        
        N = len(loader)
        for epoch in tqdm(range(n_epochs)):
            # Check if the learning rate should be increased, if test just to avoid error for first epoch
            if len(self.jacobi_grad) > 1:
                self.check_stopped(0.1,100)
            for param in self.parameters():
                w = param
                w_np = w.detach().numpy().copy()
                self.w_his.append(w_np)
            for i, (inputs,labels) in tqdm(enumerate(loader)):
                loss = self.train_step(inputs,labels)
                self.losses.append(loss)
                self.epochs.append(epoch+i/N)
                if area:
                    self.areas_mD.append(self.area_test())
            self.lr = self.lr_original
            self.optim = torch.optim.Adam(self.parameters(),lr=self.lr)
        return np.array(self.epochs), np.array(self.losses), np.array(self.cross_losses), np.array(self.jacobi_losses), np.array(self.areas_mD), np.array(self.jacobi_grad), np.array(self.cross_grad)


class UMI_Jacobi_abs(UMI_Jacobi_2):
    def loss_fn(self, x, y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        self.cross_losses.append(loss.item())

        jacobi = ft.vmap(ft.jacfwd(self.L1))(x)
        square_jacobi = (jacobi.transpose(1,2)@jacobi)
        # square_jacobi = (jacobi@jacobi.transpose(1,2))
    
        jacobi_loss = self.decay*((torch.abs(torch.linalg.det(square_jacobi)) - 1)**2).sum()
        self.jacobi_losses.append(jacobi_loss.item())
        loss += jacobi_loss
        return loss



class UMI_Jacobi_flip(UMI):
    def __init__(self,in_size,out_size,lr=1e-2,bias=False,decay=0.01):
        super().__init__(in_size,out_size,lr=lr,bias=bias)
        torch.nn.init.orthogonal_(self.L1.weight, gain=1)
        self.decay = decay
        self.cross_losses = []
        self.jacobi_losses = []

    def loss_fn(self, x, y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        self.cross_losses.append(loss.item())

        jacobi = ft.vmap(ft.jacfwd(self.L1))(x)

        square_jacobi = (jacobi.transpose(1,2)@jacobi)
        
        jacobi_loss = self.decay*((torch.linalg.det(square_jacobi) - 1)**2).sum()
        self.jacobi_losses.append(jacobi_loss.item())
        loss += jacobi_loss
        return loss

    def flip(self,x,y_true):
        for i in range(len(self.L1.weight)):
            # y_hat1 = self(x.float())
            loss1 = self.loss_fn(x,y_true).item()
            old = self.L1.weight.data[i]
            self.L1.weight.data[i] = -self.L1.weight.data[i]
            loss2 = self.loss_fn(x,y_true).item()
            if loss1 > loss2:
                self.L1.weight.data[i] = -self.L1.weight.data[i]
                break
            else:
                self.L1.weight.data[i] = old

    def train(self,loader,n_epochs=5):
        """Network training function for Jacobi
        """
        j = 0
        N = len(loader)
        for epoch in tqdm(range(n_epochs)):
            j += 1
            if j%100 == 0:
                print(last_input,last_label)
                self.flip(last_input,last_label)
            for param in self.parameters():
                w = param
                w_np = w.detach().numpy().copy()
                self.w_his.append(w_np)
            for i, (inputs,labels) in tqdm(enumerate(loader)):
                loss = self.train_step(inputs,labels)
                self.flip(loader.dataset.x,loader.dataset.y)
                self.losses.append(loss)
                self.epochs.append(epoch+i/N)
                if i == 12-1:
                    last_input = inputs[0]
                    last_label = labels[0]
        return np.array(self.epochs), np.array(self.losses), np.array(self.cross_losses), np.array(self.jacobi_losses)

            
class UMI_J2(UMI):
    def __init__(self,in_size,out_size,lr=1e-2,bias=False,decay=0.01):
        super().__init__(in_size,out_size,lr=lr,bias=bias)
        self.decay = decay
        self.cross_losses = []
        self.jacobi_losses = []

    def loss_fn(self, x, y_true):
        y_hat = self(x.float())
        loss = self.L(y_hat,y_true)
        self.cross_losses.append(loss.item())
        jacobi = ft.vmap(ft.jacfwd(self.L1))(x)
        square_jacobi = (jacobi.transpose(1,2)@jacobi)
        # Loss minimize jacobi
        jacobi_loss = self.decay*(torch.linalg.norm(square_jacobi,ord="fro",dim=(1,2))**2).sum()
        self.jacobi_losses.append(jacobi_loss.item())
        loss += jacobi_loss
        return loss



class UMI_big(nn.Module):
    def __init__(self,in_size,out_size,hidden_size=100,lr=1e-2,bias=False):
        super().__init__()
        self.L1 = nn.Linear(in_size,hidden_size,bias=bias)
        self.L2 = nn.Linear(hidden_size,out_size,bias=bias)
        self.optim = optim.Adam(params=self.parameters(),lr=lr)
        self.L = nn.CrossEntropyLoss()

    def forward(self,x):
        x = F.sigmoid(self.L1(x))
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

class UMI_bigger(nn.Module):
    def __init__(self,in_size,out_size,hidden_size=100,lr=1e-2,bias=False):
        super().__init__()
        self.L1 = nn.Linear(in_size,hidden_size,bias=bias)
        self.L2 = nn.Linear(hidden_size,hidden_size,bias=bias)
        self.L3 = nn.Linear(hidden_size,out_size,bias=bias)
        self.optim = optim.Adam(params=self.parameters(),lr=lr)
        self.L = nn.CrossEntropyLoss()

    def forward(self,x):
        x = F.sigmoid(self.L1(x))
        x = F.sigmoid(self.L2(x))
        x = self.L3(x)
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