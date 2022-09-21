import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader


class SmallModel(nn.Module):
    def __init__(self,sizes):
        super(SmallModel,self).__init__()

        self.sizes = sizes
        self.length = len(self.sizes)-1
        self.activation = torch.sigmoid
        self.activation = F.relu
        
        self.hiddens = nn.ModuleList()
        for k in range(self.length):
            self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k+1]))

    def forward(self, x):
        h = x
        for k in range(self.length):
            h = self.hiddens[k](h)
            if k!= self.length-1:
                h = self.activation(h)
        return h

    def collectParameters(self):
        all_param_list = []
        for k in range(self.length):
            for x in self.hiddens[k].parameters():
                all_param_list.append(x.view(-1))
        return torch.cat(all_param_list)


class FeedForwardDNN(nn.Module):
    def __init__(self,sizes):
        super(FeedForwardDNN,self).__init__()

        self.sizes = sizes
        self.length = len(self.sizes)-1
        self.activation = torch.tanh
        self.activation = F.relu
        
        self.hiddens = nn.ModuleList()
        for k in range(self.length):
            self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k+1]))

    def forward(self, x):
        h = x
        for k in range(self.length):
            h = self.hiddens[k](h)
            if k!= self.length-1:
                h = self.activation(h)
        return h

    def collectParameters(self):
        all_param_list = []
        for k in range(self.length):
            for x in self.hiddens[k].parameters():
                all_param_list.append(x.reshape(-1))
        return torch.cat(all_param_list)



#initialization taken from Greg Yang's work on maximal update parameterization
#https://arxiv.org/abs/2011.14522
class MuP_Relu_DNN(nn.Module):
    def __init__(self,sizes):
        super(MuP_Relu_DNN,self).__init__()

        self.sizes = sizes
        
        self.length = len(self.sizes)-1
        self.activation = F.relu
        
        self.hiddens = nn.ModuleList()
        for k in range(self.length):
            self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k+1]))

        for ll,layer in enumerate(self.hiddens): #Greg muP (fan in except for first layer)
            if ll==0:
                torch.nn.init.kaiming_normal_(layer.weight.data,a=0,mode='fan_out', nonlinearity='relu')
            layer.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(layer.weight.data,a=0,mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        h = x
        for k in range(self.length):
            h = self.hiddens[k](h)
            if k==0:
                h*=np.sqrt(self.sizes[1]) #Greg muP
            if k!= self.length-1:
                h = self.activation(h)
            else:
                h/=np.sqrt(self.sizes[k]) #Greg muP
        return h

    def collectParameters(self):
        all_param_list = []
        for k in range(self.length):
            for x in self.hiddens[k].parameters():
                all_param_list.append(x.view(-1))
        return torch.cat(all_param_list)


class MLP(nn.Module):
    def __init__(self,sizes,small_sizes=[0,16,12,8,1]):
        super(MLP,self).__init__()
        self.sizes = sizes
        #self.dnn = MuP_Relu_DNN(sizes)
        self.dnn = FeedForwardDNN(sizes)

    def forward(self, x):
        dnn_h = self.dnn(x)
        gam_h = torch.zeros_like(dnn_h)
        return dnn_h,gam_h,None
    def collectParameters(self):
        dnn_params = self.dnn.collectParameters()
        return torch.cat([dnn_params])


# should be able to adapt between
# - large block sparse, fast in training
# - small dense network, small in memory
class Blocksparse_Deep_Relu_GAM(nn.Module):
    def __init__(self,feat_in,all_indices,small_sizes=[0,16,12,8,1]):
        super(Blocksparse_Deep_Relu_GAM,self).__init__()
        ##print("INITIALIZING")
        self.all_indices = all_indices
        self.used_indices = list(range(len(all_indices)))
        
        self.all_sizes = [small_sizes for _ in all_indices]
        self.activation = F.relu
        self.sizes = []
        self.length = len(self.all_sizes[0])-1
        for k in range(self.length+1):
            size_k = 0
            if k==0:
                size_k = feat_in
            else:
                for j,index in enumerate(self.all_indices):
                    size_k += self.all_sizes[j][k]
            self.sizes.append(size_k)
            ##print(size_k)

        self.mode = 'blocksparse' # or 'compressed'
        self.models = None
                

        self.grad_masks = []  
        for k in range(self.length):
            size_k1 = 0
            size_k2 = 0
            grad_mask = torch.zeros( (self.sizes[k+1],self.sizes[k]) )
            
            for j,index in enumerate(self.all_indices):
                curr_k2 = self.all_sizes[j][k+1]
                if k==0:
                    for i in index:
                        grad_mask[size_k2:size_k2+curr_k2,i] = 1
                else:
                    curr_k1 = self.all_sizes[j][k]
                    grad_mask[size_k2:size_k2+curr_k2,size_k1:size_k1+curr_k1] = 1
                    size_k1+=curr_k1;
                size_k2+=curr_k2;
            
            self.grad_masks.append(torch.Tensor(grad_mask))

        
        self.hiddens = nn.ModuleList()
        for k in range(self.length):
            self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k+1]))
        self.bias = torch.nn.Parameter(torch.zeros(1))


        def get_sparse_hook(param_idx,model):
            def hook(grad):
                grad = grad.clone()
                grad_mask = model.grad_masks[param_idx]
                grad = grad * grad_mask #gradient mask stays on same device as gradient b/c buffer register below
                return grad
            return hook
        
        for k in range(self.length):
            #it is quite possible there are better scaling laws than this, I did not do any serious calculations
            self.hiddens[k].weight = torch.nn.Parameter( self.hiddens[k].weight.data * self.grad_masks[k] ) 
            if k!=0:
                self.hiddens[k].weight = torch.nn.Parameter( self.hiddens[k].weight.data * self.grad_masks[k] *np.sqrt(len(self.all_indices)) )
                self.hiddens[k].bias   = torch.nn.Parameter( self.hiddens[k].bias.data *np.sqrt(len(self.all_indices)) )
            self.hiddens[k].weight.register_hook(get_sparse_hook(k,self))

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        if not self.grad_masks is None:
            new_grad_masks = [grad_mask.to(*args, **kwargs)  for grad_mask in self.grad_masks]
            self.grad_masks = new_grad_masks
        return self

    def forward(self, x):
        if self.mode=='blocksparse':
            h = x
            for k in range(self.length):
                h = self.hiddens[k](h)
                if k!= self.length-1:
                    h = self.activation(h)
            out = torch.sum(h,dim=1).unsqueeze(-1)
            out = out + (self.bias).repeat(x.shape[0]).unsqueeze(-1)
            #shape_loss = torch.abs(torch.mean(h,dim=0))
            shape_loss = torch.zeros(len(self.all_indices)) 
        elif self.mode=='compressed':
            features = []  #this sequential computation can take a long time
            for j,index in enumerate(self.all_indices):
                inputs = []
                for ind in index:
                    inputs.append( x.narrow(1,ind,1) )
                inputs = torch.cat(inputs,dim=1)

                ft = self.models[j](inputs)
                features.append( ft )

            if len(features)>0:
                features = torch.cat(features,dim=1)
                out = torch.sum(features,dim=1)
                out = out.unsqueeze(-1)
                out = out + (self.bias).repeat(x.shape[0]).unsqueeze(-1)
                #shape_loss = torch.abs(torch.mean(features,dim=0))
                shape_loss = torch.zeros(len(self.all_indices))
            else:
                out = torch.zeros(x.shape[0],1)
                shape_loss = torch.zeros(len(self.all_indices))
        return out,shape_loss

    def forward_shapes(self, x):
        if self.mode=='blocksparse':
            h = x
            for k in range(self.length):
                h = self.hiddens[k](h)
                if k!= self.length-1:
                    h = self.activation(h)
        return h

    def blocksparse(self):
        if self.mode=='compressed':
            self.mode = 'blocksparse'
            device = self.bias.device
        
            self.grad_masks = []  #regenerated each time, but maybe for some applications it is better to keep them
            for k in range(self.length):
                size_k1 = 0
                size_k2 = 0
                grad_mask = torch.zeros( (self.sizes[k+1],self.sizes[k]) )
                
                for j,index in enumerate(self.all_indices):
                    curr_k2 = self.all_sizes[j][k+1]
                    if k==0:
                        for i in index:
                            grad_mask[size_k2:size_k2+curr_k2,i] = 1
                    else:
                        curr_k1 = self.all_sizes[j][k]
                        grad_mask[size_k2:size_k2+curr_k2,size_k1:size_k1+curr_k1] = 1
                        size_k1+=curr_k1;
                    size_k2+=curr_k2;
                
                self.grad_masks.append(torch.Tensor(grad_mask))

            self.hiddens = nn.ModuleList()
            for k in range(self.length):
                self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k+1]))


            def get_sparse_hook(param_idx,model):
                def hook(grad):
                    grad = grad.clone()
                    grad_mask = model.grad_masks[param_idx]
                    grad = grad * grad_mask 
                    return grad
                return hook
            
            for k in range(self.length):
                size_k1 = 0
                size_k2 = 0
                layer_weight = torch.zeros( (self.sizes[k+1],self.sizes[k]) )
                layer_bias   = torch.zeros( self.sizes[k+1] )
                
                for j,index in enumerate(self.all_indices):
                    curr_k2 = self.all_sizes[j][k+1]
                    if k==0:
                        for ii,i in enumerate(index):
                            layer_weight[size_k2:size_k2+curr_k2,i] = self.models[j].hiddens[k].weight.data[:,ii]
                        layer_bias[size_k2:size_k2+curr_k2] = self.models[j].hiddens[k].bias.data
                    else:
                        curr_k1 = self.all_sizes[j][k]
                        layer_weight[size_k2:size_k2+curr_k2,size_k1:size_k1+curr_k1] = self.models[j].hiddens[k].weight.data
                        layer_bias[size_k2:size_k2+curr_k2] = self.models[j].hiddens[k].bias.data

                        size_k1+=curr_k1;
                    size_k2+=curr_k2;

                self.hiddens[k].weight = torch.nn.Parameter( layer_weight.to(device) ) 
                self.hiddens[k].bias   = torch.nn.Parameter(  layer_bias.to(device)  ) 
                self.hiddens[k].weight.register_hook(get_sparse_hook(k,self))

            del self.models
            self.models = None
            print('blocksparsed :]')

    def compress(self):
        if self.mode == 'blocksparse':
            self.mode = 'compressed'
            device = self.bias.device

            self.models = nn.ModuleList()
            for j,index in enumerate(self.all_indices):
                sizes = self.all_sizes[j]
                sizes[0] = len(index)
                #self.models.append(  GregMuP_Relu_DNN(sizes).to(device)  )
                self.models.append(  FeedForwardDNN(sizes).to(device)  )
                

            for k in range(self.length):
                size_k1 = 0
                size_k2 = 0

                for j,index in enumerate(self.all_indices):
                    curr_k2 = self.all_sizes[j][k+1]
                    curr_k1 = self.all_sizes[j][k]

                    if k==0:
                        firstlayer = torch.zeros( (curr_k2,len(index)) )
                        for ii,i in enumerate(index):
                            firstlayer[:,ii] = self.hiddens[k].weight.data[size_k2:size_k2+curr_k2,i]
                        self.models[j].hiddens[k].weight = torch.nn.Parameter( firstlayer.to(device) )
                        self.models[j].hiddens[k].bias   = torch.nn.Parameter( self.hiddens[k].bias.data[size_k2:size_k2+curr_k2].to(device) )
                    else:
                        self.models[j].hiddens[k].weight = torch.nn.Parameter( self.hiddens[k].weight.data[size_k2:size_k2+curr_k2,size_k1:size_k1+curr_k1].to(device) )
                        self.models[j].hiddens[k].bias   = torch.nn.Parameter( self.hiddens[k].bias.data[size_k2:size_k2+curr_k2].to(device) )

                    size_k1+=curr_k1;
                    size_k2+=curr_k2;


            del self.hiddens
            self.hiddens = None
            del self.grad_masks
            self.grad_masks = None
            print('compressed :)')

    def addIndices(self, indices):
        self.all_indices.append(indices)
        sizes = [len(indices),16,12,8,1]
        self.models.append(  SmallModel(sizes)  )
    def returnLinearNorms(self):
        return torch.zeros(1)
    def returnQuadraticNorms(self):
        return torch.zeros(1)
    def collectParameters(self):
        all_param_list = []
        if self.mode=='blocksparse':
            for k in range(self.length):
                for x in self.hiddens[k].parameters():
                    all_param_list.append(x.view(-1))
        elif self.mode=='compressed':
            for model in self.models:
                model_params = model.collectParameters()
                all_param_list.append(model_params)
            if len(all_param_list)>0:
                return torch.cat(all_param_list)
            else:
                return torch.zeros(1)
        return torch.cat(all_param_list) 


class SIAN(nn.Module):
    def __init__(self,sizes,indices,  dnn_on_or_off=False,small_sizes=[0,16,12,8,1]):
        super(SIAN,self).__init__()
        self.dnn_on = dnn_on_or_off
        if not self.dnn_on:
            sizes = [sizes[0],sizes[len(sizes)-1]]
        self.sizes = sizes
        self.dnn = MuP_Relu_DNN(sizes)
        self.gam = Blocksparse_Deep_Relu_GAM(sizes[0],indices,small_sizes=small_sizes)
    def forward(self, x):
        if self.dnn_on:
            dnn_h = self.dnn(x)
        else:
            dnn_h = self.dnn(x)*0
        gam_h, shape_loss = self.gam(x)
        return dnn_h,gam_h,shape_loss
    def collectParameters(self):
        dnn_params = self.dnn.collectParameters()
        gam_params = self.gam.collectParameters()
        return torch.cat([dnn_params,gam_params])
    def returnLinearNorms(self):
        return self.gam.returnLinearNorms()
    def returnQuadraticNorms(self):
        return self.gam.returnQuadraticNorms()
    def forward_shapes(self, x):
        return self.gam.forward_shapes(x)
    def to(self, *args, **kwargs):
        self.gam = self.gam.to(*args, **kwargs)
        self = super().to(*args, **kwargs) 
        return self
    def compress(self):
        self.gam.compress()
    def blocksparse(self):
        self.gam.blocksparse()





        


