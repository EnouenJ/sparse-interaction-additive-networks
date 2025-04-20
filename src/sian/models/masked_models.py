import copy
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F



from sian.models.models  import Blocksparse_Deep_Relu_GAM #TODO: probably bad practice, need to rearrange a bit







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

        self.debug_verbose = False #01/29/2025

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
        self.biases = torch.nn.Parameter(torch.zeros(len(self.all_indices))) #02/09/2025 @ 1:00am


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
        if self.debug_verbose:
            print(h)
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



#TODO: need to support an empty indices set!!!

#From January 2024 -- copied here on 08/28/2024
# class MaskedGAM(nn.Module):
class InstaSHAPMasked_SIAN(nn.Module):
    # def __init__(self,sizes,indices,  dnn_on_or_off=False,small_sizes=[0,16,12,8,1]):
    def __init__(self,sizes,indices,  dnn_on_or_off=False,small_sizes=[0,16,12,8,1],feature_groups_dict=None):
        super(InstaSHAPMasked_SIAN,self).__init__()
        
        # #sizes[0] = 2*sizes[0]
        # self.sizes = sizes
        
        #sizes[0] = 2*sizes[0]
        self.sizes = copy.deepcopy(sizes)
        small_sizes = copy.deepcopy(small_sizes)
        small_sizes[-1] = sizes[-1]
        
        subset_indexer = np.zeros((self.sizes[0],len(indices)))
        subset_indexer_const = np.zeros((1,len(indices)))
        
        self.feature_groups_dict = feature_groups_dict #04/13/2025
        # for ii,ind in enumerate(indices):
        #     for i in ind:
        #         subset_indexer[i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
        for ii,ind in enumerate(indices):
            if self.feature_groups_dict is None:
                for i in ind:
                    subset_indexer[i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
                subset_indexer_const[0,ii] = 1 / (len(ind)+0.5)
            else:
                for i in ind:
                    # for sub_i in self.feature_groups_dict[i]:
                    sub_i = self.feature_groups_dict[i][0] #just use the first dimension (NOTE: assumes the full mask is initialized correctly)
                    subset_indexer[sub_i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
                subset_indexer_const[0,ii] = 1 / (len(ind)+0.5)

        self.subset_indexer_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer).float(), requires_grad=False )
        self.subset_indexer_const_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer_const).float(), requires_grad=False )
        
        
        self.indices = indices
        self.ungrouped_indices = indices
        if self.feature_groups_dict is not None:
            self.ungrouped_indices = []
            for ind in indices:
                new_ind = []
                for i in ind:
                    new_ind.extend(  self.feature_groups_dict[i]  )
                self.ungrouped_indices.append( tuple(new_ind) )
        self.value = 0
        # self.gam = Blocksparse_Deep_Relu_GAM(sizes[0],indices,small_sizes=small_sizes)
        self.gam = Blocksparse_Deep_Relu_GAM(sizes[0],self.ungrouped_indices,small_sizes=small_sizes) #04/13/2025
        self.precompute_off()

        self.debug_verbose = False #01/29/2025

    def forward(self, xx):
        x,S = xx
        x = x * S + self.value * (1-S)
#         if True: #appending
#             x = torch.cat((x,S),dim=1)

        if self.precomputed_shapes:
            all_shapes = self.all_shapes
        else:
            all_shapes = self.gam.forward_shapes(x)

#         shape_fn_mask = torch.ones_like(all_shapes)
#         for ii,ind in enumerate(self.indices):
#             for i in ind:
# #                 shape_fn_mask[:,ii] = torch.logical_and(shape_fn_mask[:,ii], S[:,i])
#                 shape_fn_mask[:,ii] = torch.logical_and(shape_fn_mask[:,ii], S[:,i][:,None]) #needed for classification tensor 

        shape_fn_mask = (torch.matmul(S.float(),self.subset_indexer_tensor)+self.subset_indexer_const_tensor>1).float()
        if False: #turning off for regression for now
            shape_fn_mask = shape_fn_mask[:,:,None] #needed for classification tensor 
        #print('all_shapes',all_shapes.shape)
        #print('shape_fn_mask',shape_fn_mask.shape)
        
        #print("S",S)
#         print("fn_mask",shape_fn_mask)
#         print(shape_fn_mask.dtype)
#         print(shape_fn_mask.shape)
        all_shapes = all_shapes*shape_fn_mask
        if self.debug_verbose:
            print(all_shapes)
#         gam_h = torch.sum(all_shapes,dim=1)[:,None] #REGRESSION VERSION -- idk need to handle this better later
        if True: #02/09/2025 -- whoops need to put this here
            # print("smile",(self.gam.biases[None]).repeat(x.shape[0],1).shape) #02/09/2025 - debugging
            # all_shapes=all_shapes+(self.gam.biases[None]).repeat(x.shape[0],1)*10 #02/09/2025 @ 1:00am
            all_shapes=all_shapes+(self.gam.biases[None]).repeat(x.shape[0],1)*100 #02/09/2025 @ 1:00am
        gam_h = torch.sum(all_shapes,dim=1) #classification version?
        if True: #for regression, adding back an empty dimension
            gam_h=gam_h[:,None]
            if True: #02/06/2025 -- adding this so constants can be accounted properly
                gam_h=gam_h+(self.gam.bias).repeat(x.shape[0]).unsqueeze(-1)*100
        dnn_h = torch.zeros_like(gam_h)
        shape_loss = torch.zeros(1)
        if True: #TURNING ON SHAPE_LOSS
            shape_loss = torch.sum(torch.abs(all_shapes),dim=1) #MIGHT BLOW UP FOR LARGE # OF INDICES IF I USE sum()
        if self.debug_verbose:
            print(gam_h)
            print(dnn_h)
        return dnn_h,gam_h,shape_loss

        

    def precompute_on(self, x):
        #print('precompute_on')
        self.all_shapes = self.gam.forward_shapes(x)
        #print(self.all_shapes.device)
        self.precomputed_shapes = True
    def precompute_off(self):
        self.all_shapes = None
        self.precomputed_shapes = False

    def collectParameters(self):
        gam_params = self.gam.collectParameters()
#         dnn_params = self.dnn.collectParameters()
#         return torch.cat([dnn_params,gam_params])
        return gam_params
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

        


#08/28/2024
class FastSHAPMasked_GAM(nn.Module):
    def __init__(self,sizes,indices,  dnn_on_or_off=False,small_sizes=[0,16,12,8,1]):
        super(FastSHAPMasked_GAM,self).__init__()
        
#         #sizes[0] = 2*sizes[0]
#         self.sizes = sizes
        
        #sizes[0] = 2*sizes[0]
        self.sizes = copy.deepcopy(sizes)
        small_sizes = copy.deepcopy(small_sizes)
        small_sizes[-1] = sizes[-1]
        
        subset_indexer = np.zeros((self.sizes[0],len(indices)))
        subset_indexer_const = np.zeros((1,len(indices)))
#         for ii,ind in enumerate(indices):
#             for i in ind:
#                 subset_indexer[i,ii] = 1 / (len(ind)-0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
        for ii,ind in enumerate(indices):
            for i in ind:
                subset_indexer[i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
            subset_indexer_const[0,ii] = 1 / (len(ind)+0.5)
        self.subset_indexer_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer).float(), requires_grad=False )
        self.subset_indexer_const_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer_const).float(), requires_grad=False )
        
        
        self.indices = indices
        self.value = 0
        #TODO: maybe incorporate the GAM stuff with this, but for now I don't want to copy and I don't want to touch the guts of SIAN
        # self.gam = Blocksparse_Deep_Relu_GAM(sizes[0],indices,small_sizes=small_sizes)
        self.fast_nets = torch.nn.ModuleList()
        for ii,ind in enumerate(indices):
            small_sizes[0]=sizes[0]
            self.fast_nets.append(  MLP(small_sizes) )

    def forward(self, xx):
        x,S = xx
        x = x * S + self.value * (1-S)
#         if True: #appending
#             x = torch.cat((x,S),dim=1)

            
        if False:
            all_shapes = self.gam.forward_shapes(x)
        else:
            fast_pred = [self.fast_nets[ii](x)   for ii,ind in enumerate(self.indices)]
            fast_pred = torch.concatenate([logits[0]+logits[1] for logits in fast_pred],dim=1) #TODO: double check for classification 
            all_shapes = fast_pred
            # all_shapes = #TODO put the fn here None
#         shape_fn_mask = torch.ones_like(all_shapes)
#         for ii,ind in enumerate(self.indices):
#             for i in ind:
# #                 shape_fn_mask[:,ii] = torch.logical_and(shape_fn_mask[:,ii], S[:,i])
#                 shape_fn_mask[:,ii] = torch.logical_and(shape_fn_mask[:,ii], S[:,i][:,None]) #needed for classification tensor 

        shape_fn_mask = (torch.matmul(S.float(),self.subset_indexer_tensor)+self.subset_indexer_const_tensor>1).float()
        if False: #turning off for regression for now
            shape_fn_mask = shape_fn_mask[:,:,None] #needed for classification tensor 
        #print('all_shapes',all_shapes.shape)
        #print('shape_fn_mask',shape_fn_mask.shape)
        
        #print("S",S)
#         print("fn_mask",shape_fn_mask)
#         print(shape_fn_mask.dtype)
#         print(shape_fn_mask.shape)
        all_shapes = all_shapes*shape_fn_mask
#         gam_h = torch.sum(all_shapes,dim=1)[:,None] #REGRESSION VERSION -- idk need to handle this better later
        gam_h = torch.sum(all_shapes,dim=1) #classification version?
        if True: #for regression, adding back an empty dimension
            gam_h=gam_h[:,None]
        dnn_h = torch.zeros_like(gam_h)
        shape_loss = torch.zeros(1)
        if True: #TURNING ON SHAPE_LOSS
            shape_loss = torch.sum(torch.abs(all_shapes),dim=1) #MIGHT BLOW UP FOR LARGE # OF INDICES IF I USE sum()
        return dnn_h,gam_h,shape_loss
        
    def forward_shapes(self, x):
            fast_pred = [self.fast_nets[ii](x)   for ii,ind in enumerate(self.indices)]
            fast_pred = torch.concatenate([logits[0]+logits[1] for logits in fast_pred],dim=1) #TODO: double check for classification 
            all_shapes = fast_pred
            return all_shapes

    def collectParameters(self):
        gam_params = self.gam.collectParameters()
#         dnn_params = self.dnn.collectParameters()
#         return torch.cat([dnn_params,gam_params])
        return gam_params
    def returnLinearNorms(self):
        return self.gam.returnLinearNorms()
    def returnQuadraticNorms(self):
        return self.gam.returnQuadraticNorms()
    # def forward_shapes(self, x):
    #     return self.gam.forward_shapes(x)
    def to(self, *args, **kwargs):
        # self.gam = self.gam.to(*args, **kwargs)
        self = super().to(*args, **kwargs) 
        return self
    # def compress(self):
    #     self.gam.compress()
    # def blocksparse(self):
    #     self.gam.blocksparse()



#From January 2024 -- copied here on 08/29/2024
class MaskedMLP(nn.Module):
    def __init__(self,sizes,small_sizes=[0,16,12,8,1]):
        super(MaskedMLP,self).__init__()
        sizes[0] = 2*sizes[0]
        self.sizes = sizes
        #self.dnn = MuP_Relu_DNN(sizes)
        self.dnn = FeedForwardDNN(sizes)
        self.value = 0 #masked value

    def forward(self, xx):
        x,S = xx
        x = x * S + self.value * (1-S)
        if True: #appending
            x = torch.cat((x,S),dim=1)
        dnn_h = self.dnn(x)
        gam_h = torch.zeros_like(dnn_h)
        return dnn_h,gam_h,None

    def collectParameters(self):
        dnn_params = self.dnn.collectParameters()
        return torch.cat([dnn_params])

    def precompute_on(self, x): #DO nothing for MLP, just to look pretty (for usage by GAM)
        # self.all_shapes = self.gam.forward_shapes(x)
        # self.precomputed_shapes = True
        pass
    def precompute_off(self):
        # self.all_shapes = None
        # self.precomputed_shapes = False
        pass
# class MaskedMLP2(nn.Module):
#     def __init__(self,sizes,small_sizes=[0,16,12,8,1]):
#         super(MaskedMLP2,self).__init__()
#         sizes[0] = 2*sizes[0]
#         self.sizes = sizes
#         self.dnn = MuP_Relu_DNN(sizes)
#         #self.dnn = FeedForwardDNN(sizes)
#         self.value = 0 #masked value

#     def forward(self, xx):
#         x,S = xx
#         x = x * S + self.value * (1-S)
#         if True: #appending
#             x = torch.cat((x,S),dim=1)
#         dnn_h = self.dnn(x)
#         gam_h = torch.zeros_like(dnn_h)
#         return dnn_h,gam_h,None
#     def collectParameters(self):
#         dnn_params = self.dnn.collectParameters()
#         return torch.cat([dnn_params])







#TODO: probably move this to main model area
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
    
