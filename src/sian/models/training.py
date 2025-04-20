import time
import os
import copy
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader



class TrainingArgs():
    pass

    def __init__(self, batch_size=32, number_of_epochs=300, learning_rate=5e-3, device=None, reduction_percentage=None):

        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.reduction_percentage = reduction_percentage

        self.loss_type = None #"mse" or "ce" for regression or classificaiton
        self.opt_type = "Adagrad" #default

        self.trnval_shuffle_seed = 0
        self.trnval_split_percentage = 0.70
        self.lambda1 = 5e-5


        class VerbositySettings():
            def __init__(self):
                self.VERBOSE_TRAINING = True
                pass
        self.verbosity_settings = VerbositySettings()

        class SavingSettings():
            def __init__(self):
                self.results_save_path_prefix = "results/"
                self.things_to_save = {} #TODO: dictionary or list? dpeends on where I store what I am saving while I generate it
                pass
        self.saving_settings = SavingSettings()


        class ModelConfig():
            def __init__(self):
                pass #TODO: probably should have this somewhere else or at least I mean defined somewhere else
        self.model_config = ModelConfig()

    def to_string(self):
        raise NotImplementedError("sorry, not implemented yet")



def either_normal_or_masked___gradient_descent_training(dataset_object, net, training_args):
    
    BS = training_args.batch_size
    EP = training_args.number_of_epochs
    LR = training_args.learning_rate
    
    lambda1 = training_args.lambda1
    if True:
        normalize_XY = (False,False) #TODO: move out as hyper 
        # normalize_XY = (True,True)

        # normalize_XY = (True,False) #Ttrying on 04/13/2025 @ 9:30pm -- should have been using more often presumably  -- off @ 12:30am
        
    if True:
        VERBOSE_TRAINING = training_args.verbosity_settings.VERBOSE_TRAINING
        
    MASKING_MODE = False #TODO: actually use this as an arugment
    MASKING_MODE = True
    MASKING_MODE = training_args.model_config.is_masked


    results_save_prefix = training_args.saving_settings.results_save_prefix  
    results_to_save = training_args.saving_settings.results_to_save 
    net_name = training_args.saving_settings.net_name 

    

    device = training_args.device
    net = net.to(device)

    # Initialize tracking lists for Excel output
    epoch_data = [] #TODO: looks unnecessary


    if True: #OLD VERSION WOULD SHUFFLE DIRECTLY HERE, keeping true block a little longer
        trnval_shuffle_seed = training_args.trnval_shuffle_seed
        trnval_per = training_args.trnval_split_percentage
        dataset_object.shuffle_and_split_trnval(trnval_shuffle_seed=trnval_shuffle_seed,trnval_split_percentage=trnval_per)

        trnX, trnY, valX, valY = dataset_object.pull_trnval_data()  #TODO: this is maybe not necessary, can think alongside 'trn_loader'
        print('trnX','trnY',trnX.shape,trnY.shape)

        train_reduction_percentage = training_args.reduction_percentage #NOTE: later it is worth revisiting to enable trn-only or trnval reductions separately
        if train_reduction_percentage is not None:
            trn_N = trnX.shape[0]
            trn_N = int(train_reduction_percentage*trn_N)
            trnX = trnX[:trn_N]
            trnY = trnY[:trn_N]
            

    if normalize_XY[0]:
        # trnval_m = np.mean(trnvalX,axis=0)
        # trnval_v = np.sqrt(np.var(trnvalX,axis=0))
        trnval_m = np.mean(trnX,axis=0)
        trnval_v = np.sqrt(np.var(trnX,axis=0))
        trnX = (trnX-trnval_m)/trnval_v
        valX = (valX-trnval_m)/trnval_v
    if normalize_XY[1]:
        trnval_m = np.mean(trnvalY,axis=0)
        trnval_v = np.sqrt(np.var(trnvalY,axis=0))
        trnY = (trnY-trnval_m)/trnval_v
        valY = (valY-trnval_m)/trnval_v



    trn_data = TensorDataset(torch.from_numpy(trnX).float().to(device),
                              torch.from_numpy(trnY).float().to(device))
    trn_loader = DataLoader(dataset=trn_data, batch_size=BS, shuffle=True)




    #TODO: only if tracking these in eval as well
    trn_tensor = torch.from_numpy(trnX).float().to(device)
    val_tensor = torch.from_numpy(valX).float().to(device)
    
    if training_args.opt_type=="Adagrad":
        opt = optim.Adagrad(net.parameters(), lr=LR)
    else:
        raise Exception(f"training_args.opt_type={training_args.opt_type} not recognized")
    
    #TODO: add these to all probably
    D = dataset_object.get_D()
    all_trn_accs = np.zeros(EP)
    all_val_accs = np.zeros(EP)
    all_losses = np.zeros((EP,len(trn_loader),7))
    all_trn_losses = np.zeros((EP,len(trn_loader)))
    all_val_losses = np.zeros(EP)

    if MASKING_MODE:
        all_subset_losses = torch.zeros((EP,D+1,7)).to(device)
        subset_indexer = torch.ones(D).long() 
        subset_indexer = torch.ones(D).float().to(device) #"addmv_impl_cuda" not implemented for 'Long'
    

    best_val_score = -100 #TODO: make more general
    best_net = None



    train_accuracy,val_accuracy=None,None
    full_training_start_time = time.time()
    #TODO: seed the random batching (independently of the model)
    for k in range(EP):
        if VERBOSE_TRAINING:
            print('Epoch', k)
        epoch_start_time = time.time()

        ### Training phase
        net.train()
        # for x_batch, y_batch in trn_loader:
        for j, (x_batch, y_batch) in enumerate(trn_loader):
            # print('x_batch',x_batch.shape)
            # print('y_batch',y_batch.shape)
            if not MASKING_MODE:
                dnn_logits, gam_logits, shape_loss = net(x_batch)
                logits = dnn_logits + gam_logits
            else:
                if True:
                    mask_prob = (k/EP*3) - 1.0   #this should be a simple ramp of (1/3 off; 1/3 linear; 1/3 on) 
                    unmask_prob = float(np.clip(mask_prob,0.0,1.0))
                    # print('unmask_prob',unmask_prob)
                    
                    mask_prob = 1.0-unmask_prob
        #             s_batch = (torch.randn(size=(x_batch.shape))>mask_prob).float()
        #             if np.random.rand(1)>0.5:
        #             if k<100 or np.random.rand(1)>0.5:
        #             if False:
                    if np.random.rand(1)>mask_prob:
                        s_batch = torch.ones_like(x_batch).long()
                        t_batch = torch.ones_like(x_batch).long()
                    else:
                        if False:
                            mask_p = mask_prob
                        mask_p_s = float(np.random.rand(1))
                        s_batch = (torch.rand(size=(x_batch.shape))>mask_p_s).long().to(device)
                        #Shapley dist #I HAVE BEEN DOING randN this whole time!!

                #TODO: s_batch and multi S loop (maybe S loop is not a big deal)
                dnn_logits,gam_logits,shape_loss = net( (x_batch,s_batch) )
                logits = dnn_logits + gam_logits

            l1_reg = torch.zeros(1).to(device)
            all_linear_params = net.collectParameters()
            l1_reg = lambda1 * torch.norm(all_linear_params, 1)

            mseloss_ = (y_batch.narrow(1, 0, 1) - logits.narrow(1, 0, 1)) ** 2
            mseloss = torch.mean(mseloss_) #NOTE: evaluate whether 'mseloss_' is worth having for non-masking version
            mseloss = mseloss

            loss = mseloss + l1_reg
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            # if BLAH_BLAH_BLAH___every_epoch in results_to_save: #TODO
            #     pass
            # Things worth tracking -- TODO eventually switch over to a more modular and adaptable sysmte
            all_trn_losses[k,j] = mseloss.item()
            all_losses[k,j,0] = loss.item()
            all_losses[k,j,1] = mseloss.item()
            all_losses[k,j,2] = l1_reg.item()

            if MASKING_MODE:
                s_index = torch.matmul(s_batch.float(),subset_indexer).long()
                all_subset_losses[k,:,0].index_put_( (s_index,), mseloss_[:,0].detach(), accumulate=True)
                all_subset_losses[k,:,1].index_put_( (s_index,), torch.ones_like(mseloss_)[:,0],        accumulate=True)


        ### Validation Phase
        net.eval() 
        #NOTE: need to fix for the larger datasets anyways, cant load all trn at once
        if not MASKING_MODE:
            logits_train = net(trn_tensor)
        else:
            logits_train = net( (trn_tensor,torch.ones_like(trn_tensor)) )
        logits_train = logits_train[0] + logits_train[1]
        logits_train = logits_train.cpu().detach().numpy()[:, 0]
        train_accuracy = np.mean(((trnY[:, 0]) - logits_train) ** 2)

        if not MASKING_MODE:
            logits_val = net(val_tensor)
        else:
            logits_val = net( (val_tensor,torch.ones_like(val_tensor)) )
        logits_val = logits_val[0] + logits_val[1]
        logits_val = logits_val.cpu().detach().numpy()[:, 0]
        val_accuracy = np.mean(((valY[:, 0]) - logits_val) ** 2)

        
        # if k%1==0: #TODO: pretty this up
        if True:
            val_score = -all_val_accs[k]
            if val_score > best_val_score:
                best_val_score = val_score
                best_net = copy.deepcopy(net)

        # VANILLA - Track metrics
        epoch_time = time.time() - epoch_start_time
        if VERBOSE_TRAINING:
            print(f'MSE for train and val: {train_accuracy}, {val_accuracy}')
            print(f"--- {epoch_time:.3f} seconds in epoch ---")

        

        epoch_data.append({ #TODO: right idea, but needs a lot of changes
            # "Dataset":dataset_id,
            "Dataset":dataset_object.get_dataset_id(), #TODO: implement
            # "Model": model,
            "Epoch": k,
            "Train Accuracy (MSE)": train_accuracy,
            "Validation Accuracy (MSE)": val_accuracy,
            "Time Taken (s)": epoch_time
        })

    total_training_time = time.time() - full_training_start_time
    if VERBOSE_TRAINING:
        print('FULLY TRAINED USING', total_training_time, 'seconds')

    if 'final_net' in results_to_save: 
        file_prefix = results_save_prefix
        if 'sian' in net_name:
            net.compress()
        torch.save(net,  file_prefix + net_name + '_final_net.pt')
        if 'sian' in net_name:
            net.blocksparse()

    if 'best_net' in results_to_save:  #NOTE: slight concern with keeping two model copies on the GPU
        file_prefix = results_save_prefix
        if 'sian' in net_name:
            best_net.compress()
        torch.save(best_net,  file_prefix + net_name + '_best_net.pt')
        if 'sian' in net_name:
            best_net.blocksparse()


    return net, val_tensor, train_accuracy, val_accuracy, total_training_time, None, None, None #for instant compatibility with masked


def evaluate_model_on_test_set(dataset_object, net, training_args, is_masked_model=False):
    _, _, tstX, tstY = dataset_object.pull_data()
    device = training_args.device

    test_tensor = torch.from_numpy(tstX).float().to(device)  

    if is_masked_model:
        identity_mask = torch.ones_like(test_tensor).to(device)
        test_input = (test_tensor, identity_mask)

    net = net.to(device)
    net.eval()
    
    if is_masked_model:
        logits_test = net(test_input)
    else:
        logits_test = net(test_tensor)

    logits_test = logits_test[0] + logits_test[1]
    logits_test = logits_test.cpu().detach().numpy()[:, 0]
    test_accuracy = np.mean(((tstY[:, 0]) - logits_test) ** 2)

    print(f"Test MSE: {test_accuracy}")

    return test_accuracy

