import torch
import numpy as np

        
class ModelWrapperTorch:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def get_predictions(self, batch_ppl):
        batch_ppl = torch.FloatTensor(batch_ppl).to(self.device)
        batch_conf = self.model(batch_ppl)
        return batch_conf.data.cpu()

    def __call__(self, batch_ppl):
        batch_predictions = self.get_predictions(batch_ppl)
        batch_predictions2 = (batch_predictions[:,0]).unsqueeze(1).numpy() #pre-merged logits
        return batch_predictions2
        
class MixedModelWrapperTorch: 
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def get_predictions(self, batch_ppl):
        batch_ppl = torch.FloatTensor(batch_ppl).to(self.device)
        batch_conf = self.model(batch_ppl)
        return batch_conf

    def __call__(self, batch_ppl):
        batch_predictions = self.get_predictions(batch_ppl)
        batch_predictions = batch_predictions[0] + batch_predictions[1]
        batch_predictions = batch_predictions.data.cpu()
        batch_predictions2 = (batch_predictions[:,0]).unsqueeze(1).numpy() #pre-merged logits
        return batch_predictions2
        
class MixedModelEnsembleWrapperTorch:
    def __init__(self, models, device):
        self.models = [model.to(device) for model in models]
        self.device = device
        
    def get_predictions(self, batch_ppl):
        batch_ppl = torch.FloatTensor(batch_ppl).to(self.device)
        batch_logit = torch.zeros(batch_ppl.shape[0])
        for i in range(len(self.models)):
            logits = self.models[i](batch_ppl)
            logits = (logits[0]+logits[1]).data.cpu()
            batch_logit += logits.narrow(1,0,1).squeeze()
        batch_logit /= len(self.models)
        return batch_logit

    def __call__(self, batch_ppl):
        batch_logit = self.get_predictions(batch_ppl)
        return (batch_logit).unsqueeze(1).numpy() 





class BloodMixedEnsembleWrapperTorchLogit:
    def __init__(self, models1, device):
        self.models1 = [model.to(device) for model in models1]
        self.device = device
        
    def get_predictions(self, batch_ppl):
        batch_ppl = torch.FloatTensor(batch_ppl).to(self.device)
        batch_logit1 = torch.zeros(batch_ppl.shape[0])
        for i in range(len(self.models1)):
            logits = self.models1[i](batch_ppl)
            logits = (logits[0]+logits[1]).data.cpu()
            batch_logit1 += logits.narrow(1,0,1).squeeze()
        batch_logit1 /= len(self.models1)

        return batch_logit1

    def __call__(self, batch_ppl):
        batch_logit1 = self.get_predictions(batch_ppl)
        return (batch_logit1).unsqueeze(1).numpy()  

class SKlearnEnsembleWrapperLogit:
    def __init__(self, models, merge_logits=True):
        self.models = models
        self.merge_logits = merge_logits
        

    def get_predictions(self, batch_ppl):
        batch_logit = np.zeros(batch_ppl.shape[0])
        for i in range(len(self.models)):
            ###logits = self.models[i].predict_proba(batch_ppl)[:,1]
            logits = self.models[i].predict(batch_ppl)
            batch_logit += logits
        batch_logit /= len(self.models)

        return batch_logit
    
    def __call__(self, batch_ppl):
        batch_predictions = self.get_predictions(batch_ppl)
        if self.merge_logits:
            return np.expand_dims(batch_predictions,axis=-1)
        else:
            return batch_predictions.numpy()














def get_efficient_mask_indices(inst, baseline, target):
    invert = np.sum(1*inst) >= len(inst)//2
    if invert:
        context = target.copy()
        insertion_target = baseline
        mask_indices = np.argwhere(inst==False).flatten()
    else:
        context = baseline.copy()
        insertion_target = target
        mask_indices = np.argwhere(inst==True).flatten()
    return mask_indices, context, insertion_target




class BasicXformer:
    def __init__(self, target_ppl, baseline_ppl):
        self.target = target_ppl
        self.baseline = baseline_ppl
        self.num_features = len(self.target)

    def simple_xform(self, inst):
        mask_indices = np.argwhere(inst==True).flatten()
        id_list = list(self.baseline)
        for i in mask_indices:
            id_list[i] = self.target[i]
        return id_list
        
    def efficient_xform(self, inst):
        mask_indices, base, change = get_efficient_mask_indices(inst, self.baseline, self.target)
        for i in mask_indices:
            base[i] = change[i]
        return base

    def get_contrastive_validities(self):
        validities = {}
        for i in range(self.num_features):
            if self.target[i]==self.baseline[i]:
                validities[i] = False
            else:
                validities[i] = True
        return validities

    def __call__(self, inst):
        instance = self.efficient_xform(inst)
        return instance 


def CustomGroupedXformer(num_grouped_features,num_full_features,feature_grouping_dictionary):
    
    class CustomizedGroupedXformer():
        def __init__(self, target_ppl, baseline_ppl):
            self.target = target_ppl
            self.baseline = baseline_ppl
            self.num_features = num_grouped_features
            self.num_full_features = num_full_features

        def group_masks(self, inst):
            new_inst = np.ones( self.num_full_features ).astype(bool)
            for grouped_feat in feature_grouping_dictionary:
                corresp_feats=feature_grouping_dictionary[grouped_feat] #all features corresponding to the group
                new_inst[corresp_feats] = inst[grouped_feat]
            return new_inst
            
        def simple_xform(self, inst):
            mask_indices = np.argwhere(inst==True).flatten()
            id_list = list(self.baseline)
            for i in mask_indices:
                id_list[i] = self.target[i]
            return id_list
            
        def efficient_xform(self, inst):
            inst = self.group_masks(inst)
            mask_indices, base, change = get_efficient_mask_indices(inst, self.baseline, self.target)
            for i in mask_indices:
                base[i] = change[i]
            return base

        def get_contrastive_validities(self):
            validities = {}
            for grouped_feat in feature_grouping_dictionary:
                corresp_feats=feature_grouping_dictionary[grouped_feat] #all features corresponding to the group
                if (self.target[corresp_feats]==self.baseline[corresp_feats]).all():
                    validities[grouped_feat] = False
                else:
                    validities[grouped_feat] = True
            return validities

        def __call__(self, inst):
            instance = self.efficient_xform(inst)
            return instance

    return CustomizedGroupedXformer

    

class MaskedXformer:
    def __init__(self, target_ppl, masker):
        self.target = target_ppl
        self.masker = masker
        self.num_features = len(self.target)

    def simple_xform(self, cond_mask):
        return self.masker( (self.target,cond_mask) )
        
    # def efficient_xform(self, inst):
    #     mask_indices, base, change = get_efficient_mask_indices(inst, self.baseline, self.target)
    #     for i in mask_indices:
    #         base[i] = change[i]
    #     return base

    def get_contrastive_validities(self):
        validities = {}
        for i in range(self.num_features):
            # if self.target[i]==self.baseline[i]:
            if False: #should always be valid with this style of masking??
                validities[i] = False
            else:
                validities[i] = True
        return validities

    def __call__(self, cond_mask):
        instance = self.simple_xform(cond_mask)
        return instance 

#for usage when the pytorch model already accepts masks as a pair
class MaskedXformer_v2:
    # def __init__(self, target_ppl, masker):
    def __init__(self, target_ppl):
        self.target = target_ppl
        #self.masker = masker
        self.num_features = len(self.target)

    def simple_xform(self, cond_mask):
        ####return self.masker( (self.target,cond_mask) )
        return (self.target,cond_mask)
        
    # def efficient_xform(self, inst):
    #     mask_indices, base, change = get_efficient_mask_indices(inst, self.baseline, self.target)
    #     for i in mask_indices:
    #         base[i] = change[i]
    #     return base

    def get_contrastive_validities(self):
        validities = {}
        for i in range(self.num_features):
            # if self.target[i]==self.baseline[i]:
            if False: #should always be valid with this style of masking??
                validities[i] = False
            else:
                validities[i] = True
        return validities

    def __call__(self, cond_mask):
        instance = self.simple_xform(cond_mask)
        return instance 


class Masked_MixedModelWrapperTorch:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def get_predictions(self, batch_ppl):
        #print('batch_ppl',batch_ppl)
        batch_inp = torch.FloatTensor(batch_ppl[0]).to(self.device)
        batch_msk = torch.FloatTensor(batch_ppl[1]).to(self.device)
        #print('batch_inp',batch_inp.shape)
        #print('batch_msk',batch_msk.shape)
        batch_conf = self.model( (batch_inp,batch_msk) )
        return batch_conf

    def __call__(self, batch_ppl):
        batch_predictions = self.get_predictions(batch_ppl)
        batch_predictions = batch_predictions[0] + batch_predictions[1]
        batch_predictions = batch_predictions.data.cpu()
        batch_predictions2 = (batch_predictions[:,0]).unsqueeze(1).numpy() #pre-merged logits
        return batch_predictions2