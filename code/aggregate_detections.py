import numpy as np

from basic_wrappers import BasicXformer,CustomGroupedXformer
from explainer import Archipelago



def aggregateDetections_only1D(model_wrap,valX,K,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    
    for k in range(K):
        if verbose:
            if k%10==0:
                print('\t',k)
        target_person = valX[k]
        baseline_person = np.zeros_like(valX[0])
        xf_k = BasicXformer(target_person,baseline_person); 

        archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
        detects = archipel_k.archdetect_1D()
        for key in detects:
            pass

        for interaction in detects['interactions']:
            key = interaction[0]
            value = interaction[1]
            if key in aggregate:
                aggregate[key] += value
                aggregate2[key] += detects['main_effects'][key]
                aggregate3[key] += detects['derivatives'][key]
            else:
                aggregate[key] = value
                aggregate2[key] = detects['main_effects'][key]
                aggregate3[key] = detects['derivatives'][key]
            
    
    for key in aggregate:
        aggregate[key] /= K
        aggregate2[key] /= K
        aggregate3[key] /= K
    return aggregate,aggregate2,aggregate3

def aggregateDetections_only2D(model_wrap,valX,K,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    
    for k in range(K):
        if verbose:
            if k%10==0:
                print('\t',k)
        target_person = valX[k]
        baseline_person = np.zeros_like(valX[0])
        xf_k = BasicXformer(target_person,baseline_person); 

        archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
        detects = archipel_k.archdetect_2D()
        for key in detects:
            pass

        for interaction in detects['interactions']:
            key = interaction[0]
            value = interaction[1]
            if key in aggregate:
                aggregate[key] += value
                aggregate2[key] += detects['pairwise_effects'][key]
                aggregate3[key] += detects['derivatives'][key]
            else:
                aggregate[key] = value
                aggregate2[key] = detects['pairwise_effects'][key]
                aggregate3[key] = detects['derivatives'][key]
            
    
    for key in aggregate:
        aggregate[key] /= K
        aggregate2[key] /= K
        aggregate3[key] /= K
    return aggregate,aggregate2,aggregate3

def aggregateDetections_anyD(model_wrap,valX,K,interaction_list,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    
    for k in range(K):
        if verbose:
            if k%10==0:
                print('\t',k)
        target_person = valX[k]
        baseline_person = np.zeros_like(valX[0])
        xf_k = BasicXformer(target_person,baseline_person); 

        archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
        detects = archipel_k.archdetect_kD(interaction_list)
        for key in detects:
            pass

        for interaction in detects['interactions']:
            key = interaction[0]
            value = interaction[1]
            if key in aggregate:
                aggregate[key] += value
                #aggregate2[key] += detects['main_effects'][key]
                aggregate3[key] += detects['derivatives'][key]
            else:
                aggregate[key] = value
                #aggregate2[key] = detects['main_effects'][key]
                aggregate3[key] = detects['derivatives'][key]
            
    
    for key in aggregate:
        aggregate[key] /= K
        #aggregate2[key] /= K
        aggregate3[key] /= K
    return aggregate,None,aggregate3



def aggregateContrastiveDetections_only1D(model_wrap,valX,K,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    aggregate4 = {}
    
    for k1 in range(K):
        for k2 in range(k1+1,K):
            if verbose:
                if k1%10==0 and k2==k1+1:
                    print('\t',k1)
            target_person = valX[k1]
            baseline_person = valX[k2]
            xf_k = BasicXformer(target_person,baseline_person); 

            archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
            detects = archipel_k.archdetect_1D()
            valid_inds = xf_k.get_contrastive_validities()

            for interaction in detects['interactions']:
                key = interaction[0]
                value = interaction[1]
                
                valid=True
                for thing in key:
                    if not valid_inds[thing]:
                        valid = False
                
                if valid:
                    if key in aggregate:
                        aggregate[key] += value
                        aggregate2[key] += detects['main_effects'][key]
                        aggregate3[key] += detects['derivatives'][key]
                        aggregate4[key] += 1
                    else:
                        aggregate[key] = value
                        aggregate2[key] = detects['main_effects'][key]
                        aggregate3[key] = detects['derivatives'][key]
                        aggregate4[key] = 1
                else:
                    if k1==K-2 and k2==K-1:
                        if key not in aggregate: 
                            aggregate[key] = 0.
                            aggregate2[key] = 0.
                            aggregate3[key] = 0.
                            aggregate4[key] = -1
            
    
    for key in aggregate:
        aggregate[key]  /= aggregate4[key]
        aggregate2[key] /= aggregate4[key]
        aggregate3[key] /= aggregate4[key]
    return aggregate,aggregate2,aggregate3,aggregate4

def aggregateContrastiveDetections_only2D(model_wrap,valX,K,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    aggregate4 = {}
    
    for k1 in range(K):
        for k2 in range(k1+1,K):
            if verbose:
                if k1%10==0 and k2==k1+1:
                    print('\t',k1)
            target_person = valX[k1]
            baseline_person = valX[k2]
            xf_k = BasicXformer(target_person,baseline_person); 

            archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
            detects = archipel_k.archdetect_2D()
            valid_inds = xf_k.get_contrastive_validities()

            for interaction in detects['interactions']:
                key = interaction[0]
                value = interaction[1]
                
                valid=True
                for thing in key:
                    if not valid_inds[thing]:
                        valid = False
                
                if valid:
                    if key in aggregate:
                        aggregate[key] += value
                        aggregate2[key] += detects['pairwise_effects'][key]
                        aggregate3[key] += detects['derivatives'][key]
                        aggregate4[key] += 1
                    else:
                        aggregate[key] = value
                        aggregate2[key] = detects['pairwise_effects'][key]
                        aggregate3[key] = detects['derivatives'][key]
                        aggregate4[key] = 1
                else:
                    if k1==K-2 and k2==K-1:
                        if key not in aggregate: 
                            aggregate[key] = 0.
                            aggregate2[key] = 0.
                            aggregate3[key] = 0.
                            aggregate4[key] = -1

    for key in aggregate:
        aggregate[key]  /= aggregate4[key]
        aggregate2[key] /= aggregate4[key]
        aggregate3[key] /= aggregate4[key]
    return aggregate,aggregate2,aggregate3,aggregate4

def aggregateContrastiveDetections_anyD(model_wrap,valX,K,interaction_list,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    aggregate4 = {}
    
    for k1 in range(K):
        for k2 in range(k1+1,K):
            if verbose:
                if k1%10==0 and k2==k1+1:
                    print('\t',k1)
            target_person = valX[k1]
            baseline_person = valX[k2]
            xf_k = BasicXformer(target_person,baseline_person);  

            archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
            detects = archipel_k.archdetect_kD(interaction_list)
            valid_inds = xf_k.get_contrastive_validities()

            for interaction in detects['interactions']:
                key = interaction[0]
                value = interaction[1]
                
                valid=True
                for thing in key:
                    if not valid_inds[thing]:
                        valid = False
                
                if valid:
                    if key in aggregate:
                        aggregate[key] += value
                        #aggregate2[key] += detects['main_effects'][key]
                        aggregate3[key] += detects['derivatives'][key]
                        aggregate4[key] += 1
                    else:
                        aggregate[key] = value
                        #aggregate2[key] = detects['main_effects'][key]
                        aggregate3[key] = detects['derivatives'][key]
                        aggregate4[key] = 1
                else:
                    if k1==K-2 and k2==K-1:
                        if key not in aggregate:
                            aggregate[key] = 0.
                            aggregate2[key] = 0.
                            aggregate3[key] = 0.
                            aggregate4[key] = -1
            
    
    for key in aggregate:
        aggregate[key]  /= aggregate4[key]
        #aggregate2[key] /= aggregate4[key]
        aggregate3[key] /= aggregate4[key]
    return aggregate,None,aggregate3,aggregate4



def aggregateGroupedContrastiveDetections_only1D(model_wrap,valX,K,CustomXformer,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    aggregate4 = {}
    
    for k1 in range(K):
        for k2 in range(k1+1,K):
            if verbose:
                if k1%10==0 and k2==k1+1:
                    print('\t',k1)
            target_person = valX[k1]
            baseline_person = valX[k2]
            xf_k = CustomXformer(target_person,baseline_person); 

            archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
            detects = archipel_k.archdetect_1D()
            valid_inds = xf_k.get_contrastive_validities()

            for interaction in detects['interactions']:
                key = interaction[0]
                value = interaction[1]
                
                valid=True
                for thing in key:
                    if not valid_inds[thing]:
                        valid = False
                
                if valid:
                    if key in aggregate:
                        aggregate[key] += value
                        aggregate2[key] += detects['main_effects'][key]
                        aggregate3[key] += detects['derivatives'][key]
                        aggregate4[key] += 1
                    else:
                        aggregate[key] = value
                        aggregate2[key] = detects['main_effects'][key]
                        aggregate3[key] = detects['derivatives'][key]
                        aggregate4[key] = 1
                else:
                    if k1==K-2 and k2==K-1:
                        if key not in aggregate: 
                            aggregate[key] = 0.
                            aggregate2[key] = 0.
                            aggregate3[key] = 0.
                            aggregate4[key] = -1
            
    
    for key in aggregate:
        aggregate[key]  /= aggregate4[key]
        aggregate2[key] /= aggregate4[key]
        aggregate3[key] /= aggregate4[key]
    return aggregate,aggregate2,aggregate3,aggregate4

def aggregateGroupedContrastiveDetections_only2D(model_wrap,valX,K,CustomXformer,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    aggregate4 = {}
    
    for k1 in range(K):
        for k2 in range(k1+1,K):
            if verbose:
                if k1%10==0 and k2==k1+1:
                    print('\t',k1)
            target_person = valX[k1]
            baseline_person = valX[k2]
            xf_k = CustomXformer(target_person,baseline_person); 

            archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
            detects = archipel_k.archdetect_2D()
            valid_inds = xf_k.get_contrastive_validities()

            for interaction in detects['interactions']:
                key = interaction[0]
                value = interaction[1]
                
                valid=True
                for thing in key:
                    if not valid_inds[thing]:
                        valid = False
                
                if valid:
                    if key in aggregate:
                        aggregate[key] += value
                        aggregate2[key] += detects['pairwise_effects'][key]
                        aggregate3[key] += detects['derivatives'][key]
                        aggregate4[key] += 1
                    else:
                        aggregate[key] = value
                        aggregate2[key] = detects['pairwise_effects'][key]
                        aggregate3[key] = detects['derivatives'][key]
                        aggregate4[key] = 1
                else:
                    if k1==K-2 and k2==K-1:
                        if key not in aggregate: 
                            aggregate[key] = 0.
                            aggregate2[key] = 0.
                            aggregate3[key] = 0.
                            aggregate4[key] = -1

    for key in aggregate:
        aggregate[key]  /= aggregate4[key]
        aggregate2[key] /= aggregate4[key]
        aggregate3[key] /= aggregate4[key]
    return aggregate,aggregate2,aggregate3,aggregate4

def aggregateGroupedContrastiveDetections_anyD(model_wrap,valX,K,CustomXformer,interaction_list,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    aggregate4 = {}
    
    for k1 in range(K):
        for k2 in range(k1+1,K):
            if verbose:
                if k1%10==0 and k2==k1+1:
                    print('\t',k1)
            target_person = valX[k1]
            baseline_person = valX[k2]
            xf_k = CustomXformer(target_person,baseline_person);  

            archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
            detects = archipel_k.archdetect_kD(interaction_list)
            valid_inds = xf_k.get_contrastive_validities()

            for interaction in detects['interactions']:
                key = interaction[0]
                value = interaction[1]
                
                valid=True
                for thing in key:
                    print(thing)
                    if not valid_inds[thing]:
                        valid = False
                
                if valid:
                    if key in aggregate:
                        aggregate[key] += value
                        #aggregate2[key] += detects['main_effects'][key]
                        aggregate3[key] += detects['derivatives'][key]
                        aggregate4[key] += 1
                    else:
                        aggregate[key] = value
                        #aggregate2[key] = detects['main_effects'][key]
                        aggregate3[key] = detects['derivatives'][key]
                        aggregate4[key] = 1
                else:
                    if k1==K-2 and k2==K-1:
                        if key not in aggregate:
                            aggregate[key] = 0.
                            aggregate2[key] = 0.
                            aggregate3[key] = 0.
                            aggregate4[key] = -1
            
    
    for key in aggregate:
        aggregate[key]  /= aggregate4[key]
        #aggregate2[key] /= aggregate4[key]
        aggregate3[key] /= aggregate4[key]
    return aggregate,None,aggregate3,aggregate4


def aggregateGroupedDetections_anyD(model_wrap,valX,K,CustomXformer,interaction_list,verbose=True):
    aggregate = {}
    aggregate2 = {}
    aggregate3 = {}
    aggregate4 = {}
    
    for k1 in range(K):
        #for k2 in range(k1+1,K):
        if True:
            k2=k1+1
            if verbose:
                if k1%10==0 and k2==k1+1:
                    print('\t',k1)
            target_person = valX[k1]
            #baseline_person = valX[k2]
            baseline_person = np.zeros_like(valX[0])
            xf_k = CustomXformer(target_person,baseline_person);  

            archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
            detects = archipel_k.archdetect_kD(interaction_list)
            #valid_inds = xf_k.get_contrastive_validities()

            for interaction in detects['interactions']:
                key = interaction[0]
                value = interaction[1]

                if key in aggregate:
                    aggregate[key] += value
                    #aggregate2[key] += detects['main_effects'][key]
                    aggregate3[key] += detects['derivatives'][key]
                    aggregate4[key] += 1
                else:
                    aggregate[key] = value
                    #aggregate2[key] = detects['main_effects'][key]
                    aggregate3[key] = detects['derivatives'][key]
                    aggregate4[key] = 1
            
    
    for key in aggregate:
        aggregate[key]  /= K
        #aggregate2[key] /= K
        aggregate3[key] /= K
    return aggregate,None,aggregate3

