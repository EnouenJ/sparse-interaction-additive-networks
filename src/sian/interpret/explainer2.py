

#TODO
#needs to do single inference and batch inference?

import numpy as np
import torch

# from sian.fis import powerset
# from sian.interpret import BasicXformer
# from sian.fis.combinatorial_utils import powerset #TODO: avoid '(most likely due to a circular import)' with more intentional import structures
from sian import powerset  #yes, this was the problem (still double checking)
from sian.interpret.basic_wrappers import BasicXformer
from sian.interpret.basic_wrappers import CustomGroupedXformer


class FID_Hyperparameters():
    pass


class masked_FID_Hyperparameters():
    def __init__(self, fid_masking_style, output_type, 
                       score_type_name, inc_rem_pel_list, 
                       grouped_features_dict):
                       
        self.is_masked_model = True
        self.masking_style  = fid_masking_style
        self.output_type = output_type

        self.score_type_name = score_type_name
        self.inc_rem_pel_list = inc_rem_pel_list
        self.grouped_features_dict = grouped_features_dict


class unmasked_FID_Hyperparameters():
    def __init__(self, fid_masking_style, output_type, 
                       score_type_name, inc_rem_pel_list,
                       device, #needed for wrapper object still 
                       grouped_features_dict):
                       
        self.is_masked_model = False
        self.masking_style  = fid_masking_style
        self.output_type = output_type

        self.score_type_name = score_type_name
        self.inc_rem_pel_list = inc_rem_pel_list
        self.grouped_features_dict = grouped_features_dict

        self.device = device
    









# Feature Interaction Detection -- Explainer Object
class FID_Explainer():
    def __init__(self):
        pass
    
    def get_explanation_scores(self, valX, AGG_K, current_frontier): #TODO: are these the best arguments to use?
        raise NotImplementedError("must extend this explanation score method")




class JamArchipelago():
    pass
    # def __init__(self, model):
    def __init__(self, trained_model, fid_hypers):

        self.model = trained_model

        if True:
            assert not fid_hypers.is_masked_model
            self.masking_style = fid_hypers.masking_style
            self.output_type = fid_hypers.output_type

            self.score_type_name = fid_hypers.score_type_name
            self.inc_rem_pel_list = fid_hypers.inc_rem_pel_list
            self.grouped_features_dict = fid_hypers.grouped_features_dict
            self.setGroupedXformer(self.grouped_features_dict)

        self.cached_results = None





        self.batch_size = 20
        self.verbose = True
        self.verbose = False
        self.output_indices = 0

    # def explain(self, )


    def setGroupedXformer(self, grouped_features_dict): #04/13/2025
        self.grouped_features_dict = grouped_features_dict
        D = grouped_features_dict["D"]
        D0 = grouped_features_dict["D0"]
        feature_grouping_dictionary = {}
        for d in range(D0):
            feature_grouping_dictionary[d] = grouped_features_dict[d]
        self.myGroupedXformer = CustomGroupedXformer(D0, D, feature_grouping_dictionary)



    def verbose_iterable(self, iterable):
        if self.verbose:
            from tqdm import tqdm
            return tqdm(iterable)
        else:
            return iterable



    #copied from Archipelago code (a post copy)
    def batch_set_inference(self, set_indices, context, insertion_target, include_context=False):
        """
            Creates archipelago type data instances and runs batch inference on them
            All "sets" are represented as tuples to work as keys in dictionaries
        """

        num_batches = int(np.ceil(len(set_indices) / self.batch_size))
        
        scores = {}
        for b in self.verbose_iterable(range(num_batches)):
            batch_sets = set_indices[b*self.batch_size:(b+1)*self.batch_size]
            # print('batch_sets',batch_sets)
            data_batch = []
            for index_tuple in batch_sets:
                new_instance = context.copy()
                for i in index_tuple:
                    new_instance[i] = insertion_target[i]

                if self.data_xformer is not None:
                    new_instance = self.data_xformer(new_instance)
                    #print('new_instance',new_instance) #08/29/2024

                data_batch.append(new_instance)

            if include_context and b == 0:
                if self.data_xformer is not None:
                    data_batch.append(self.data_xformer(context))
                else:
                    data_batch.append(context)
            # print('data_batch',data_batch)

            #08/29/2024 modification #TODO: fix this hackiness later
            ###preds = self.model(np.array(data_batch))
            if type(data_batch[0])==tuple:
                preds = self.model( (np.array([thing[0] for thing in data_batch]), np.array([thing[1] for thing in data_batch]))  )
            else:
                # print('not in masked land') #01/26/2025
                preds = self.model(np.array(data_batch))
            # print('preds',preds)

            for c, index_tuple in enumerate(batch_sets):
                scores[index_tuple] = preds[c, self.output_indices]    
            if include_context and b == 0:
                context_score = preds[-1, self.output_indices] 
        
        output = {"scores": scores}
        if include_context and num_batches > 0:
            output["context_score"] = context_score
        return output


    def get_archipelago_values(self, baseline, target, interactions_list, 
                               return_Inclusion_and_Removal_values=True, use_original_Archipelago=False, return_total_effects=True):
        
        
        
        use_original_Archipelago=True #NOTE: useless for right now


        inters_subset_list = []
        for interaction_tuple in interactions_list:
            pset_inter = powerset(interaction_tuple)
            for subset_tuple in pset_inter:
                if not subset_tuple in inters_subset_list:
                    inters_subset_list.append(subset_tuple)

        # all_inclusion_scores = self.get_inclusion_scores(baseline, target, inters_subset_list)
        # all_removal_scores   = self.get_removal_scores(baseline, target, inters_subset_list)
        # all_inclusion_scores = self.batch_set_inference(baseline, target, inters_subset_list)["scores"]
        # all_removal_scores   = self.batch_set_inference(target, baseline, inters_subset_list)["scores"] #TODO: code this
        # all_inclusion_scores = self.batch_set_inference(inters_subset_list, baseline, target)["scores"]
        # all_removal_scores   = self.batch_set_inference(inters_subset_list, target, baseline)["scores"] 
        # all_inclusion_scores = self.batch_set_inference(inters_subset_list, target, baseline)["scores"]
        # all_removal_scores   = self.batch_set_inference(inters_subset_list, baseline, target)["scores"] 
        # all_removal_scores   = self.batch_set_inference(inters_subset_list, target, baseline)["scores"] 
        # all_inclusion_scores = self.batch_set_inference(inters_subset_list, baseline, target)["scores"]

        target = np.ones( self.data_xformer.num_features ).astype(bool)
        baseline = np.zeros( self.data_xformer.num_features ).astype(bool)
        # print('target',target)
        all_inclusion_scores = self.batch_set_inference(inters_subset_list, baseline, target)["scores"]
        all_removal_scores   = self.batch_set_inference(inters_subset_list, target, baseline)["scores"] 



        all_scores = {}
        for interaction_tuple in interactions_list:
            inc_inter_score = 0.0
            rem_inter_score = 0.0
            inc_total_score = 0.0
            rem_total_score = 0.0


            pset_inter = powerset(interaction_tuple)
            for subset_tuple in pset_inter:
                inc_score_sub = all_inclusion_scores[subset_tuple]
                rem_score_sub = all_removal_scores[subset_tuple]

                #if interaction_tuple==(0,1): #verbose for now TODO remove
                # if interaction_tuple==(0,): #verbose for now TODO remove
                #     pass
                #     print(subset_tuple)
                #     print('inc_score_sub',inc_score_sub)
                #     print('rem_score_sub',rem_score_sub)
                #     print()
                
                if len(subset_tuple)==0:
                    inc_total_score -= inc_score_sub
                    rem_total_score += rem_score_sub
                elif len(subset_tuple)==len(interaction_tuple):
                    inc_total_score += inc_score_sub
                    rem_total_score -= rem_score_sub
                inc_inter_score += (-1)**(  (len(interaction_tuple)-len(subset_tuple))%2  ) * inc_score_sub
                rem_inter_score += (-1)**(  (len(subset_tuple))%2  ) * rem_score_sub


            old_arch_inter_score = 0.0
            old_arch_total_score = 0.0
            new_arch_inter_score = 0.0
            new_arch_total_score = 0.0
            old_arch_inter_score = 0.5*inc_inter_score**2 + 0.5*rem_inter_score**2
            old_arch_total_score = inc_total_score
            new_arch_inter_score = 0.5*inc_inter_score+0.5*rem_inter_score
            new_arch_total_score = 0.5*inc_total_score + 0.5*rem_total_score

            scores = {
                "inc_inter_score" : inc_inter_score,
                "rem_inter_score" : rem_inter_score,
                "inc_total_score" : inc_total_score,
                "rem_total_score" : rem_total_score,
                
                "old_arch_inter_score" : old_arch_inter_score,
                "old_arch_total_score" : old_arch_total_score,
                "new_arch_inter_score" : new_arch_inter_score,
                "new_arch_total_score" : new_arch_total_score,
            }
            all_scores[interaction_tuple] = scores
        return all_scores

    # def get_layerwise_archipelago_scores(self, valX, AGG_K, current_frontier):
    def get_explanation_scores(self, valX, AGG_K, current_frontier):
        if self.masking_style == 'zero_baseline':
            the_archipelago_scores = self._accumulate_baseline_archipelago(valX, AGG_K, current_frontier)
        elif self.masking_style == 'triangle_marginal':
            the_archipelago_scores = self._accumulate_triangle_interventional_archipelago(valX, AGG_K, current_frontier)
        elif self.masking_style == 'sampled_marginal':
            pass #NOTE: check if implemented
            #the_archipelago_scores = self._accumulate_sampled_interventional_archipelago(valX, AGG_K, current_frontier)
        else:
            raise Exception("Not implemented masking style, check masked versions instead")
        return the_archipelago_scores

    # def get_batchwise_archipelago_scores(self, valX, AGG_K, current_frontier):
    #     if self.masking_style == 'zero_baseline':
    #         the_archipelago_scores = self._accumulate_baseline_archipelago(valX, AGG_K, current_frontier)
    #     elif self.masking_style == 'triangle_marginal':
    #         the_archipelago_scores = self._accumulate_triangle_interventional_archipelago(valX, AGG_K, current_frontier)
    #     elif self.masking_style == 'sampled_marginal':
    #         pass #NOTE: check if implemented
    #         #the_archipelago_scores = self._accumulate_sampled_interventional_archipelago(valX, AGG_K, current_frontier)
    #     else:
    #         raise Exception("Not ipmlemented masking style, check masked versions instead")
    #     return the_archipelago_scores


    def _accumulate_triangle_interventional_archipelago(self, valX, K, interactions_list, accumultion_mode="simple_square"):
        

        
        
        verbose = True

        accum_scores = {}
        valid_counts = {}
        for subset in interactions_list:
            accum_scores[subset] = {}
            valid_counts[subset] = 0

        to_square_or_not_dict = {
                "inc_inter_score" : True,
                "rem_inter_score" : True,
                "inc_total_score" : True,
                "rem_total_score" : True,
                
                "old_arch_inter_score" : False,
                "old_arch_total_score" : True,
                "new_arch_inter_score" : True,
                "new_arch_total_score" : True,

        }
            
        for k1 in range(K):
            for k2 in range(k1+1,K):
                if verbose:
                    if k1%10==0 and k2==k1+1:
                        print('\t',k1)
                target_person = valX[k1]
                baseline_person = valX[k2]
                # print('target_person',target_person)
                # print('baseline_person',baseline_person)
                # xf_k = BasicXformer(target_person,baseline_person) 
                # self.data_xformer = BasicXformer(target_person,baseline_person)  #TODO: remove this
                # self.data_xformer = BasicXformer(baseline_person,target_person)  #TODO: wtf this

                if False: #04/13/2025 @ 2:30am
                    self.data_xformer = BasicXformer(target_person,baseline_person)
                else:
                    self.data_xformer = self.myGroupedXformer(target_person,baseline_person) 
                

                # archipel_k = Archipelago(model_wrap, data_xformer=xf_k,output_indices=0,batch_size=20)
                # detects = archipel_k.archdetect_1D()
                # valid_inds = xf_k.get_contrastive_validities()
                
                valid_inds = self.data_xformer.get_contrastive_validities()  #TODO: remove this
                #print(valid_inds)

                all_scores = self.get_archipelago_values(baseline_person,target_person,interactions_list,)

                for subset in all_scores:

                    valid=True
                    for thing in subset:
                        if not valid_inds[thing]:
                            valid = False

                    if valid:
                        score = all_scores[subset]
                        for name in score:
                            score_name = score[name]
                            #TODO: technically more eloquent would be to average over the 'target', then square, then average over all possible targets
                            if to_square_or_not_dict[name]: #need to accumulate over a positive quantity when aggregating for 
                                score_name=score_name**2
                            if name in accum_scores[subset]:
                                accum_scores[subset][name] += score_name
                            else:
                                accum_scores[subset][name] = score_name
                        valid_counts[subset] += 1
                    #     if subset==(0,):
                    #         print(accum_scores[subset]["old_arch_inter_score"])
                    else:
                        pass
                    #     if subset==(0,):
                    #         print("000")

        #take the mean instead of sum
        for subset in accum_scores:
            for name in accum_scores[subset]:
                # accum_scores[subset][name] /= K*(K-1)/2
                accum_scores[subset][name] /= valid_counts[subset] #TODO: I think there is a bug when there are no differences in the entire triangle set

        return accum_scores

        
    def _accumulate_baseline_archipelago(self, valX, K, interactions_list, accumultion_mode="simple_square"):
        verbose = True
        raise NotImplementedError("just ignoring this for now")

        accum_scores = {}
        valid_counts = {}
        for subset in interactions_list:
            accum_scores[subset] = {}
            valid_counts[subset] = 0

        to_square_or_not_dict = {
                "inc_inter_score" : True,
                "rem_inter_score" : True,
                "inc_total_score" : True,
                "rem_total_score" : True,
                
                "old_arch_inter_score" : False,
                "old_arch_total_score" : True,
                "new_arch_inter_score" : True,
                "new_arch_total_score" : True,

        }
    
        for k in range(K):
            if verbose:
                if k%10==0:
                    print('\t',k)
            target_person = valX[k]
            baseline_person = np.zeros_like(valX[0])
            self.data_xformer = BasicXformer(target_person,baseline_person) 
            valid_inds = self.data_xformer.get_contrastive_validities()    #TODO: remove these two lines




            all_scores = self.get_archipelago_values(baseline_person,target_person,interactions_list,)
            for subset in all_scores:

                valid=True
                for thing in subset:
                    if not valid_inds[thing]:
                        valid = False

                if valid:
                    score = all_scores[subset]
                    for name in score:
                        score_name = score[name]
                        #TODO: technically more eloquent would be to average over the 'target', then square, then average over all possible targets
                        if to_square_or_not_dict[name]: #need to accumulate over a positive quantity when aggregating for 
                            score_name=score_name**2
                        if name in accum_scores[subset]:
                            accum_scores[subset][name] += score_name
                        else:
                            accum_scores[subset][name] = score_name
                    valid_counts[subset] += 1

        to_remove_list =[]
        for subset in accum_scores:
            # a_subset = list(accum_scores.keys())[0]
            # for name in accum_scores[a_subset]:
            #     accum_scores[subset][name] = 0.0
            #     valid_counts[subset] = -1
            if valid_counts[subset]==0:
                to_remove_list.append(subset)
        # for subset in to_remove_list:
        #     del accum_scores[subset]

        #take the mean instead of sum
        for subset in accum_scores:
            for name in accum_scores[subset]:
                accum_scores[subset][name] /= valid_counts[subset]
            accum_scores[subset]['valid_counts'] = valid_counts[subset]
        return accum_scores






    def _accumulate_sampled_interventional_archipelago(self, valX, K, interactions_list, K2_count=None, accumultion_mode="simple_square"):
        
        raise NotImplementedError("just ignoring this for now")

        
        verbose = True

        accum_scores = {}
        valid_counts = {}
        for subset in interactions_list:
            accum_scores[subset] = {}
            valid_counts[subset] = 0

        to_square_or_not_dict = {
                "inc_inter_score" : True,
                "rem_inter_score" : True,
                "inc_total_score" : True,
                "rem_total_score" : True,
                
                "old_arch_inter_score" : False,
                "old_arch_total_score" : True,
                "new_arch_inter_score" : True,
                "new_arch_total_score" : True,

        }

        if K2_count is None:
            K2_count = K - 1


        for k1 in range(K):
            possible_k2s = (np.random.permutation(K-1)+k1+1)%K

            if verbose:
                if k1%10==0:
                    print('\t',k1)

            for k2 in possible_k2s[K2_count]:
                target_person = valX[k1]
                baseline_person = valX[k2]
                

                self.data_xformer = BasicXformer(target_person,baseline_person)  #TODO: remove this                
                valid_inds = self.data_xformer.get_contrastive_validities()  #TODO: remove this


                all_scores = self.get_archipelago_values(baseline_person,target_person,interactions_list,)

                for subset in all_scores:

                    valid=True
                    for thing in subset:
                        if not valid_inds[thing]:
                            valid = False

                    if valid:
                        score = all_scores[subset]
                        for name in score:
                            score_name = score[name]
                            #TODO: technically, more eloquent would be to average over the 'target', then square, then average over all possible targets
                            if to_square_or_not_dict[name]: #need to accumulate over a positive quantity when aggregating for 
                                score_name=score_name**2
                            if name in accum_scores[subset]:
                                accum_scores[subset][name] += score_name
                            else:
                                accum_scores[subset][name] = score_name
                        valid_counts[subset] += 1
                    #     if subset==(0,):
                    #         print(accum_scores[subset]["old_arch_inter_score"])
                    else:
                        pass
                    #     if subset==(0,):
                    #         print("000")

        #take the mean instead of sum
        for subset in accum_scores:
            for name in accum_scores[subset]:
                # accum_scores[subset][name] /= K*(K-1)/2
                accum_scores[subset][name] /= valid_counts[subset]

        return accum_scores





class JamMaskedArchipelago():
    
    # def __init__(self, masked_model):
    def __init__(self, trained_model, fid_hypers):

        self.masked_model = trained_model

        if True:
            assert fid_hypers.is_masked_model
            self.masking_style = fid_hypers.masking_style
            self.output_type = fid_hypers.output_type

            self.score_type_name = fid_hypers.score_type_name
            self.inc_rem_pel_list = fid_hypers.inc_rem_pel_list
            self.grouped_features_dict = fid_hypers.grouped_features_dict



        self.cached_results = None #TODO TODO





        self.batch_size = 20
        self.verbose = True
        self.verbose = False
        self.output_indices = 0



    def get_batched_tuple_predictions(self, inp_tensor, inp_mask_tensor, bs=None):


        if bs==None: 
            bs=16 #default batch size
            bs=256 
            val_N = inp_tensor.shape[0]
            bs=val_N 
        net_device = next(self.masked_model.parameters()).device #there is probably a better way (assuming one device rn)
        

        
        mask_N = inp_mask_tensor.shape[0]
        out_logits_list = []
        for mn in range(mask_N):
        
            batches = (inp_tensor.shape[0] // bs) + int(bool(inp_tensor.shape[0] % bs))
            inp_mask_mask_tensor = inp_mask_tensor[mn][None].repeat(bs,1).float().to(net_device)
            for b in range(batches):
                up_bsb = bs*b+bs
                if b+1==batches:
                    if inp_tensor.shape[0] % bs != 0:
                        up_bsb = inp_tensor.shape[0]
                        inp_mask_mask_tensor=inp_mask_mask_tensor[:(up_bsb%bs)]

                #TODO: is net_device here efficient? what about float?
                out_logits = self.masked_model( (inp_tensor[bs*b:up_bsb].to(net_device),inp_mask_mask_tensor ) ) 
                out_logits = out_logits[0]+out_logits[1]
                out_logits = out_logits.cpu().detach()
                out_logits_list.append(out_logits)

        final_out_logits = torch.concatenate( out_logits_list, dim=0)
        final_out_logits=final_out_logits.reshape(mask_N,val_N,-1)
        return final_out_logits



    # def get_layerwise_archipelago_scores(self, valX, AGG_K, current_frontier):
    def get_explanation_scores(self, valX, AGG_K, current_frontier):
        # grouped_features_dict = {"D": D, "D0": D} #TODO: this is invalid right now
        # grouped_features_dict = self.grouped_features_dict

        # the_archipelago_scores = {}
        # # arch_tensor, _ = self.get_archipelago_for_candidates(
        # #     current_frontier, 
        # #     valX,
        # #     grouped_features_dict
        # # )
        # arch_tensor, _ = self.get_archipelago_for_candidates(
        #     current_frontier, 
        #     valX
        # )
        # for idx, inter in enumerate(current_frontier):
        #     inc_score = arch_tensor[idx, 0].mean()
        #     rem_score = arch_tensor[idx, 1].mean()
        #     arch_score = float(abs(inc_score - rem_score))
        #     #TODO: NOT HERE
        #     score_type_name = "archipelago"
        #     score_type_name = "old_arch_inter_score" #uhhhh.. unclear.. maybe rahils fault though
        #     the_archipelago_scores[inter] = {score_type_name: arch_score}
        #     # print(f"Interaction {inter}: inc_score={inc_score}, rem_score={rem_score}, arch_score={arch_score}")

        # return the_archipelago_scores


        archipelago_tensor, semitruth  = self.get_archipelago_for_candidates(
            current_frontier, 
            valX,
            # grouped_features_dict
        )
        print('archipelago_tensor',archipelago_tensor.shape) #NOTE: just remove?
        print('semitruth',semitruth.shape) 
        
        the_sobol_cov_archipelago_scores = {}
        for cc2,cand in enumerate(current_frontier):
            the_sobol_cov_archipelago_scores[cand] = {}
        for cc2,cand in enumerate(current_frontier):
            inc_score=archipelago_tensor[cc2,0]
            rem_score=archipelago_tensor[cc2,1]
            archipelago_score = 0.5*inc_score + 0.5*rem_score
            
            inc_cov = np.cov(inc_score.T,semitruth.T)
            rem_cov = np.cov(rem_score.T,semitruth.T)
            pel_cov = np.cov(archipelago_score.T,semitruth.T)

            the_sobol_cov_archipelago_scores[cand]['inc_inter_sobol_score']      = inc_cov[0,1]
            the_sobol_cov_archipelago_scores[cand]['rem_inter_sobol_score']      = rem_cov[0,1]
            the_sobol_cov_archipelago_scores[cand]['new_arch_inter_sobol_score'] = pel_cov[0,1]
        return the_sobol_cov_archipelago_scores

    # def get_batchwise_archipelago_scores(self, valX, current_frontier):
    # def get_batchwise_archipelago_scores(self, valX, AGG_K, current_frontier):

    #     archipelago_tensor, semitruth  = self.get_archipelago_for_candidates(
    #         current_frontier, 
    #         valX,
    #         # grouped_features_dict
    #     )
    #     print('archipelago_tensor',archipelago_tensor.shape) #NOTE: just remove?
    #     print('semitruth',semitruth.shape) 
        
    #     the_sobol_cov_archipelago_scores = {}
    #     for cc2,cand in enumerate(current_frontier):
    #         the_sobol_cov_archipelago_scores[cand] = {}
    #     for cc2,cand in enumerate(current_frontier):
    #         inc_score=archipelago_tensor[cc2,0]
    #         rem_score=archipelago_tensor[cc2,1]
    #         archipelago_score = 0.5*inc_score + 0.5*rem_score
            
    #         inc_cov = np.cov(inc_score.T,semitruth.T)
    #         rem_cov = np.cov(rem_score.T,semitruth.T)
    #         pel_cov = np.cov(archipelago_score.T,semitruth.T)

    #         the_sobol_cov_archipelago_scores[cand]['inc_inter_sobol_score']      = inc_cov[0,1]
    #         the_sobol_cov_archipelago_scores[cand]['rem_inter_sobol_score']      = rem_cov[0,1]
    #         the_sobol_cov_archipelago_scores[cand]['new_arch_inter_sobol_score'] = pel_cov[0,1]
    #     return the_sobol_cov_archipelago_scores


    def build_archipelago_masks_from_candidates(self, candidates, grouped_features_dict): #TODO: this does not need to be a self thing and is a pytorch helper (it seems)

        D=grouped_features_dict["D"]
        D0=grouped_features_dict["D0"]
        C=len(candidates)
        tuple_dict = {}
        tuple_tensor_list = []
        for cc,cand in enumerate(candidates):
            powerset_list = powerset(cand)
        
            for subset in powerset_list:
                if subset not in tuple_dict:
                    tuple_dict[subset] = len(tuple_tensor_list)
                    
                    new_tensor = torch.zeros(D,dtype=bool)
                    for i in subset:
                        if D==D0:
                            new_tensor[i] = 1
                        else:
                            for ii in grouped_features_dict[i]:
                                new_tensor[ii] = 1
                    tuple_tensor_list.append(new_tensor)
        tuple_masks = torch.stack(tuple_tensor_list, dim=0)
        return tuple_masks, tuple_dict


    # def get_archipelago_for_candidates(self, candidates, inp_tensor, grouped_features_dict):
    def get_archipelago_for_candidates(self, candidates, inp_tensor):
        CC=len(candidates)
        N=len(inp_tensor)
        archipelago_tensor = np.zeros((CC,2,N)) #ought to be smaller since CxN could be large
        # C = grouped_features_dict["C"]
        # archipelago_tensor = np.zeros((CC,2,N,C))
        scores_arr = np.zeros((CC,7))
        tuple_masks, tuple_location_dict = self.build_archipelago_masks_from_candidates(candidates, self.grouped_features_dict)
        tuple_inc_output_logits = self.get_batched_tuple_predictions(inp_tensor,tuple_masks)#,self.masked_model)
        tuple_rem_output_logits = self.get_batched_tuple_predictions(inp_tensor,~tuple_masks)#,self.masked_model)
        

        for cc,cand in enumerate(candidates):
            powerset_list = powerset(cand)
            inc_score = 0.0
            rem_score = 0.0
            
            for subset in powerset_list:
                if (len(subset)-len(cand))%2==0:
                    inc_score += tuple_inc_output_logits[tuple_location_dict[subset]]
                elif (len(subset)-len(cand))%2==1:
                    inc_score -= tuple_inc_output_logits[tuple_location_dict[subset]]
                if (len(subset))%2==0:
                    rem_score += tuple_rem_output_logits[tuple_location_dict[subset]]
                elif (len(subset))%2==1:
                    rem_score -= tuple_rem_output_logits[tuple_location_dict[subset]]
            inc_score=inc_score[:,0]
            rem_score=rem_score[:,0]
            #archipelago_score = 0.5*inc_score + 0.5*rem_score
            archipelago_tensor[cc,0] = inc_score
            archipelago_tensor[cc,1] = rem_score
            #archipelago_tensor[cc,2] = archipelago_score
        semitruth = tuple_rem_output_logits[tuple_location_dict[()]]#[:,0]
        return archipelago_tensor, semitruth