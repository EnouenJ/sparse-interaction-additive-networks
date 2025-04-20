


# from sian.fis import powerset
# from sian.fis import constructHigherInteractions
# from sian.interpret import fancy_plot_archipelago_covariances #same deal (circular import)
# from sian.interpret.plotting import fancy_plot_archipelago_covariances
from sian import powerset
from sian import constructHigherInteractions #circular import 


from sian.interpret import JamMaskedArchipelago
from sian.interpret import JamArchipelago
from sian.interpret.basic_wrappers import MixedModelWrapperTorch #TODO: remove this probably

import time
import json
import numpy as np

from sian.interpret import fancy_plot_archipelago_covariances






class FIS_Hyperparameters():
    def __init__(self):
        self.explainer = None

    def add_the_explainer(self, explainer):
        self.explainer = explainer


class batchwise_FIS_Hyperparameters(FIS_Hyperparameters):
    def __init__(self, MAX_K, tau_thresholds, number_of_rounds, interactions_per_round,
                       explainer,
                    #    fis_masking_style, score_type_name, 
                    #    input_tensor, output_type, #NOTE: REVISIT NECESSITY
                    #    input_tensor, #NOTE: REVISIT NECESSITY
                    #    AGG_K=100,
                       tuples_initialization=None,pick_underlings=False,fill_underlings=False,PLOTTING=True):
        
        self.FIS_type = 'batchwise'
        self.explainer = explainer #TODO: add it like this
        
        
        self.tau_thresholds   = tau_thresholds
        self.number_of_rounds = number_of_rounds
        self.interactions_per_round = interactions_per_round

        # self.input_tensor = input_tensor
        # self.output_type = output_type

        self.tuples_initialization = tuples_initialization
        self.pick_underlings = pick_underlings
        self.fill_underlings = fill_underlings

        self.PLOTTING = PLOTTING
    
    
        # ['zero_baseline','triangle_marginal',     'masking_based',]
        # self.fis_masking_style = fis_masking_style #NOTE: moved inside explainer
        
        # ['old_arch_inter_score', 'inc_inter_score', 'rem_inter_score', 'new_arch_inter_score',]
        # self.score_type_name = score_type_name #NOTE: moved inside explainer
        
        #GAM maximum interaction size [1,2,3,4,5,...]
        self.MAX_K = MAX_K
        
        # self.AGG_K = AGG_K
        self.FRONTIER_MAX_SIZE = 1000 #NOTE: doesnt get used right now I think
        
        pass
    
    def get_json(self):
        return { #TODO: acutally write this part
            "FIS_type" : self.FIS_type,
            "MAX_K" : self.MAX_K,
            "tau_thresholds" : self.tau_thresholds,
            "number_of_rounds" : self.number_of_rounds,
            "interactions_per_round" : self.interactions_per_round,


            # "fis_masking_style" : self.fis_masking_style,
            # "score_type_name" : self.score_type_name,
            # "MAX_K" : self.MAX_K,
            # "AGG_K" : self.AGG_K,
            # "FRONTIER_MAX_SIZE" : self.FRONTIER_MAX_SIZE,
        }
    
    def load_json(self, json_dict):
        raise NotImplementedError("sorry :)")
    
    def save_as_json(self, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            json_thing = self.get_json()
            json.dump(json_thing, f, ensure_ascii=False, indent=4)




class layerwise_FIS_Hyperparameters(FIS_Hyperparameters):
    def __init__(self, MAX_K, tau_thresholds, theta_thresholds, 
                       explainer,
                    #    input_data, output_type,
                    #    input_data,
                    #    fis_masking_style, score_type_name, 
                       AGG_K=100,theta_percentile_mode=False,FRONTIER_MAX_SIZE=1000):
        
        self.FIS_type = 'layerwise'
        self.explainer = explainer #TODO: add it like this
        
        # self.input_data = input_data
        # assert output_type=="regression"
        
        self.theta_thresholds = theta_thresholds
        self.tau_thresholds   = tau_thresholds
    
    
        # ['zero_baseline','triangle_marginal',     'masking_based',]
        # self.fis_masking_style = fis_masking_style #NOTE: moved inside explainer
        
        # ['old_arch_inter_score', 'inc_inter_score', 'rem_inter_score', 'new_arch_inter_score',]
        # self.score_type_name = score_type_name #NOTE: moved inside explainer
        
        #GAM maximum interaction size [1,2,3,4,5,...]
        self.MAX_K = MAX_K
        
        # self.AGG_K = AGG_K
        self.theta_percentile_mode = theta_percentile_mode
        self.FRONTIER_MAX_SIZE = FRONTIER_MAX_SIZE
        
        pass
    
    def get_json(self):
        return {
            "FIS_type" : self.FIS_type,
            "MAX_K" : self.MAX_K,
            "theta_thresholds" : self.theta_thresholds,
            "tau_thresholds" : self.tau_thresholds,

            # "fis_masking_style" : self.fis_masking_style,
            # "score_type_name" : self.score_type_name,
            # "AGG_K" : self.AGG_K,
            "FRONTIER_MAX_SIZE" : self.FRONTIER_MAX_SIZE,
            "theta_percentile_mode" : self.theta_percentile_mode,
        }
    
    def load_json(self, json_dict):
        raise NotImplementedError("sorry :)")
    
    def save_as_json(self, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            json_thing = self.get_json()
            json.dump(json_thing, f, ensure_ascii=False, indent=4)


















#NOTE: add support for the classification based ones not using Sobol exactly (maybe not worth it for a bit?)
# def check_candidate_covariance_archipelago_more_general_v3(candidates, masked_net, inp_tensor=None, frontier=None,
#                                                         reaccumulate_underlings=False,
#                                                         grouped_features_dict=None,
#                                                         fid_mode='archipelago', PLOTTING=True):





# def run_a_single_round_of_batchwise_FIS_for_regression(candidates, archipelago_object, inp_tensor=None, frontier=None,
def run_a_single_round_of_batchwise_FIS_for_regression(candidates, archipelago_object, inp_tensor=None, AGG_K=100, frontier=None,
                                                        reaccumulate_underlings=False, 
                                                        # grouped_features_dict=None,
                                                        output_mode='regression',fid_mode='archipelago', PLOTTING=True):

    if inp_tensor is None: #NOTE: this is broken as a script
        inp_tensor = val_tensor
        print('...assuming you want to run on \'val_tensor\'')
        

        
    needed_list = candidates
    if reaccumulate_underlings:
        assert (frontier is not None), "frontier is None, but you are trying to reaccumulate non-frontier"
        needed_list = build_nonincluded_subsets_of_frontier_and_candidates(frontier, candidates)
        
    C2=len(needed_list)
    scores_arr2 = np.zeros((C2,7))
    CC=len(candidates)
    scores_arr = np.zeros((CC,7))


    if output_mode=="regression":           
        AGG_K = 100 #TODO: PULL THIS OUT SOMEHOW
        the_archipelago_scores = archipelago_object.get_explanation_scores(inp_tensor, AGG_K, needed_list)

    else:
        raise NotImplentedError("didnt copy paste this because of the confusion around entropy-Archipelago")



    if output_mode=="classification":    
        C = self.grouped_features_dict["C"] #TODO: not implemented, and havent check if change to 'self.grouped_features_dict' is well supported

    for cc2,cand in enumerate(needed_list):

        if output_mode=="regression": 
            for ff,score_name in enumerate(archipelago_object.inc_rem_pel_list):
                scores_arr2[cc2,ff] = the_archipelago_scores[cand][score_name]



        elif output_mode=="classification": #TODO: probably more interesting things can be done here (from updated KL perspective)
            raise NotImplementedError("never ported from doing this")
            scores_arr2[cc2,0] = np.mean(inc_score)
            scores_arr2[cc2,1] = np.mean(rem_score)
            scores_arr2[cc2,2] = np.mean(archipelago_score)
        else:
            raise Exception("not supported \'output_mode\'="+str(output_mode))

    
    if not reaccumulate_underlings:
        scores_arr=scores_arr2
    else:
        for cc,cand in enumerate(candidates):
            powerset_list = powerset(cand)
            for subset in powerset_list:
                if subset not in frontier:
                    cc2 = needed_list.index( subset )
                    scores_arr[cc] += scores_arr2[cc2]
                
    if PLOTTING:
        #plot_archipelago_covariates(candidates,scores_arr[:,:3],)
        fancy_plot_archipelago_covariances(candidates,scores_arr[:,:3],)
    
    return scores_arr




# def batchwise_feature_interaction_selection_algorithm(config_hypers):
def batchwise_feature_interaction_selection_algorithm(config_hypers, inp_tensor, AGG_K): #pulling these back out
    
    tau_thresholds = config_hypers.tau_thresholds
    MAX_K = config_hypers.MAX_K
    TOP_K = config_hypers.interactions_per_round #TODO: RENAME INSIDE HERE
    num_rounds = config_hypers.number_of_rounds

    tuples_initialization = config_hypers.tuples_initialization
    PICK_UNDERLINGS = config_hypers.pick_underlings
    FILL_UNDERLINGS = config_hypers.fill_underlings
    PLOTTING = config_hypers.PLOTTING

    
    jam_arch = config_hypers.explainer
    # inp_tensor = config_hypers.input_tensor
    ###output_mode = config_hypers.output_type
    output_mode = jam_arch.output_type


    
    # fid_mode = 'archipelago'
    # fid_mode_index = ['inc', 'rem', 'archipelago'].index(fid_mode)
    fid_mode = jam_arch.score_type_name
    inc_rem_pel_list = jam_arch.inc_rem_pel_list
    fid_mode_index = inc_rem_pel_list.index(fid_mode)
    
    if tuples_initialization is None:
        D0 = jam_arch.grouped_features_dict['D0']    
        frontier = [()]
        candidates = [(i,) for i in range(D0)]
        selections = []
    else:
        frontier, candidates, selections = tuples_initialization
    
    
        
    start_time = time.time()
    for nr in range(0, num_rounds):
        print('nr', nr)

        if len(candidates)>0:
            if output_mode == "regression":
                scores_arr = run_a_single_round_of_batchwise_FIS_for_regression(
                    candidates, jam_arch, inp_tensor, AGG_K,
                    frontier=frontier + selections,
                    output_mode=output_mode,
                    reaccumulate_underlings=FILL_UNDERLINGS,
                    PLOTTING=PLOTTING,
                )
            elif output_mode == "classification":
                raise Exception("not implemented yet, but idk just make it like regression for now")
                scores_arr = check_candidate_covariance_archipelago_more_general_v4(
                    candidates, masked_net, inp_tensor,
                    frontier=frontier + selections,
                    # grouped_features_dict=grouped_features_dict,
                    reaccumulate_underlings=FILL_UNDERLINGS
                )
            else:
                raise ValueError(f"Unsupported output_mode: {output_mode}")

            inds2 = np.argsort(-scores_arr[:, fid_mode_index])
            new_selections = [candidates[ind] for ind in inds2[:TOP_K]]
            print('new_selections', new_selections)
            
            if PICK_UNDERLINGS:
                new_selections_unders = []
                for sel in new_selections:
                    sel_pow = powerset(sel)
                    for sel_und in sel_pow:
                        if not (sel_und in selections or sel_und in new_selections or sel_und in new_selections_unders):
                            new_selections_unders.append(sel_und)
                print('new_selections_unders', new_selections_unders)
                new_selections.extend(new_selections_unders)

            selections.extend(new_selections)
            print('selections', selections)

            new_cands = []
            if selections:
                n = max([len(sel) for sel in selections])
                if MAX_K is not None: #set to MAX_K if greater than MAX_K
                    n = min(n, MAX_K-1)
                for nn in range(1, n + 1):
                    tau = tau_thresholds.get(nn + 1, 1.0) #should default to strongest hierarchy to avoid extra calculations when left unspecified
                    new_precands_nn = constructHigherInteractions(selections, n=nn, tau=tau)
                    new_cands.extend(new_precands_nn)
            print('new_cands', new_cands) #NOTE: slightly repetitive with overselecting from 'constructHigherInteractions()' which doesnt account for already selected tuples

            for sel in new_selections:
                if not PICK_UNDERLINGS:
                    if sel in candidates:
                        candidates.remove(sel)
                else:
                    if sel in candidates:
                        candidates.remove(sel)
            for cand in new_cands:
                if cand not in selections and cand not in candidates:
                    candidates.append(cand)

            print('candidates', candidates)
            print(time.time() - start_time, 'seconds')
        else:
            break
        
    # return frontier, candidates, selections
    other_results = {
        "frontier" : frontier,
        "candidates" : candidates,
    }
    return selections, other_results















#from notebook for rahil -- layerwise original algorithm
# def layerwise_feature_interaction_selection_algorithm(jam_arch, valX, fis_hypers):
# def layerwise_feature_interaction_selection_algorithm(fis_hypers):
def layerwise_feature_interaction_selection_algorithm(fis_hypers, valX, AGG_K): #pulling these back out
    theta_thresholds = fis_hypers.theta_thresholds
    tau_thresholds = fis_hypers.tau_thresholds
    # AGG_K = fis_hypers.AGG_K
    MAX_K = fis_hypers.MAX_K
    
    theta_percentile_mode = fis_hypers.theta_percentile_mode
    FRONTIER_MAX_SIZE = 1000

    jam_arch = fis_hypers.explainer
    # valX = fis_hypers.input_data
    
    score_type_name = jam_arch.score_type_name
    
    raw_theta_values = {}
    D = valX.shape[1]
    D0 = jam_arch.grouped_features_dict["D0"]
    
    curr_K = 0
    current_interactions = [()]
    total_archipelago_score_time = 0
    
    while curr_K < MAX_K:
        curr_K += 1
        
        print(f"Starting iteration K={curr_K}")
        print(f"Current interactions: {current_interactions}")
        print(f"Using tau threshold: {tau_thresholds.get(curr_K, 'Not set')}")
        print(f"Using theta threshold: {theta_thresholds[curr_K]}")
        
        if curr_K == 1:
            # current_frontier = [(i,) for i in range(D)]
            current_frontier = [(i,) for i in range(D0)] #04/13/2025 @ 2:00am
        else:
            current_frontier = constructHigherInteractions(current_interactions, curr_K-1, 
                                                         tau=tau_thresholds[curr_K])
            if len(current_frontier) > FRONTIER_MAX_SIZE:
                current_frontier = current_frontier[:FRONTIER_MAX_SIZE]

        if not current_frontier:
            break

        print('checking the values for:', current_frontier)
        
        start_time = time.time()
        the_archipelago_scores = {}
        
        if True:
            # the_archipelago_scores = jam_arch.get_explanation_scores(valX, AGG_K, current_frontier)
            the_archipelago_scores = jam_arch.get_explanation_scores(valX, AGG_K, current_frontier)

            
        end_time = time.time()
        total_archipelago_score_time += (end_time - start_time)
                
        arch_scores = []
        score_dict = {}
        # print('score_type_name',score_type_name)

        for inter in the_archipelago_scores:
            arch_score = the_archipelago_scores[inter][score_type_name]
            #TODO: still causes random issues, I think it is a zero issue for the random sample having no vacation days (rare event)
            #TODO: I think the correct solution is to more directly line up this version with the 'marginal' approach

            # # print(inter,the_archipelago_scores[inter])
            # if dataset_str != "otherSource_cal_housing": 
            #     arch_score = the_archipelago_scores[inter][score_type_name] #TODO: still causes random issues, I think it is a zero issue for the random sample having no vacation days (rare event)
            #     #TODO: I think the correct solution is to more directly line up this version with the 'marginal' approach
            # else:
            #     arch_score = the_archipelago_scores[inter].get(score_type_name, 0.0)
            
            # if arch_score > 0: #TODO: REMOVED on 04/13/2025 @ 3:30am -- unclear if necessary, but fails with sobol version; hence breaking the theta_percentiles stuff
            #     log_score = np.log(arch_score)
            #     arch_scores.append(log_score)
            #     score_dict[inter] = log_score
            score_dict[inter] = arch_score
            arch_scores.append(arch_score) #NOTE: ensure this is reasonable/ I am using positive/negative values, not that I forgot to squaare
        
        # Rest of the threshold computation remains the same
        print('theta_percentile_mode',theta_percentile_mode)
        print('arch_scores',arch_scores)
        if arch_scores:
            print('theta_percentile_mode',theta_percentile_mode)
            if theta_percentile_mode:
                percentile_val = 100 * (1 - theta_thresholds[curr_K])
                percentile_threshold = np.percentile(arch_scores, percentile_val)
                raw_theta_values[curr_K] = {
                    'theta_percentile': theta_thresholds[curr_K],
                    'raw_theta_value': percentile_threshold
                }
            else:
                percentile_threshold = theta_thresholds[curr_K]
                raw_theta_values[curr_K] = {
                    'theta_percentile': percentile_threshold,
                    'raw_theta_value': percentile_threshold
                }
        else:
            percentile_threshold = float('-inf')
            raw_theta_values[curr_K] = {
                'raw_theta_value': percentile_threshold
            }
        
        new_interactions_to_add = []
        for inter, log_score in score_dict.items():
            if log_score > percentile_threshold:
                new_interactions_to_add.append(inter)
        
        if new_interactions_to_add:
            current_interactions.extend(new_interactions_to_add)

        print('now having the interaction set:', current_interactions)
    
    print(f"Total archipelago score time: {total_archipelago_score_time} seconds")
    # return current_interactions, total_archipelago_score_time, raw_theta_values
    other_results = {
        "raw_theta_values" : raw_theta_values
    }
    return current_interactions, other_results 