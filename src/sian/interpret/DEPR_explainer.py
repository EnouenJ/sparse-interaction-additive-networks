import numpy as np

#NOTE: depricated right now in favor of "explainer2", still need to make sure there are not lingering bugs because of this

#from Michael Tsang's Archipelago implementation
#https://arxiv.org/abs/2006.10965

class Explainer:
    
    def __init__(self, model, target=None, baseline=None, data_xformer=None, output_indices=0, batch_size=20, verbose=False):
                
        target, baseline = self.arg_checks(target, baseline, data_xformer)
        
        self.model = model
        self.target = np.squeeze(target)
        self.baseline = np.squeeze(baseline)
        self.data_xformer = data_xformer
        self.output_indices = output_indices
        self.batch_size = batch_size
        self.verbose = verbose
    
    
    def arg_checks(self, target, baseline, data_xformer):
        if ( (target is None) and (data_xformer is None) ):
            raise ValueError("Either target or data xformer must be defined")

        if target is not None and baseline is None: 
            raise ValueError("If target is defined, the baseline must also defined")

        if data_xformer is not None and target is None:
            target = np.ones( data_xformer.num_features ).astype(bool)
            baseline = np.zeros( data_xformer.num_features ).astype(bool)
        return target, baseline
    
    def verbose_iterable(self, iterable):
        if self.verbose:
            from tqdm import tqdm
            return tqdm(iterable)
        else:
            return iterable
        
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
        

class Archipelago(Explainer):
    
    def __init__(self, model, target=None, baseline=None, data_xformer=None, output_indices=0, batch_size=20, verbose=False):
        Explainer.__init__(self, model, target, baseline, data_xformer, output_indices, batch_size, verbose)
        self.inter_sets = None
        self.main_effects = None
        
    
    def archattribute(self, set_indices):
        """
            Gets archipelago attributions of index sets
        """
        if not set_indices:
            return dict()
        scores = self.batch_set_inference(set_indices, self.baseline, self.target, include_context=True)
        set_scores = scores["scores"]
        baseline_score = scores["context_score"]
        for index_tuple in set_scores:
            set_scores[index_tuple] -= baseline_score
        return set_scores


    def archdetect(self, get_main_effects=True, get_pairwise_effects=True, single_context=False, weights = [0.5,0.5]):
        """
            Detects interactions and sorts them
            Optional: gets archipelago main effects and/or pairwise effects from function reuse
            "Effects" are archattribute scores
        """
        search_a = self.search_feature_sets(self.baseline, self.target, get_main_effects=get_main_effects, get_pairwise_effects=get_pairwise_effects)
        inter_a = search_a["interactions"]
        
        # notice that target and baseline have swapped places in the arg list
        search_b = self.search_feature_sets(self.target, self.baseline)
        inter_b = search_b["interactions"]

        inter_strengths = {}
        for pair in inter_a:
            if single_context:
                inter_strengths[pair] = inter_b[pair]**2
            else:
                inter_strengths[pair] = weights[1]*inter_a[pair]**2 + weights[0]*inter_b[pair]**2
        sorted_scores = sorted(inter_strengths.items(), key=score_sorter_key_fn(self.output_indices))
        
        output = {"interactions": sorted_scores}
        for key in search_a:
            if key not in output:
                output[key] = search_a[key]
        output['derivatives'] = inter_a 
        return output
    
    
    def explain(self, top_k=None, separate_effects=False):
        if (self.inter_sets is None) or (self.main_effects is None):
            detection_dict = self.archdetect(get_pairwise_effects = False)
            inter_strengths = detection_dict["interactions"]
            self.main_effects = detection_dict["main_effects"]
            self.inter_sets,_ = zip(*inter_strengths)

        if isinstance(top_k, int):
            thresholded_inter_sets = self.inter_sets[:top_k]
        elif top_k is None:
            thresholded_inter_sets = self.inter_sets
        else:
            raise ValueError("top_k must be int or None")
            
        inter_sets_merged = merge_overlapping_sets(thresholded_inter_sets)
        inter_effects = self.archattribute(inter_sets_merged)
        
        if separate_effects:
            return inter_effects, self.main_effects
        
        merged_indices = merge_overlapping_sets(set(self.main_effects.keys()) | set(inter_effects.keys()))
        merged_explanation = dict()
        for s in merged_indices:
            if s in inter_effects:
                merged_explanation[s] = inter_effects[s]
            elif s[0] in self.main_effects:
                assert(len(s)==1)
                merged_explanation[s] = self.main_effects[s[0]]
            else:
                raise ValueError("Error: index should have been in either main_effects or inter_effects")
        return merged_explanation


    def search_feature_sets(self, context, insertion_target, get_interactions=True, get_main_effects=False, get_pairwise_effects=False):
        """
            Gets optional pairwise interaction strengths, optional main effects, and optional pairwise effects
            "Effects" are archattribute scores
            All three options are combined to reuse function calls
        """
        num_feats = context.size
        idv_indices = [(i,) for i in range(num_feats)]

        preds = self.batch_set_inference(idv_indices, context, insertion_target, include_context=True)
        idv_scores, context_score = preds["scores"], preds["context_score"]

        output = {}

        if get_interactions:
            pair_indices = []
            pairwise_effects = {}
            for i in range(num_feats):
                for j in range(i+1, num_feats):
                    pair_indices.append((i,j))

            preds = self.batch_set_inference(pair_indices, context, insertion_target)
            pair_scores = preds["scores"]
            
            inter_scores = {}
            for i, j in pair_indices:
                # interaction detection
                ell_i = np.abs(context[i].item() - insertion_target[i].item())
                ell_j = np.abs(context[j].item() - insertion_target[j].item())
                inter_scores[(i,j)] = 1/(ell_i*ell_j) * ( context_score - idv_scores[(i,)] - idv_scores[(j,)] + pair_scores[(i,j)] )
                
                if get_pairwise_effects: # leverage existing function calls to compute pairwise effects
                    pairwise_effects[(i,j)] = pair_scores[(i,j)] - context_score
                    
            output["interactions"] = inter_scores
            if get_pairwise_effects:
                output["pairwise_effects"] = pairwise_effects

        if get_main_effects: # leverage existing function calls to compute main effects
            main_effects = {}
            for i in idv_scores:
                main_effects[i[0]] = idv_scores[i] - context_score
            output["main_effects"] = main_effects
            
        return output

    



    def archdetect_1D(self, get_main_effects=True, single_context=False):
        weights = [0.5,0.5]
        """
            Detects interactions and sorts them
            Optional: gets archipelago main effects and/or pairwise effects from function reuse
            "Effects" are archattribute scores
        """
        search_a = self.search_feature_sets_1D(self.baseline, self.target, get_main_effects=get_main_effects)
        inter_a = search_a["interactions"]
        
        # notice that target and baseline have swapped places in the arg list
        search_b = self.search_feature_sets_1D(self.target, self.baseline)
        inter_b = search_b["interactions"]

        inter_strengths = {}
        for single in inter_a:
            if single_context:
                inter_strengths[single] = inter_b[single]**2
            else:
                inter_strengths[single] = weights[1]*inter_a[single]**2 + weights[0]*inter_b[single]**2
        sorted_scores = sorted(inter_strengths.items(), key=score_sorter_key_fn(self.output_indices))

        output = {"interactions": sorted_scores}
        for key in search_a:
            if key not in output:
                output[key] = search_a[key]
        output['derivatives'] = inter_a 
        return output
    
    def search_feature_sets_1D(self, context, insertion_target, get_main_effects=False, get_interactions=True):
        """
            Gets optional pairwise interaction strengths, optional main effects, and optional pairwise effects
            "Effects" are archattribute scores
            All three options are combined to reuse function calls
        """
        num_feats = context.size
        idv_indices = [(i,) for i in range(num_feats)]

        preds = self.batch_set_inference(idv_indices, context, insertion_target, include_context=True)
        idv_scores, context_score = preds["scores"], preds["context_score"]

        output = {}

        if get_interactions:

            inter_scores = {}
            for i_tup in idv_indices:
                i=i_tup[0]
                ell_i = np.abs(context[i].item() - insertion_target[i].item())
                inter_scores[(i,)] = 1/(ell_i) * ( context_score - idv_scores[(i,)] )
                
                if get_main_effects: # leverage existing function calls to compute main effects
                    main_effects = {}
                    for i_tup in idv_scores:
                        main_effects[i_tup] = idv_scores[i_tup] - context_score
                    output["main_effects"] = main_effects
            output["interactions"] = inter_scores


        return output

    def archdetect_2D(self, get_main_effects=True, get_pairwise_effects=True, single_context=False):
        weights = [0.5,0.5]
        """
            Detects interactions and sorts them
            Optional: gets archipelago main effects and/or pairwise effects from function reuse
            "Effects" are archattribute scores
        """
        search_a = self.search_feature_sets_2D(self.baseline, self.target, get_main_effects=get_main_effects, get_pairwise_effects=get_pairwise_effects)
        inter_a = search_a["interactions"]
        
        # notice that target and baseline have swapped places in the arg list
        search_b = self.search_feature_sets_2D(self.target, self.baseline)
        inter_b = search_b["interactions"]

        inter_strengths = {}
        for tuple in inter_a:
            if single_context:
                inter_strengths[tuple] = inter_b[tuple]**2
            else:
                inter_strengths[tuple] = weights[1]*inter_a[tuple]**2 + weights[0]*inter_b[tuple]**2
        sorted_scores = sorted(inter_strengths.items(), key=score_sorter_key_fn(self.output_indices))
        
        output = {"interactions": sorted_scores}
        for key in search_a:
            if key not in output:
                output[key] = search_a[key]
        output['derivatives'] = inter_a
        return output
    
    def search_feature_sets_2D(self, context, insertion_target, get_interactions=True, get_main_effects=False, get_pairwise_effects=False):
        """
            Gets optional pairwise interaction strengths, optional main effects, and optional pairwise effects
            "Effects" are archattribute scores
            All three options are combined to reuse function calls
        """
        num_feats = context.size
        idv_indices = [(i,) for i in range(num_feats)]

        preds = self.batch_set_inference(idv_indices, context, insertion_target, include_context=True)
        idv_scores, context_score = preds["scores"], preds["context_score"]

        output = {}

        if get_interactions:
            pair_indices = []
            pairwise_effects = {}
            for i in range(num_feats):
                for j in range(i+1, num_feats):
                    pair_indices.append((i,j))

            preds = self.batch_set_inference(pair_indices, context, insertion_target)
            pair_scores = preds["scores"]
            
            inter_scores = {}
            for i, j in pair_indices:
                # interaction detection
                ell_i = np.abs(context[i].item() - insertion_target[i].item())
                ell_j = np.abs(context[j].item() - insertion_target[j].item())
                inter_scores[(i,j)] = 1/(ell_i*ell_j) * ( context_score - idv_scores[(i,)] - idv_scores[(j,)] + pair_scores[(i,j)] )
                
                if get_pairwise_effects: # leverage existing function calls to compute pairwise effects
                    pairwise_effects[(i,j)] = pair_scores[(i,j)] - context_score

                
            output["interactions"] = inter_scores
            if get_pairwise_effects:
                output["pairwise_effects"] = pairwise_effects

        if get_main_effects: # leverage existing function calls to compute main effects
            main_effects = {}
            for i in idv_scores:
                main_effects[i[0]] = idv_scores[i] - context_score
            output["main_effects"] = main_effects
            
        return output

    def archdetect_kD(self, interactions_list, get_higher_effects=True, only_necessary_background=True, single_context=False, get_contrastive_validities = True):
        weights = [0.5,0.5]
        """
            Detects interactions and sorts them
            Optional: gets archipelago main effects and/or pairwise effects from function reuse
            "Effects" are archattribute scores
        """
        search_a = self.search_feature_sets_kD(self.baseline, self.target, interactions_list, only_necessary_background=only_necessary_background, get_higher_effects=get_higher_effects)
        inter_a = search_a["interactions"]
        
        # notice that target and baseline have swapped places in the arg list
        search_b = self.search_feature_sets_kD(self.target, self.baseline, interactions_list, only_necessary_background=only_necessary_background)
        inter_b = search_b["interactions"]

        inter_strengths = {}
        for pair in inter_a:
            if single_context:
                inter_strengths[pair] = inter_b[pair]**2
            else:
                inter_strengths[pair] = weights[1]*inter_a[pair]**2 + weights[0]*inter_b[pair]**2
        sorted_scores = sorted(inter_strengths.items(), key=score_sorter_key_fn(self.output_indices))
        
        output = {"interactions": sorted_scores}
        for key in search_a:
            if key not in output:
                output[key] = search_a[key]
        output['derivatives'] = inter_a

        return output
    
    def search_feature_sets_kD(self, context, insertion_target, interactions_list, only_necessary_background=True, get_interactions=True, get_higher_effects=False):
        """
            Gets optional pairwise interaction strengths, optional main effects, and optional pairwise effects
            "Effects" are archattribute scores
            All three options are combined to reuse function calls
        """
        #HIGHER ORDER THAN THREE
        inters_subset_list = []
        for interaction_tuple in interactions_list:
            pset_inter = powerset(interaction_tuple)
            for subset_tuple in pset_inter:
                if not subset_tuple in inters_subset_list:
                    inters_subset_list.append(subset_tuple)


        if not only_necessary_background:
            num_feats = context.size
            idv_indices = [(i,) for i in range(num_feats)]

            preds = self.batch_set_inference(idv_indices, context, insertion_target, include_context=True)
            idv_scores, context_score = preds["scores"], preds["context_score"]


            pair_indices = []
            pairwise_effects = {}
            for i in range(num_feats):
                for j in range(i+1, num_feats):
                    pair_indices.append((i,j))

            preds = self.batch_set_inference(pair_indices, context, insertion_target)
            pair_scores = preds["scores"]

            
        output = {}
        if get_interactions:    
            higher_scores = self.batch_set_inference(inters_subset_list, context, insertion_target)["scores"]

            idv_scores  = {}
            pair_scores = {}
            for subset_tuple in inters_subset_list:
                L=len(subset_tuple)
                if L==1:
                    idv_scores[subset_tuple]  = higher_scores[subset_tuple]
                elif L==2:
                    pair_scores[subset_tuple] = higher_scores[subset_tuple]
                elif L==0:
                    context_score             = higher_scores[subset_tuple]

            
            inter_scores = {}
            if not only_necessary_background:
                for i, j in pair_indices:
                    
                    # interaction detection
                    ell_i = np.abs(context[i].item() - insertion_target[i].item())
                    ell_j = np.abs(context[j].item() - insertion_target[j].item())
                    inter_scores[(i,j)] = 1/(ell_i*ell_j) * ( context_score - idv_scores[(i,)] - idv_scores[(j,)] + pair_scores[(i,j)] )
                    
                    #if get_pairwise_effects: # leverage existing function calls to compute pairwise effects
                    #    pairwise_effects[(i,j)] = pair_scores[(i,j)] - context_score

                    
                    
            #for interaction_tuple in interactions_list:
            for interaction_tuple in inters_subset_list:
                k=len(interaction_tuple)
                score = 0
                pset_inter = powerset(interaction_tuple)
                for subset_tuple in pset_inter:
                    if len(subset_tuple)==0:
                        score += context_score
                    elif len(subset_tuple)==1:
                        score -= idv_scores[subset_tuple]
                    elif len(subset_tuple)==2:
                        score += pair_scores[subset_tuple]
                    elif (len(subset_tuple)%2)==1:
                        score -= higher_scores[subset_tuple]
                    elif (len(subset_tuple)%2)==0:
                        score += higher_scores[subset_tuple]
                
                for kk in range(k):
                    i = interaction_tuple[kk]
                    score /= np.abs(context[i].item() - insertion_target[i].item())
                inter_scores[interaction_tuple] = score
                #if get_higher_effects and not only_necessary_background: 
                #    pairwise_effects[interaction_tuple] = inter_scores[interaction_tuple] - context_score
                    
                    
                    
                
            output["interactions"] = inter_scores

        if not only_necessary_background: # leverage existing function calls to compute main effects
            main_effects = {}
            for i in idv_scores:
                main_effects[i[0]] = idv_scores[i] - context_score
            output["main_effects"] = main_effects
            
        return output





from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    

def merge_overlapping_sets(lsts, output_ints=False):
    """Check each number in our arrays only once, merging when we find
    a number we have seen before.
    
    O(N) mergelists5 solution from https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    """
    
    def locatebin(bins, n):
        """
        Find the bin where list n has ended up: Follow bin references until
        we find a bin that has not moved.
        """
        while bins[n] != n:
            n = bins[n]
        return n
    
    data = []
    for lst in lsts:
        if type(lst) not in {list, set, tuple}:
            lst = {lst}
        data.append(set(lst))

    bins = list(range(len(data)))  # Initialize each bin[n] == n
    nums = dict()

    sets = []
    for lst in lsts:
        if type(lst) not in {list, set, tuple}:
            lst = {lst}
        sets.append(set(lst))
    
    for r, row in enumerate(data):
        for num in row:
            if num not in nums:
                # New number: tag it with a pointer to this row's bin
                nums[num] = r
                continue
            else:
                dest = locatebin(bins, nums[num])
                if dest == r:
                    continue # already in the same bin

                if dest > r:
                    dest, r = r, dest   # always merge into the smallest bin

                data[dest].update(data[r]) 
                data[r] = None
                # Update our indices to reflect the move
                bins[r] = dest
                r = dest 
    
    # take single values out of sets
    output = []
    for s in data:
        if s:
            if output_ints and len(s) == 1:
                output.append(next(iter(s)))
            else:
                output.append(tuple(sorted(s)))
    
    return output

def score_sorter_key_fn(output_indices):
    if output_indices == 0:
        return lambda kv: -kv[1]
    else:
        return lambda kv: -kv[1][0]

    
