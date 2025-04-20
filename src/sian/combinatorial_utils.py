from itertools import chain, combinations
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def oneless(iterable): #subsets one element less
    s = list(iterable)
    return combinations(s, len(s)-1)
def issubset(sub1,sub2):
    issub = True
    for i1 in sub1:
        if i1 not in sub2:
            return False
    return issub


def create_singles(D):
    return 

#'n' = highest order in the current set
def constructHigherInteractions(interactions_list, n, tau=0.5):
    next_list = []
    possible_dict = {}
    
    pppp = len(interactions_list)
    for i in range(pppp-1):
        tuple1 = interactions_list[i]
        if len(tuple1)==n:
            for j in range(i+1,pppp):
                tuple2 = interactions_list[j]
                if len(tuple2)==n:

                    tuple_combined = list(tuple1) #TODO: IMPLICITLY ASSUMING THAT THERE IS AT LEAST TWO!!!! (i.e. tau*(n+1) >= 2, which isnt always the case)
                    for thing in tuple2:
                        if not thing in tuple_combined:
                            tuple_combined.append(thing)

                    if len(tuple_combined)==(n+1):
                        tuple_combined = tuple(sorted(tuple_combined))

                        if not tuple_combined in possible_dict:
                            score = 0
                            #oneless_tuples = list(powerset(tuple_combined))
                            oneless_tuples = list(oneless(tuple_combined))
                            for subset in oneless_tuples:
                                if subset in interactions_list:
                                    score += 1
                            #print(tuple_combined,'\t',score)
                            possible_dict[tuple_combined] = score
                            
                            if score/(n+1) >= tau:
                                next_list.append(tuple_combined)

    return next_list


def build_strict_frontier(included_tuples,K,D):
    strict_frontier = []
    for n in range(K):
        if n==0:
            next_list = [(i,) for i in range(D)]
        else:
            next_list = constructHigherInteractions(included_tuples,n,tau=1.0)
            
        print('n',n,'\tnext_list',next_list)
        for thing in next_list:
            if not thing in included_tuples:
                strict_frontier.append(thing)
                
    return strict_frontier
    
def build_nonincluded_subsets_of_frontier_and_candidates(frontier,candidates):
    needed_list = []
    for cc,cand in enumerate(candidates):
        powerset_list = powerset(cand)
        for subset in powerset_list:
            if subset not in frontier:
                if subset not in needed_list:
                    needed_list.append(subset)
    return needed_list
    