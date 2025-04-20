import numpy as np
import torch



import os





from sian.data.data_loader import loadDataset #NOTE: NEED TO DEPRICATE
# from sian.data.data_loader import loadLabels
from sian.data.data_loader import loadHeader


from sian.data.data_loader import preprocess_dataset
from sian.data.data_loader import load_from_preprocessed




class Final_TabularDataset():
    # def __init__(self, dataset_path, load_from_indicator="preprocessed"):
    def __init__(self, dataset_str, preproc_owner, type="realworld.tabular",
                    #    load_from_path_for_original_csv=None, save_from_path_for_my_personal_numpy=None):
                       load_dataset_path=None, save_dataset_path=None):
                       

        SAVING_PREPROCESSED_VERSION = True
        OVERWRITE_EXISTING_PROCESSED = False


        already_preprocessed = False
        header_dict = loadHeader(save_dataset_path)
        print('header_dict',header_dict)
        if header_dict is not None:
            if header_dict["dataset_id"]==dataset_str and header_dict["preproc_owner"]==preproc_owner:
                already_preprocessed = True
            else:
                if OVERWRITE_EXISTING_PROCESSED:
                    pass #just overwrite implicitly
                else:
                    raise Exception(f"existing preprocessed dataset at save location  &&  OVERWRITE_EXISTING_PROCESSED={OVERWRITE_EXISTING_PROCESSED}")


        if not already_preprocessed:
            print("PROCESSING AND LOADING")
            #maybe have this part tell us if it is an "X->Y" dataset or an "X only" dataset
            header_dict, XY_tuple, label_stuff_dict = \
            preprocess_dataset(dataset_str, preproc_owner=preproc_owner,
                               load_dataset_path=load_dataset_path, save_dataset_path=save_dataset_path,
                               SAVING_PREPROCESSED_VERSION=SAVING_PREPROCESSED_VERSION)
        else:
            print("LOADING FROM EXISTING")
            ###XY_stuff, label_stuff = load_from_preprocessed(save_dataset_path)
            header_dict, XY_tuple, label_stuff_dict = load_from_preprocessed(save_dataset_path)
            
        
        self.header_dict = header_dict
        self.label_stuff_dict = label_stuff_dict

        if True:
            trnvalX, trnvalY, tstX, tstY = XY_tuple
            if len(trnvalY.shape)==1:
                trnvalY = trnvalY[:, None] #adding fake dimension here
                tstY    = tstY[:, None] 

            self.trnvalX = trnvalX
            self.trnvalY = trnvalY
            self.tstX = tstX
            self.tstY = tstY

            self.trnX, self.valX, self.trnY, self.valY = None, None, None, None

            del XY_tuple
            del trnvalX, trnvalY, tstX, tstY


        pass
        #TODO: check if there is already something in the preprocessed folder
        #      if it is 'correct', lining up with the signature, then let it be
        #      otherwise decide to overwrite it (possibly with a warning)

        #TODO: next keep this preprocessed thing inside the class internals
        #      do the next basic things like testing split and trnval splitting


    def get_D(self):
        return self.trnvalX.shape[1]
        
    def get_C(self):
        return self.trnvalY.shape[1]

    def get_dataset_id(self):
        return self.header_dict["dataset_id"]
        
    def get_readable_labels(self):
        #print(self.label_stuff_dict)
        #return self.label_stuff_dict["readable_labels"]
        # return self.label_stuff_dict[0] #TODO: not a dictionary right now
        return self.label_stuff_dict["readable_labels"]

    # def get_datatype_labels(self):
    #     return self.label_stuff_dict[1] #TODO: not a dictionary right now
        
    def get_full_readable_labels(self):
        # return self.label_stuff_dict[2] #TODO: not a dictionary right now
        return self.label_stuff_dict["full_readable_labels"]

    def get_grouped_feature_dict(self):
        grouped_features_dict = {}
        full_readable_labels = self.get_full_readable_labels()
        # D0 = len(full_readable_labels.keys()) #WRONG, NOT EXTENDABLE
        D0 = full_readable_labels["D0"]
        D  = self.get_D()
        grouped_features_dict["D0"] = D0
        grouped_features_dict["D"]  = D
        for d in range(D0):
            feat_d_info = full_readable_labels[d]
            startdim = feat_d_info['startdim']
            numdims  = feat_d_info['numdims']
            grouped_features_dict[d] = list(range(startdim,startdim+numdims))
        return grouped_features_dict
    def get_task_type(self):
        full_readable_labels = self.get_full_readable_labels()
        return full_readable_labels["task_type"]









    def shuffle_and_split_trnval(self, trnval_shuffle_seed=None, trnval_split_percentage = .7):
        if trnval_shuffle_seed is None:
            np.random.seed(None)
            self.trnval_shuffle_seed = np.random.randint()
        else:
            self.trnval_shuffle_seed = trnval_shuffle_seed
        np.random.seed(self.trnval_shuffle_seed)
        print('trnval_shuffle_seed',trnval_shuffle_seed)
        print('self.trnval_shuffle_seed',self.trnval_shuffle_seed)

        M_NUM = self.trnvalX.shape[0]
        rand_indices = np.random.permutation(M_NUM)
        M_TRN_NUM = int(M_NUM * trnval_split_percentage)
        self.trnX, self.valX = self.trnvalX[rand_indices[:M_TRN_NUM]], self.trnvalX[rand_indices[M_TRN_NUM:]]
        self.trnY, self.valY = self.trnvalY[rand_indices[:M_TRN_NUM]], self.trnvalY[rand_indices[M_TRN_NUM:]]
        pass


    #TODO: add generality for "X,Y" data and "onlyX" data
    #NOTE: consider renaming this and/or protecting the internal objects a little more stringently
    def pull_data(self):
        return (self.trnvalX,self.trnvalY,self.tstX,self.tstY)
    def pull_trnval_data(self):
        return (self.trnX,self.trnY,self.valX,self.valY)

    #TODO:
    '''
    directly implement the trnval loader right here, no?

    '''















#TODO: need to make this much more general, and with friendlier data storage requirements (if the user wants to not use this specific numpy format)
class TabularDataset():

    def __init__(self, dataset_path, load_from_indicator="preprocessed"):

        #load the CSV

        #store the train/val/test splits

        #store the readable labels and all the orders of the features

        
        self.trnvalX = None
        self.trnvalY = None
        self.tstX = None
        self.tstY = None


        if load_from_indicator=="preprocessed":
            self.load_preprocessed_dataset_directly(dataset_path)
        elif load_from_indicator=="generative preprocessed":
            self.load_from_preprocessed_generative_dataset(dataset_path)


        if True:
            raise Exception("DEPRICATED PLEASE CHANGE TO NEW TabDS() FORMAT")
            # if os.path.exists(dataset_path+'data_labels.json'):
            #     self.readable_labels,self.datatype_labels = loadLabels(dataset_path+'data_labels.json')
            


    def load_preprocessed_dataset_directly(self, dataset_path):
        # trnvalX = np.load("") #TODO: change this back to trnval
        trnX1, trnY1, tstX, tstY = loadDataset(dataset_path)
        trnY1 = trnY1[:, None] #TODO: consider doing this processing elsewhere
        tstY  = tstY[:, None]

        self.trnvalX = trnX1
        self.trnvalY = trnY1
        self.tstX = tstX
        self.tstY = tstY

        self.trnX, self.valX, self.trnY, self.valY = None, None, None, None



    def load_from_preprocessed_generative_dataset(self, dataset_path, target_index): #TODO: maybe update to multiple targets
        X = np.load("../../../exp1_v2/X.npy")

        N = X.shape[0]
        D = X.shape[1]
        not_targets = list(range(D))
        not_targets.remove(target_index)
        X1 = X[:,not_targets]
        Y1 = X[:,target_index][:,None]


        readable_labels = {}
        for d in range(target_index):
            readable_labels[d] = "X"+str(d+1)
        for d in range(target_index,D): 
            readable_labels[d] = "X"+str(d+2)
        D=D-1
        del readable_labels[D]
        self.readable_labels = readable_labels


        
        split = (0.9, 0.1)
        np.random.seed(0)
        perm = np.random.permutation(N)
        trn_N = int(split[0]*N)
        self.trnvalX = X1[perm[:trn_N]]
        self.trnvalY = Y1[perm[:trn_N]]
        self.tstX = X1[perm[trn_N:]]
        self.tstY = Y1[perm[trn_N:]]


    def load_from_generative_dataset_numpy_array(self, X, target_index): #TODO: maybe update to multiple targets
        
        N = X.shape[0]
        D = X.shape[1]
        not_targets = list(range(D))
        not_targets.remove(target_index)
        X1 = X[:,not_targets]
        Y1 = X[:,target_index][:,None]


        readable_labels = {}
        for d in range(target_index):
            readable_labels[d] = "X"+str(d+1)
        for d in range(target_index,D): 
            readable_labels[d] = "X"+str(d+2)
        D=D-1
        del readable_labels[D]
        self.readable_labels = readable_labels


        
        split = (0.9, 0.1)
        split = (0.8, 0.2)
        np.random.seed(0)
        perm = np.random.permutation(N)
        trn_N = int(split[0]*N)
        self.trnvalX = X1[perm[:trn_N]]
        self.trnvalY = Y1[perm[:trn_N]]
        self.tstX = X1[perm[trn_N:]]
        self.tstY = Y1[perm[trn_N:]]





    def shuffle_and_split_trnval(self, trnval_shuffle_seed=None, trnval_split_percentage = .7):
        self.trnval_shuffle_seed = np.random.seed(None)
        print('trnval_shuffle_seed',trnval_shuffle_seed)
        print('self.trnval_shuffle_seed',self.trnval_shuffle_seed)

        M_NUM = self.trnvalX.shape[0]
        rand_indices = np.random.permutation(M_NUM)
        M_TRN_NUM = int(M_NUM * trnval_split_percentage)
        self.trnX, self.valX = self.trnvalX[rand_indices[:M_TRN_NUM]], self.trnvalX[rand_indices[M_TRN_NUM:]]
        self.trnY, self.valY = self.trnvalY[rand_indices[:M_TRN_NUM]], self.trnvalY[rand_indices[M_TRN_NUM:]]
        pass


    #TODO: rename this lol
    def pull_data(self):
        return (self.trnvalX,self.trnvalY,self.tstX,self.tstY)
    def pull_trnval_data(self):
        return (self.trnX,self.trnY,self.valX,self.valY)

    def get_D(self):
        return self.trnvalX.shape[1]
        
    def get_C(self):
        return self.trnvalY.shape[1]

    def get_dataset_id(self):
        return "not_implemented_yet_dataset_ID"



class TabularDatasetFromGenerativeDataset(): #TODO: clean up

    def __init__(self, dataset_path, target_index=0):

        #load the CSV

        #store the train/val/test splits

        #store the readable labels and all the orders of the features

        
        X = np.load(dataset_path+"X.npy")

        N = X.shape[0]
        D = X.shape[1]
        not_targets = list(range(D))
        not_targets.remove(target_index)
        X1 = X[:,not_targets]
        Y1 = X[:,target_index][:,None]


        readable_labels = {}
        for d in range(target_index):
            readable_labels[d] = "X"+str(d+1)
        for d in range(target_index,D): 
            readable_labels[d] = "X"+str(d+2)
        D=D-1
        del readable_labels[D]
        self.readable_labels = readable_labels


        
        split = (0.9, 0.1)
        np.random.seed(0)
        perm = np.random.permutation(N)
        trn_N = int(split[0]*N)
        self.trnvalX = X1[perm[:trn_N]]
        self.trnvalY = Y1[perm[:trn_N]]
        self.tstX = X1[perm[trn_N:]]
        self.tstY = Y1[perm[trn_N:]]

        

        ###raise NotImplementedError("This is not implemented.")

        
    #03/23/2025 -- adding haphazardly to fix causal stuff?
    def shuffle_and_split_trnval(self, trnval_shuffle_seed=None, trnval_split_percentage = .7):
        self.trnval_shuffle_seed = np.random.seed(None)
        print('trnval_shuffle_seed',trnval_shuffle_seed)
        print('self.trnval_shuffle_seed',self.trnval_shuffle_seed)

        M_NUM = self.trnvalX.shape[0]
        rand_indices = np.random.permutation(M_NUM)
        M_TRN_NUM = int(M_NUM * trnval_split_percentage)
        self.trnX, self.valX = self.trnvalX[rand_indices[:M_TRN_NUM]], self.trnvalX[rand_indices[M_TRN_NUM:]]
        self.trnY, self.valY = self.trnvalY[rand_indices[:M_TRN_NUM]], self.trnvalY[rand_indices[M_TRN_NUM:]]
        pass

    #TODO: rename this lol
    # def pull_data(self):
    #     return (self.trnX1,self.trnY1,self.tstX,self.tstY)
    def pull_data(self):
        return (self.trnvalX,self.trnvalY,self.tstX,self.tstY)
    def pull_trnval_data(self):
        return (self.trnX,self.trnY,self.valX,self.valY)

    def get_D(self):
        return self.trnvalX.shape[1]
        
    def get_C(self):
        return self.trnvalY.shape[1]

    def get_dataset_id(self):
        return "not_implemented_Yet_dataset_ID"



class TabularGenerativeDataset(): 

    def __init__(self, dataset_path, target_index=0):        
        # X = np.load("../../../exp1_v2/X.npy")
        X = np.load(dataset_path+"X.npy")

        N = X.shape[0]
        D = X.shape[1]
        X1 = X

        readable_labels = {}
        for d in range(D):
            readable_labels[d] = "X"+str(d+1)
        self.readable_labels = readable_labels

        
        split = (0.9, 0.1)
        np.random.seed(0)
        perm = np.random.permutation(N)
        trn_N = int(split[0]*N)
        self._trnvalX = X1[perm[:trn_N]]
        self._tstX = X1[perm[trn_N:]]




    #03/23/2025 -- adding haphazardly to fix causal stuff?
    def shuffle_and_split_trnval(self, trnval_shuffle_seed=None, trnval_split_percentage = .7):
        self.trnval_shuffle_seed = np.random.seed(None)
        print('trnval_shuffle_seed',trnval_shuffle_seed)
        print('self.trnval_shuffle_seed',self.trnval_shuffle_seed)

        M_NUM = self.trnvalX.shape[0]
        rand_indices = np.random.permutation(M_NUM)
        M_TRN_NUM = int(M_NUM * trnval_split_percentage)
        self.trnX, self.valX = self.trnvalX[rand_indices[:M_TRN_NUM]], self.trnvalX[rand_indices[M_TRN_NUM:]]
        self.trnY, self.valY = self.trnvalY[rand_indices[:M_TRN_NUM]], self.trnvalY[rand_indices[M_TRN_NUM:]]
        pass

    #TODO: rename this lol
    # def pull_gen_data(self): #TODO: bugfix this later
    #     return (self._trnX1,self._tstX)
    # def pull_data(self):
    #     return (self.trnX1,self.trnY1,self.tstX,self.tstY)
    def pull_gen_data(self): #TODO: bugfix this later
        return (self._trnvalX,self._tstX)
    def pull_data(self):
        return (self.trnvalX,self.trnvalY,self.tstX,self.tstY)
    def pull_trnval_data(self):
        return (self.trnX,self.trnY,self.valX,self.valY)

    # def get_D(self):
    #     return self._trnX1.shape[1]
        
    # def get_C(self):
    #     if self.trnX1 is not None:
    #         return self.trnY1.shape[1]
    #     else:
    #         raise Exception("This is a generative dataset, there is no specific Y target")
    def get_D(self):
        return self._trnvalX.shape[1]
        
    def get_C(self):
        if self.trnvalX is not None:
            return self.trnvalY.shape[1]
        else:
            raise Exception("This is a generative dataset, there is no specific Y target")

    def get_dataset_id(self):
        return "not_implemented_Yet_dataset_ID"

    # def prep_directional_prediction(self, target_index=0):
    #     self.trnX1 = np.copy(self._trnX1)
    #     self.trnY1 = np.copy(self.trnX1[:,target_index][:,None])
    #     self.trnX1[:,target_index] = 0.0
        
    #     self.tstX = np.copy(self._tstX)
    #     self.tstY = np.copy(self.tstX[:,target_index][:,None])
    #     self.tstX[:,target_index] = 0.0
    def prep_directional_prediction(self, target_index=0):
        self.trnvalX = np.copy(self._trnvalX)
        self.trnvalY = np.copy(self.trnvalX[:,target_index][:,None])
        self.trnvalX[:,target_index] = 0.0
        
        self.tstX = np.copy(self._tstX)
        self.tstY = np.copy(self.tstX[:,target_index][:,None])
        self.tstX[:,target_index] = 0.0
