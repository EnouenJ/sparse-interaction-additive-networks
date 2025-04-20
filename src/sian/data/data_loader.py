import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import csv
import json

from sian.utils import gettimestamp
import warnings


# def final_save_dataset(path, X, Y, split=None)
def save_full_dataset(path, X, Y):
    np.save(path+'fullX.npy', X)
    np.save(path+'fullY.npy', Y)

def load_full_dataset(path):
    fullX = np.load(path+'fullX.npy')
    fullY = np.load(path+'fullY.npy')
    return (fullX,fullY)

# splits and saves
def final_save_dataset(path, X, Y, trainval_portion, is_shuffled, shuffle_seed, SAVING=True):
    if shuffle_seed is None and is_shuffled is True:
        warnings.warn("potentially unreproducible shuffling behavior")
    
    NUM = X.shape[0]
    TRN_NUM = int(trainval_portion*NUM)
    TST_NUM = NUM - TRN_NUM

    if not is_shuffled:
        trnvalX = (X[:TRN_NUM])
        trnvalY = (Y[:TRN_NUM])
        tstX = (X[-TST_NUM:])
        tstY = (Y[-TST_NUM:])
    else:
        np.random.seed(shuffle_seed)
        perm = np.random.permutation(NUM)
        trnvalX = (X[perm[:TRN_NUM]])
        trnvalY = (Y[perm[:TRN_NUM]])
        tstX = (X[perm[-TST_NUM:]])
        tstY = (Y[perm[-TST_NUM:]])
        
    if not os.path.exists(path):
        os.mkdir(path)
    if SAVING:
        np.save(path+'trnvalX.npy',trnvalX)
        np.save(path+'trnvalY.npy',trnvalY)
        np.save(path+'tstX.npy',tstX)
        np.save(path+'tstY.npy',tstY)
    return trnvalX,trnvalY,tstX,tstY

def final_load_dataset(path):
    trnvalX = np.load(path+'trnvalX.npy')
    trnvalY = np.load(path+'trnvalY.npy')
    tstX = np.load(path+'tstX.npy')
    tstY = np.load(path+'tstY.npy')
    return trnvalX,trnvalY,tstX,tstY

# def final_save_labels(path, readable_labels, datatype_labels):
# def final_save_labels(path, readable_labels, datatype_labels, full_readable_labels):
# def final_save_labels(path, readable_labels, full_readable_labels):
def final_save_labels(path, data_labels_dict):
    # print("final_save_labels()",readable_labels)
    # print("final_save_labels()",full_readable_labels)
    # data_labels_dict = {
    #     "readable_labels" : readable_labels,
    #     # "datatype_labels" : datatype_labels,
    #     "full_readable_labels" : full_readable_labels,
    # }
    with open(path+'data_labels.json', 'w', encoding='utf-8') as f:
        json.dump(data_labels_dict, f, ensure_ascii=False, indent=4)

def final_load_labels(path):
    labels_path = path+'data_labels.json'
    if not os.path.exists(labels_path):
        return None,None
    else:
        with open(labels_path,'r') as f:
            labels_dict = json.load(f)
    readable_labels = convert_dict_keys_to_int(labels_dict["readable_labels"])
    # datatype_labels = convert_dict_keys_to_int(labels_dict["datatype_labels"])
    full_readable_labels = convert_dict_keys_to_int(labels_dict["full_readable_labels"])
    # return readable_labels,datatype_labels #TODO: convert back to dict-ish
    # return readable_labels,datatype_labels,full_readable_labels  #TODO: convert back to dict-ish
    # return readable_labels,full_readable_labels  #TODO: convert back to dict-ish

    if True:
        data_labels_dict = {
            "readable_labels" : readable_labels,
            "full_readable_labels" : full_readable_labels,
        }
        return data_labels_dict


def final_save_header(path, header_dict):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(path, exist_ok=True)
    with open(path+'data_header.json', 'w', encoding='utf-8') as f:
        json.dump(header_dict, f, ensure_ascii=False, indent=4)

def final_load_header(path):
    header_path = path+'data_header.json'
    if not os.path.exists(header_path):
        return None
    else:
        with open(header_path,'r') as f:
            header_dict = json.load(f)
    return header_dict









def saveDataset(path,X,Y,split=None):
    if split is None:
        split = (.8,.2)
    NUM = X.shape[0]
    TRN_NUM = int(split[0]*NUM)
    TST_NUM = NUM - TRN_NUM
    
    trnX = (X[:TRN_NUM])
    trnY = (Y[:TRN_NUM])
    tstX = (X[-TST_NUM:])
    tstY = (Y[-TST_NUM:])
        
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(path+'trnvalX.npy',trnX)
    np.save(path+'trnvalY.npy',trnY)
    np.save(path+'tstX.npy',tstX)
    np.save(path+'tstY.npy',tstY)
    
def loadDataset(path):
    trnX = np.load(path+'trnvalX.npy')
    trnY = np.load(path+'trnvalY.npy')
    tstX = np.load(path+'tstX.npy')
    tstY = np.load(path+'tstY.npy')
    return trnX,trnY,tstX,tstY


#TODO: currently has no shuffling (needs a seed); also need to integrate with v1 differences
def shuffleAndSaveDataset_v2(path,X,Y):
    # split = (.7,.1,.1,.1)
    split = (.6,.1,.1,.2)
    NUM = X.shape[0]
    TRN_NUM = int(split[0]*NUM)
    VAL_NUM = int(split[1]*NUM)
    VXL_NUM = int(split[2]*NUM)
    TST_NUM = NUM - TRN_NUM - VAL_NUM - VXL_NUM
    print(TRN_NUM,VAL_NUM,VXL_NUM,TST_NUM)

    trnX = (X[:TRN_NUM])
    trnY = (Y[:TRN_NUM])
    valX = (X[TRN_NUM:TRN_NUM+VAL_NUM])
    valY = (Y[TRN_NUM:TRN_NUM+VAL_NUM])
    vxlX = (X[TRN_NUM+VAL_NUM:TRN_NUM+VAL_NUM+VXL_NUM])
    vxlY = (Y[TRN_NUM+VAL_NUM:TRN_NUM+VAL_NUM+VXL_NUM])
    tstX = (X[-TST_NUM:])
    tstY = (Y[-TST_NUM:])

    if not os.path.exists(path):
        os.mkdir(path)
    np.save(path+'trnX.npy',trnX)
    np.save(path+'trnY.npy',trnY)
    np.save(path+'valX.npy',valX)
    np.save(path+'valY.npy',valY)
    np.save(path+'vxlX.npy',vxlX)
    np.save(path+'vxlY.npy',vxlY)
    np.save(path+'tstX.npy',tstX)
    np.save(path+'tstY.npy',tstY)

def loadPreshuffledDataset(path):
    trnX = np.load(path+'trnX.npy')
    trnY = np.load(path+'trnY.npy')
    valX = np.load(path+'valX.npy')
    valY = np.load(path+'valY.npy')
    vxlX = np.load(path+'vxlX.npy')
    vxlY = np.load(path+'vxlY.npy')
    tstX = np.load(path+'tstX.npy')
    tstY = np.load(path+'tstY.npy')
    return trnX,trnY,valX,valY,vxlX,vxlY,tstX,tstY



def convert_dict_keys_to_int(label_dict):
    new_dict = {}
    for key_str in label_dict:
        # print("key_str",key_str)
        try: #NOTE: care on this because of floats and maybe .isdigit() adapted for negatives is better
            new_dict[int(key_str)] = label_dict[key_str]
        except ValueError:
            new_dict[key_str] = label_dict[key_str]
    return new_dict


# def saveLabels(path, readable_labels, datatype_labels):
#     data_labels_dict = {
#         "readable_labels" : readable_labels,
#         "datatype_labels" : datatype_labels,
#     }
#     # with open(path+'readable_labels.json', 'w', encoding='utf-8') as f:
#     #     json.dump(readable_labels, f, ensure_ascii=False, indent=4)
#     with open(path+'data_labels.json', 'w', encoding='utf-8') as f:
#         json.dump(data_labels_dict, f, ensure_ascii=False, indent=4)

# def loadLabels(path):
#     # with open(path) as f:
#     #     labels = json.load(f)
#     # return labels
#     # with open(path) as f:
#     with open(path) as f: #TODO: add like loadheader (saftey and suffix)
#         labels_dict = json.load(f)
#     readable_labels = convert_dict_keys_to_int(labels_dict["readable_labels"])
#     datatype_labels = convert_dict_keys_to_int(labels_dict["datatype_labels"])
#     return readable_labels,datatype_labels

def saveHeader(path, header_dict):
    with open(path+'data_header.json', 'w', encoding='utf-8') as f:
        json.dump(header_dict, f, ensure_ascii=False, indent=4)

def loadHeader(path):
    header_path = path+'data_header.json'
    if not os.path.exists(header_path):
        return None
    else:
        # with open(path) as f:
        with open(header_path,'r') as f:
            header_dict = json.load(f)
    return header_dict


# def preprocess_dataset(dataset_id, load_dataset_path="data/", save_dataset_path="data/", preproc_owner=None):
def preprocess_dataset(dataset_id, preproc_owner=None,    load_dataset_path="data/", save_dataset_path="data/", SAVING_PREPROCESSED_VERSION=True):
    if preproc_owner is None:
        # preproc_owner = "Chan"
        preproc_owner = "SIAN2022"


    datetimestr = gettimestamp()
    header_dict = {
        "Preprocessed Datetime" : datetimestr, 
        "dataset_id" : dataset_id,
        "preproc_owner" : preproc_owner,
        "load_dataset_path" : load_dataset_path,
        "save_dataset_path" : save_dataset_path,
    }


    preprocessing_function = None
    if dataset_id == "UCI_275_bike_sharing_dataset":
        # preprocess_bike_sharing_dataset(load_dataset_path,save_dataset_path,preproc_owner)
        preprocessing_function = preprocess_bike_sharing_dataset
    elif dataset_id == "UCI_186_wine_quality":
        preprocessing_function = preprocess_wine_quality_dataset
    elif dataset_id == "UCI_2_adults_dataset":
        preprocessing_function = preprocess_adults_income_dataset
    elif dataset_id == "UCI_31_tree_cover_type_dataset":
        preprocessing_function = preprocess_tree_cover_dataset
    else:
        raise NotImplementedError('dataset not implemented yet')
        



    XY_stuff, label_stuff = preprocessing_function(load_dataset_path,save_dataset_path,preproc_owner)
    label_dict = {
        "readable_labels" : label_stuff[0],
        "full_readable_labels" : label_stuff[1],
    }
    fullX, fullY, trainval_portion, is_test_split_shuffled, shuffle_test_split_seed = XY_stuff #NOTE: dictionary probably makes this easier
    if trainval_portion is None:
        trainval_portion = 0.80 # default to 80/20 split
    header_dict["is_test_split_shuffled"]  = is_test_split_shuffled
    header_dict["shuffle_test_split_seed"] = shuffle_test_split_seed
    header_dict["trainval_portion"]        = trainval_portion
    if SAVING_PREPROCESSED_VERSION:
        final_save_header(save_dataset_path, header_dict) 
        ### ### saveDataset(save_dataset_path, *XY_stuff) #MAYBE A DICTIONARY IS BETTER?? #TODO: yeah, it is, I convert to a dict later anyways
        trnvalX,trnvalY,tstX,tstY = final_save_dataset(save_dataset_path, fullX, fullY, 
                                                        trainval_portion, is_test_split_shuffled, shuffle_test_split_seed)
        # saveLabels(save_dataset_path, *label_stuff)        
        # final_save_labels(save_dataset_path, *label_stuff)
        final_save_labels(save_dataset_path, label_dict)
    else:
        trnvalX,trnvalY,tstX,tstY = final_save_dataset(save_dataset_path, fullX, fullY, 
                                                        trainval_portion, is_test_split_shuffled, shuffle_test_split_seed, SAVING=False)
    return header_dict, (trnvalX,trnvalY,tstX,tstY), label_dict

def load_from_preprocessed(save_dataset_path):
    header_dict = final_load_header(save_dataset_path) 
    trnvalX,trnvalY,tstX,tstY = final_load_dataset(save_dataset_path)
    label_stuff = final_load_labels(save_dataset_path)
    return header_dict, (trnvalX,trnvalY,tstX,tstY), label_stuff











































#NOTE: should eventually change to os.join() to be more generic

def preprocess_bike_sharing_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):
    if preproc_owner=="SIAN2022" or preproc_owner=="Chan":
        file = open(load_dataset_path+'hour.csv','r')
        N = 17379 #total number of samples
        bike_share_data = np.zeros((N,17))

        file.readline()
        i=0
        for line in file:
            values = line.replace('\n','').split(',')
            day = int(values[1][8:10])
            values[1] = day

            bike_share_data[i] = [float(val) for val in values]
            i+=1
            
        bike_share_data = bike_share_data[:,1:]    
        bike_share_data[:,13:16] = np.log(1+bike_share_data[:,13:16])
        # saveDataset(save_dataset_path, bike_share_data[:,:13],bike_share_data[:,15])
        XY_stuff = (bike_share_data[:,:13], bike_share_data[:,15], None, False, None)

        
        readable_labels = {
            0  : "day",
            1  : "season",
            2  : "year",
            3  : "month",
            4  : "hour",

            5  : "holiday",
            6  : "day of week",
            7  : "workday",

            8  : "weather",
            9  : "temperature",
            10 : "feels_like_temp",
            11 : "humidity",
            12 : "wind speed",
        }
        datatype_labels = {
            0  : ['semidiscrete', 1, 31], #'ordinal'
            1  : ['semidiscrete', 1, 4],
            2  : ['semidiscrete', 0, 2],
            3  : ['semidiscrete', 1, 12],
            4  : ['semidiscrete', 0, 24],

            5  : ['semidiscrete', 0, 2],
            6  : ['semidiscrete', 0, 7],
            7  : ['semidiscrete', 0, 2],

            #8  : ['semidiscrete', 1, 3],
            8  : ['semidiscrete', 1, 4],   ###		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
            9  : ['continuous', ],
            10 : ['continuous', ],
            11 : ['semicontinuous', 0.0, 1.0],
            12 : ['semicontinuous', 0.0, None],
        }
        #TODO: subreadable labels
        sub_readable_labels = {
            0  : "day",
            1  : ["winter","spring","summer","fall"],   #guessing based on months
            2  : "year",
            3  : "month",
            4  : "hour",

            5  : ["no holiday", "special holiday"],
            6  : ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],  #day of week (pretty sure starts at sunday -- ought to double check)
            7  : ["weekend/holiday", "workday"],

            8  : ['Clear','Mist','Light Rain/Snow', 'Heavy Rain/Snow'],   #Weather
            9  : "temperature",
            10 : "feels_like_temp",
            11 : "humidity",
            12 : "wind speed",
        }
        full_readable_labels = {
            -1 : {}, #TODO: add the output
            'task_type' : "regression",
            "D0" : 13,

            0 : {"label" : "day", 
                "startdim" : 0, "numdims" : 1, 
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 1, "count" : 31},
            1 : {"label" : "season",
                "startdim" : 1, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 1, "count" : 4, #NOTE: should I label these "disc.ordinal.min" and "disc.ordinal.count"?
                "sublabels" : ["winter","spring","summer","fall"],},
            2 : {"label" : "year",
                "startdim" : 2, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 2},
            3 : {"label" : "month",
                "startdim" : 3, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 1, "count" : 12},
            4 : {"label" : "hour",
                "startdim" : 4, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 24},

            5 : {"label" : "holiday",
                "startdim" : 5, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 2},
            6 : {"label" : "day of week",
                "startdim" : 6, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 7},
            7 : {"label" : "workday",
                "startdim" : 7, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 2},
            8 : {"label" : "weather",
                "startdim" : 8, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinalish.categoricalish",
                "min" : 1, "count" : 4,
                "sublabels" : ['Clear','Mist','Light Rain/Snow', 'Heavy Rain/Snow'],},
                
            9 : {"label" : "temperature",
                "startdim" : 9, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            10: {"label" : "feels_like_temp",
                "startdim" : 10, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            11: {"label" : "humidity",
                "startdim" : 11, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : 1.0},
            12: {"label" : "wind speed",
                "startdim" : 12, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None},
        }


        # saveLabels(save_dataset_path, readable_labels, datatype_labels)
        # label_stuff = (readable_labels, datatype_labels)
        # label_stuff = (readable_labels, datatype_labels, full_readable_labels)
        label_stuff = (readable_labels, full_readable_labels)
        # print('--- processed and saved ---')
        print('--- processed and NOT saved ---')

        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")


def preprocess_energy_dataset(load_dataset_path, save_dataset_path, preproc_owner):
    if preproc_owner=="Chan":
        # Open the dataset file
        file = open(load_dataset_path+'energydata_complete.csv', 'r')
        
        # Read header and remaining lines
        header = file.readline()
        lines = file.readlines()
        N = len(lines)
        
        energy_data = np.zeros((N, 33))  # Adding new features from date
        
        i = 0
        for line in lines:
            values = line.replace('\n', '').replace('"', '').split(',')
            
            # Extract timestamp and convert to datetime object
            timestamp = values[0]
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            
            # Extract meaningful components from the date
            day = dt.day          # Day of the month
            hour = dt.hour        # Hour of the day
            minute = dt.minute    # Minute of the hour
            month = dt.month      # Month of the year
            weekday = dt.weekday()  # Day of the week (0=Monday, 6=Sunday)
            
            # Replace 'date' column with extracted features
            values = [day, hour, minute, month, weekday] + values[1:]
            
            # Convert all remaining values to float and populate the array
            energy_data[i] = [float(val.strip()) if isinstance(val, str) else float(val) for val in values]
            i += 1
        
        # Remove duplicate or irrelevant columns (e.g., 'rv2')
        energy_data = np.delete(energy_data, 32, axis=1)
        
        # Log-transform 'Appliances' and other necessary columns
        # energy_data[:, 5:7] = np.log(1 + energy_data[:, 5:7])  # Transform 'Appliances' and 'Lights'
        # Save processed dataset
        saveDataset(save_dataset_path, energy_data[:, 5:], energy_data[:, 4])  # Features, Target ('Appliances' is column 4 now)
        print('--- processed and saved ---')

    elif preproc_owner=="SIAN2022":

        file = open(load_dataset_path+'energydata_complete.csv', 'r')
        # Read header and remaining lines
        header = file.readline()
        lines = file.readlines()
        N = len(lines)

        energydatas = np.zeros( (N,31) )
        # (31 features) like original paper
        xd=0
        file_to_open = os.path.join(load_dataset_path, "energydata_complete.csv")
        with open(file_to_open, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            for row in spamreader:
                #print(row)
                if xd!=0:
                    datestr = row.pop(0)#row[0]
                    year=int(datestr[:4])
                    mon=int(datestr[5:7])
                    day=int(datestr[8:10])
                    #print(year,mon,day)
                    hour=int(datestr[11:13])
                    minu=int(datestr[14:16])
                    sec=int(datestr[17:19])
                    #print(hour,minu,sec)
                    #print(len(row))
                    newrow = [float(thing) for thing in row]
                    xdd = datetime(year,mon,day)
                    weekday_num = (xdd.weekday())
                    #print(weekday_num)
                    weekend_status = 1
                    if weekday_num==5 or weekday_num==6:
                        weekend_status = 0

                    energydatas[xd-1,:28] = newrow
                    energydatas[xd-1,28]  = hour*60*60+minu*60+sec
                    energydatas[xd-1,29]  = weekend_status
                    energydatas[xd-1,30]  = weekday_num
                    
                if xd==0:
                    pass
                    #print(len(row))
                    for thing in row:
                        pass
                        #print(thing)

                xd+=1
                #if xd>3:
                #if xd>1 and len(datestr)!=len('2016-01-11 18:30:00'):
                #if xd>1 and xdd.weekday()!=0:
                    #quit()
        print(xd-1)


        # PHASE TWO of ADJUSTING --- TODO: double check these renormaliziation values and do preloading of hardcoded scaling in a more clean fashion
        means = [97.6949581960983, 3.8018748416518875, 21.686571386748575, 40.259739279784384, 20.341219463849324, 40.42042041370675, 22.267610984880132, 39.24250007720708, 20.855334722410657, 39.02690378814554, 19.59210632801864, 50.94928262962212, 7.910939332403554, 54.609083387583084, 20.267106470136902, 35.38820021508418, 22.02910672298018, 42.93616537238887, 19.485828160608985, 41.552400753376595, 7.4116645553585565, 755.5226019761848, 79.75041803901698, 4.0397517101596145, 38.33083354446415, 3.7607068659741527, 24.988033485049435, 24.988033485049435, 42907.12946541677, 0.7227261211046364, 2.977248543197365]
        varis = [102.52229296483686, 7.9357865338828555, 1.606024953504309, 3.9791980108449656, 2.1929179723029724, 4.069709427602022, 2.006059709288938, 3.2544940346086375, 2.0428327184256, 4.341210661356689, 1.8445765380903627, 9.021805724091333, 6.090192304747093, 31.14901666343978, 2.109939865220912, 5.114078456225064, 1.9561121604537024, 5.224228315096331, 2.0146613412833356, 4.151392141435619, 5.317274083683381, 7.399253187484764, 14.900710023289937, 2.451158501562397, 11.79441992615117, 4.194541559356488, 14.49626657342557, 14.49626657342557, 24939.38895001474, 0.4476528509657154, 1.9855671861501942]

        means = [100, 3.8, 22, 40, 20, 40, 20, 40, 20, 40, 20, 50, 8, 50, 20, 35, 20, 40, 20, 40, 7, 750, 80, 4.0, 40, 4, 25, 25, 40000]#, 0.75, 3.0]
        varis = [100, 8,  1.6,  4,  2,  4,  2,  4,  2,  4,  2,  9, 6, 30,  2,  5,  2,  5,  2,  4, 5, 7.5, 15, 2.5, 12, 4, 15, 15, 25000]#, 0.45, 2.0]

        # N=energydatas.shape[0]
        for i in range(29):
            #print(means[i],varis[i])
            energydatas[:,i] = (energydatas[:,i]-means[i])/varis[i]
        all_energies_adjusted = np.zeros( (N,29+2+7) ) #38
        all_energies_adjusted[:,:29] = energydatas[:,:29]

        all_energies_adjusted[np.arange(N),(29+energydatas[:,29]).astype(int)] = 1
        all_energies_adjusted[np.arange(N)[:,None],(31+energydatas[:,30]).astype(int)[:,None]] = 1

        saveDataset(save_dataset_path, all_energies_adjusted[:, 1:], all_energies_adjusted[:, :1])  # Features, Target ('Appliances' is column 4 now)
        print('--- processed and saved ---')
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")





def preprocess_wine_quality_dataset(load_dataset_path, save_dataset_path, preproc_owner):


    if preproc_owner=="Chan":
        raise Exception("not updated to non-saving version")
        # Define file paths
        red_wine_file = load_dataset_path+'winequality-red.csv'
        white_wine_file = load_dataset_path+'winequality-white.csv'
        
        # Load and preprocess red wine dataset
        red_data = preprocess_single_wine_dataset_with_color(red_wine_file, color="red")
        print(f'Red wine dataset processed. Shape: {red_data.shape}')
        
        # Load and preprocess white wine dataset
        white_data = preprocess_single_wine_dataset_with_color(white_wine_file, color="white")
        print(f'White wine dataset processed. Shape: {white_data.shape}')
        
        # Combine datasets
        combined_data = np.vstack([red_data, white_data])
        print(f'Combined dataset shape: {combined_data.shape}')
        
        # Separate features and target
        features = combined_data[:, :-1]
        target = combined_data[:, -1]
        
        # Save processed dataset
        saveDataset(save_dataset_path, features, target)
        print('--- processed and saved combined dataset ---')

    elif preproc_owner=="SIAN2022":

        white_wines = loadWines(load_dataset_path+'winequality-white.csv',4898)
        red_wines = loadWines(load_dataset_path+'winequality-red.csv',1599)
        white_wines[:,11]=1
        red_wines[:,12]=1

        # print('white_wines',white_wines.shape)
        # print('red_wines',red_wines.shape)
        wines = np.concatenate([white_wines,red_wines],axis=0)
        # print('wines',wines.shape)

        XY_stuff = (wines[:, :-1], wines[:, -1], None, True, 0)


        
        # readable_labels = ['fixed acidity', 'volatile acidity', 'citric acid',
        #            'residual sugar', 'chlorides', 'free sulfur dioxide',
        #            'total sulfur dioxide', 'density', 'pH', 'sulphates',
        #            'alcohol', 'quality']
        readable_labels_list = ['fixed acidity', 'volatile acidity', 'citric acid',
                   'residual sugar', 'chlorides', 'free sulfur dioxide',
                   'total sulfur dioxide', 'density', 'pH', 'sulphates',
                   'alcohol', 'color', 'quality']
        readable_labels = {}
        # datatype_labels = {}
        for d,label in enumerate(readable_labels_list[:-1]):
            readable_labels[d] = label
            # datatype_labels[d] = ['continuous']
        # label_stuff = (readable_labels, datatype_labels)
        full_readable_labels = {
            -1 : {"label" : "wine quality",
                "startdim" : 0, "numdims" : 1,
                "encoding" : "cts.disc", "type" : "disc.ordinal",
                "count" : 10,
            },
            "task_type" : "regression",
            "D0" : 12,

            0 : {"label" : "fixed acidity", 
                "startdim" : 0, "numdims" : 1, 
                "encoding" : "cts.raw", "type" : "cts.",
                },
            1 : {"label" : "volatile acidity",
                "startdim" : 1, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            2 : {"label" : "citric acid",
                "startdim" : 2, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            3 : {"label" : "residual sugar",
                "startdim" : 3, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            4 : {"label" : "chlorides",
                "startdim" : 4, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            5 : {"label" : "free sulfur dioxide",
                "startdim" : 5, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            6 : {"label" : "total sulfur dioxide",
                "startdim" : 6, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            7 : {"label" : "density",
                "startdim" : 7, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            8 : {"label" : "pH",
                "startdim" : 8, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            9 : {"label" : "sulphates",
                "startdim" : 9, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            10: {"label" : "alcohol",
                "startdim" : 10, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            11: {"label" : "color",
                "startdim" : 11, "numdims" : 2,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 2,
                "sublabels" : ["white", "red"],},
        }



        # label_stuff = (readable_labels, datatype_labels, full_readable_labels)
        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff

    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")


# @Chan
def preprocess_single_wine_dataset_with_color(file_path, color):
    # Open the dataset file
    with open(file_path, 'r') as file:
        header = file.readline().strip().split(';')
        lines = file.readlines()
    
    N = len(lines)
    num_features = len(header)
    
    wine_data = np.zeros((N, num_features + 2))

    color_one_hot = [1, 0] if color == "red" else [0, 1]
    
    for i, line in enumerate(lines):
        values = line.strip().split(';')
        numeric_values = [float(val) for val in values]
        wine_data[i] = color_one_hot + numeric_values

    return wine_data

# @James
def loadWines(CSV_FILE_NAME,N):
    wines = np.zeros( (N,14) )
    xd=0
    
    with open(CSV_FILE_NAME, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='\"')
        for row in spamreader:
            if xd!=0:
                newrow = [float(thing) for thing in row]
                quality = newrow.pop(11)
                wines[xd-1,:11] = newrow
                wines[xd-1,13]  = quality
                
            if xd==0:
                pass
                #print(len(row))
                for thing in row:
                    pass
                    #print(thing)

            xd+=1
    return wines





def preprocess_adults_income_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):
    if preproc_owner=="InstaSHAP2025":
        N = 32561
        D = 15
        CTS_VARS = [1,3, 11,12,13]
        CSV_FILENAME = load_dataset_path+'adult.data'
        

        def get_cumulative_index_sizes(I_ks):
            D=len(I_ks)
            cum_n = 0
            cum_I_ks = [0,]
            for d in range(D):
                cum_n += I_ks[d]
                cum_I_ks.append(cum_n)
            return cum_I_ks

        event_dictionary = {
            0 : ['<=50K', '>50K'],
            1 : ['<20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-79', '80-89', '90+'],
            2 : ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked', None],
            3 : ['<50K', '50K-100K', '100K-150K', '150K-200K', '200K-250K', '250K-300K', '300K-350K', '350K-400K', '400K+'],
            4 : ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'],

            #DUPLICATED FEATURE AS A NUMERICAL VALUE
            5 : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            
            6 : ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
            7 : ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv', None],
            8 : ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
            9 : ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
            10 : ['Male', 'Female'],
            11 : ['0', '<1K', '1K-3K', '3K-10K', '10K-30K', '30K+'],
            12 : ['1K', '1K-1.5K', '1.5K-2K', '2K-2.5K', '2.5K-5K'],
            13 : ['0-10', '10-20', '20-30', '30-35', '35-40', '40-45', '45-50', '50-60', '60-80', '80-100'],
            14 : ['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands', None],
        }

        readable_label_dict = {
            0 : "income",
            1 : "age",
            2 : "workclass",
            3 : "fnlwgt",
            4 : "education",
            5 : "education-num",
            6 : "marital-status",
            7 : "occupation",
            8 : "relationship",
            9 : "race",
            10 : "sex",
            11 : "capital-gain",
            12 : "capital-loss",
            13 : "hours-per-week",
            14 : "native-country",
        }

        bins_dictionary = {
            1 : [14.5, 19.5, 24.5, 29.5, 34.5, 39.5, 44.5, 49.5, 54.5, 59.5, 64.5, 69.5,79.5,89.5,99.5],
            3 : [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 1500000],
            
            11 : [-0.5,0.5,1000.5,3000.5,10000.5,   30000.5, 100000.5],
            12 : [-0.5,1000.5,1500.5,2000.5,2500.5,5000.5],
            13 : [0.5, 10.5, 20.5,30.5, 35.5, 40.5, 45.5, 50.5, 60.5, 80.5,100.5],
        }
        D=15


        I_ks = []
        for d in range(D):
            if d in CTS_VARS:
                I_ks.append( 1 )
                #I_ks.append( len(bins_dictionary[d])-1 )
                #assert len(bins_dictionary[d]) == len(event_dictionary[d])+1, 'mislabeled readable labels @ '+str(d) +\
                #                                                                '  ('+str(len(bins_dictionary[d])-1)+','+str(len(event_dictionary[d]))+')'
            else:
                I_ks.append( len(event_dictionary[d]) )
        I_ks=tuple(I_ks)
        print('I_ks    \t',I_ks)
        print("D",len(I_ks),"\t\tonehot D",sum(I_ks))

        cum_I_ks = get_cumulative_index_sizes(I_ks)
        print('cum_I_ks \t',cum_I_ks)

        rescale_cts_vars_dict = {
            1 : (40,10),
            3 : (200*1000,100*1000),
            
            11: (1000,8000),
            12: (80,400),
            13: (40,10),
        }

        N = 32561
        ONEHOT_D = sum(list(I_ks))
        X_arr = np.zeros( (N,ONEHOT_D), dtype=float )

        import csv
        with open(CSV_FILENAME, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',') #, quotechar='|')
            n=0
            for row in spamreader:
                if n<N: #white space at the end of the file
                    cum_n = 0
                    for d in range(D):
                        if d>0:
                            value = row[d-1].strip()
                        else:
                            value = row[-1].strip()

                        v_id = -1
                        if d in CTS_VARS:
                            value = int(value)

                            # v_id = -1
                            # bins_d = bins_dictionary[d]
                            # for bb,b in enumerate(bins_d):
                            #     if value>b:
                            #         v_id = bb

                            #NOTE: JAM ON 04/13/2025 -- this looks like where I made the original change
                            #                            everything seems fine, I just used mahgenta as a starting point
                            value_rescaled = (value - rescale_cts_vars_dict[d][0]) / rescale_cts_vars_dict[d][1]
                            X_arr[n,cum_n] = value_rescaled
                            cum_n += 1

                        else:
                            if d==5:
                                value = int(value)
                            if value not in event_dictionary[d]:
                                print('FAILURE\t',d)
                            v_id = event_dictionary[d].index(value)
        
                            X_arr[n,cum_n+v_id] = 1
                            cum_n += I_ks[d]
                pass
                n+=1
        print(n)



        #REMOVING FEATURE #5

        d_to_remove = 5
        print(I_ks[5],cum_I_ks[5])
        I_5 = I_ks[5]
        cum_I_5 = cum_I_ks[5]

        print('X_arr',X_arr.shape)
        X_arr = np.concatenate([X_arr[:,:cum_I_5],X_arr[:,cum_I_5+I_5:]],axis=-1)
        print('X_arr',X_arr.shape)

        for d in range(D):
            if d>5:
                readable_label_dict[d-1] = readable_label_dict[d]
                event_dictionary[d-1]    = event_dictionary[d]
        I_ks=list(I_ks)
        I_ks.pop(5)
        I_ks=tuple(I_ks)
        D=14
        del readable_label_dict[14]
        del event_dictionary[14]

        print("D",len(I_ks),"\t\tonehot D",sum(I_ks))
        print('I_ks    \t',I_ks)
        cum_I_ks = get_cumulative_index_sizes(I_ks)
        print('cum_I_ks \t',cum_I_ks)



        #NOTE: double check doing binary classification with two dimensions
        # XY_stuff = (X_arr[:, 2:], X_arr[:, :2], None, True, 0) #TODO: onehot output is too general right now
        XY_stuff = (X_arr[:, 2:], X_arr[:, 1], None, True, 0)




        
        full_readable_labels = {
            # -1 : {"label" : "income level",
            #     "startdim" : 0, "numdims" : 2,
            #     "encoding" : "disc.onehot", "type" : "disc.ordinal",
            #     "count" : 2,
            #     "sublabels" : ['<=50K', '>50K'],
            # },
            -1 : {"label" : "income level",
                "startdim" : 0, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "disc.ordinal",
                "min" : 0, "count" : 2,
                "sublabels" : ['<=50K', '>50K'],
            },
            "task_type" : "binary_classification",
            "D0" : 13,

            0 : {"label" : "age", 
                "startdim" : 0, "numdims" : 1, 
                "encoding" : "cts.raw", "type" : "cts.",
                },
            1 : {"label" : "workclass",
                "startdim" : 1, "numdims" : 10,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 10,
                "sublabels" : ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked', None],
                },
            2 : {"label" : "fnlwgt",
                "startdim" : 11, "numdims" : 1, #NOTE: I shouldnt be manually inputting 'startdim' #TODO: add a cumulative one like above (but for all datasets)
                "encoding" : "cts.raw", "type" : "cts.",
                },
            3 : {"label" : "education",
                "startdim" : 12, "numdims" : 16,
                "encoding" : "disc.onehot", "type" : "disc.ordinal",
                "count" : 16,
                "sublabels" : ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'],
                },
            4 : {"label" : "marital-status",
                "startdim" : 28, "numdims" : 7,
                "encoding" : "cts.raw", "type" : "cts.",
                "count" : 7,
                "sublabels" :  	 ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
                },
            5 : {"label" : "occupation",
                "startdim" : 35, "numdims" : 16,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 16,
                "sublabels" :  	 ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv', None],
                },
            6 : {"label" : "relationship",
                "startdim" : 51, "numdims" : 6,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 6,
                "sublabels" :  	  ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
                },
            7 : {"label" : "race",
                "startdim" : 57, "numdims" : 5,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 5,
                "sublabels" :  	 ['white', 'black', 'asian-pac-islander', 'native-american-indian-eskimo', 'other'],
                },
            8 : {"label" : "sex",
                "startdim" : 62, "numdims" : 2,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 2,
                "sublabels" :  	['male', 'female'] ,
                },

            9 : {"label" : "capital-gain",
                "startdim" : 64, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            10: {"label" : "capital-loss",
                "startdim" : 65, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            11: {"label" : "hours-per-week",
                "startdim" : 66, "numdims" : 2,
                "encoding" : "cts.positive", "type" : "cts.",
                "lb" : 0.0, "ub" : None},
                
            12 : {"label" : "native-country",
                "startdim" : 67, "numdims" : 43,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 43,
                "sublabels" :  	['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands', None]
                },
        }
        readable_labels={}
        for thing in full_readable_labels:
            if type(thing)==int:
                readable_labels[thing] = full_readable_labels[thing]['label']

        # label_stuff = (readable_labels, datatype_labels, full_readable_labels)
        label_stuff = (readable_labels, full_readable_labels)

        #FROM INSTASHAP CODE: TODO
        # 1 : ['state gov', 'self emp\n(not inc)', 'private', 'federal gov', 'local gov',
        #         '(missing value)', 'self emp\n(inc)', 'without pay', 'never worked', None],
        #     3 : ['preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS grad', 'some college',
        #         'assoc. voc', 'assoc. acdm', 'bachelors', 'masters', 'prof school', 'doctorate'],
        #     4 : ['never\nmarried', 'married\n(civ spouse)', 'divorced',
        #         'married\n(spouse\nabsent)', 'separated',
        #         'married\n(AF spouse)', 'widowed'],
        #     5 : ['admin/clerical', 'exec/managerial', 'handlers/cleaners', 'prof/specialty', 'other service',
        #         'sales', 'craft/repair', 'transport/moving', 'farming/fishing', 'machine op inspct', 'tech support',
        #         '(missing value)', 'protective serv', 'armed forces', 'priv house serv', None],
        #     6 : ['not in\nfamily', 'husband', 'wife', 'own\nchild', 'unmarried', 'other\nrelative'],
        #     #7 : ['white', 'black', 'asian/\npacific\nislander', 'american\nindian/\neskimo', 'other'],
        #     7 : ['white', 'black', 'asian', 'american\nindian', 'other'],
        #     8 : ['male', 'female'],



        print('--- processed and NOT saved ---')
        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")
        





def preprocess_tree_cover_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):
    if preproc_owner=="InstaSHAP2025":







        readable_labels_dict = {
            'elevation (m)'     : 0,
            'aspect (azimuth)'  : 1,
            'slope (deg)'       : 2,
            'horizontal_dist_to_hydro (m)' : 3,
            'vertical_dist_to_hydro (m)'   : 4,
            'horizontal_dist_to_road (m)'  : 5,
            'hillshade_9am'  : 6, #[0,255)
            'hillshade_noon' : 7, #[0,255)
            'hillshade_3pm'  : 8, #[0,255)
            'horizontal_dist_to_fire_point (m)' : 9,
            
            'wilderness_area_label' : list(range(10,14)), #Rawah=1, Neota=2, Comanche Peak=3, Cache la Poudre =4
            'soil_type' : list(range(14,54)),
        }


        # original splits to have a balanced training set and unbalanced test set
        # (maybe not reasonable under modern ML frameworks). 
        # either way, we ignore it
        TRN_N =  11340
        VAL_N =   3780
        TST_N = 565892
        N = TRN_N+VAL_N+TST_N  #581,012

        all_data_array = np.zeros((N,55),dtype=int)

        CSV_PATH = load_dataset_path + "covtype.data"

        xd=0
        with open(CSV_PATH, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                all_data_array[xd] = [int(thing) for thing in row]
                xd+=1

        # quant_mus  = [2959.36, 155.65, 14.10, 269.43, 46.42, 2350.15, 212.15, 223.32, 142.53, 1980.29]
        # quant_vars = [ 279.98, 111.91,  7.49, 212.55, 58.30, 1559.25,  26.77,  19.77,  38.27, 1324.19]


        # soil_remappings = {
        #     0 : list(range(0,6)),
        #     1 : list(range(6,8)),
        #     2 : list(range(8,9)),
        #     3 : list(range(9,13)),
        #     4 : list(range(13,15)),
        #     5 : list(range(15,17)),
        #     6 : list(range(17,18)),
        #     7 : list(range(18,21)),
        #     8 : list(range(21,23)),
        #     9 : list(range(23,34)),
        #     10 : list(range(34,40)),    
        # }
        # soil_remapping_tensor = np.zeros((40,11))
        # for c2 in range(11):
        #     for c1 in soil_remappings[c2]:
        #         soil_remapping_tensor[c1,c2] = 1
                
        # soil_remappings2 = {
        #     0 : list(range(0,6)),
        #     1 : list(range(6,8)),
        #     2 : list(range(8,13)),
        #     3 : list(range(13,15)),
        #     4 : list(range(15,18)),
        #     5 : list(range(18,34)),
        #     6 : list(range(34,40)),    
        # }
        # soil_remapping_tensor2 = np.zeros((40,7))
        # for c2 in range(7):
        #     for c1 in soil_remappings2[c2]:
        #         soil_remapping_tensor2[c1,c2] = 1
                
        soil_remappings3 = { 
            0 : list(range(0,6)),   #2 --> lower montane
            1 : list(range(6,18)),  #3,4,5,6 --> upper montane
            2 : list(range(18,34)), #7 --> subalpine
            3 : list(range(34,40)), #8 --> alpine  
        }
        soil_remapping_tensor3 = np.zeros((40,4))
        for c2 in range(4):
            for c1 in soil_remappings3[c2]:
                soil_remapping_tensor3[c1,c2] = 1

        all_simple_soils_arr3 = np.matmul(all_data_array[:,14:54],soil_remapping_tensor3)
        all_data3 = np.concatenate([all_data_array[:,:14],all_simple_soils_arr3],axis=1)

        one_hot_labels = np.zeros((all_data_array.shape[0],7),dtype=int)
        one_hot_labels[np.arange(all_data_array.shape[0]) , (all_data_array[:, 54]-1).astype(int)] = 1


        # XY_stuff = (all_data3, all_data_array[:, 54], None, True, 0)
        XY_stuff = (all_data3, one_hot_labels, None, True, 0)



        """
        Name                                     Data Type    Measurement                       Description

        Elevation                               quantitative    meters                       Elevation in meters
        Aspect                                  quantitative    azimuth                      Aspect in degrees azimuth
        Slope                                   quantitative    degrees                      Slope in degrees
        Horizontal_Distance_To_Hydrology        quantitative    meters                       Horz Dist to nearest surface water features
        Vertical_Distance_To_Hydrology          quantitative    meters                       Vert Dist to nearest surface water features
        Horizontal_Distance_To_Roadways         quantitative    meters                       Horz Dist to nearest roadway
        Hillshade_9am                           quantitative    0 to 255 index               Hillshade index at 9am, summer solstice
        Hillshade_Noon                          quantitative    0 to 255 index               Hillshade index at noon, summer soltice
        Hillshade_3pm                           quantitative    0 to 255 index               Hillshade index at 3pm, summer solstice
        Horizontal_Distance_To_Fire_Points      quantitative    meters                       Horz Dist to nearest wildfire ignition points
        Wilderness_Area (4 binary columns)      qualitative     0 (absence) or 1 (presence)  Wilderness area designation
        Soil_Type (40 binary columns)           qualitative     0 (absence) or 1 (presence)  Soil Type designation
        Cover_Type (7 types)                    integer         1 to 7                       Forest Cover Type designation
        """
        pass        
        class_labels = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine","Cottonwood/Willow","Aspen","Douglas-fir","Krummholz"]
        full_readable_labels = {
            # -1 : {"label" : "tree species",
            #     "startdim" : 0, "numdims" : 7,
            #     "encoding" : "disc.ordinal", "type" : "disc.categorical",
            #     "min" : 1, "count" : 7,
            #     "sublabels" : class_labels,
            # },
            -1 : {"label" : "tree species",
                "startdim" : 0, "numdims" : 7,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 7,
                "sublabels" : class_labels,
            },
            "task_type" : "multiclass_classification",
            "D0" : 12,

            0 : {"label" : "elevation (m)", 
                "startdim" : 0, "numdims" : 1, 
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                },
            1 : {"label" : "aspect (azimuth)",
                "startdim" : 1, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'degrees azimuth',
                },
            2 : {"label" : "slope (deg)",
                "startdim" : 2, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'degrees',
                },
            3 : {"label" : "Horizontal_Distance_To_Hydrology",
                "startdim" : 3, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                'description' : "Horz Dist to nearest surface water features",
                },
            4 : {"label" : "Vertical_Distance_To_Hydrology",
                "startdim" : 4, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                'description' : "Vert Dist to nearest surface water features",
                },
            5 : {"label" : "Horizontal_Distance_To_Roadways",
                "startdim" : 5, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                'description' : "Horz Dist to nearest roadway",
                },
            9 : {"label" : "Horizontal_Distance_To_Fire_Points", #idk why put as 9 instead of putting here
                "startdim" : 9, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                'description' : "Horz Dist to nearest wildfire ignition points",
                },

            
            6 : {"label" : "Hillshade_9am",
                "startdim" : 6, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'pixel brightness [0,256)',
                'description' : 'Hillshade index at 9am, summer solstice',
                },
            7 : {"label" : "Hillshade_Noon",
                "startdim" : 7, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'pixel brightness [0,256)',
                'description' : 'Hillshade index at noon, summer solstice',
                },
            8 : {"label" : "Hillshade_3pm",
                "startdim" : 8, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'pixel brightness [0,256)',
                'description' : 'Hillshade index at 3pm, summer solstice',
                },


            10: {"label" : "wilderness area",
                "startdim" : 10, "numdims" : 4,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 4,
                "sublabels" : ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"],},
            11: {"label" : "soil type",
                "startdim" : 14, "numdims" : 4,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 4,
                "sublabels" : ["lower montane", "upper montane", "subalpine", "alpine"],},
        }
        readable_labels={}
        for thing in full_readable_labels:
            if type(thing)==int:
                readable_labels[thing] = full_readable_labels[thing]['label']

        # label_stuff = (readable_labels, datatype_labels, full_readable_labels)
        label_stuff = (readable_labels, full_readable_labels)




        print('--- processed and NOT saved ---')
        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")
        





