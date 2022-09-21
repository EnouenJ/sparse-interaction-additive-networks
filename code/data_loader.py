import numpy as np
import os

    
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
    np.save(path+'trnX.npy',trnX)
    np.save(path+'trnY.npy',trnY)
    np.save(path+'tstX.npy',tstX)
    np.save(path+'tstY.npy',tstY)
    
def loadDataset(path):
    trnX = np.load(path+'trnX.npy')
    trnY = np.load(path+'trnY.npy')
    tstX = np.load(path+'tstX.npy')
    tstY = np.load(path+'tstY.npy')
    return trnX,trnY,tstX,tstY




def preprocess_bike_sharing_dataset():
    file = open('data/hour.csv','r')
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
    saveDataset('data/',bike_share_data[:,:13],bike_share_data[:,15])
    print('--- processed and saved ---')




