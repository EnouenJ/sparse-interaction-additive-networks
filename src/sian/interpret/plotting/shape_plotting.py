

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor #TODO: add try catch b/c not totally necessary




#TODO: ultimately should just be a "plotting_config" for a more general plotting function
def plot_all_sinewaves_from_synthetic(X_dataset, y):
    PLOT_LOCAL_PDP = True

    # for d in range(D):
    for d in X_dataset:
    
        X_d = X_dataset[d]
        if len(X_d.shape)>1:
            I_d = X_d.shape[1]
        else:
            I_d = 1
        if I_d>1:
            X_d = np.matmul(X_d,np.arange(I_d)) #/ I_d
        X_d=X_d[:,None]

        if PLOT_LOCAL_PDP:
            knn_regressor = KNeighborsRegressor(n_neighbors=300)
            knn_regressor.fit(X_d,y)
            predictions = knn_regressor.predict(X_d)


        plt.scatter(X_d, y, c=y, cmap='Spectral')
        plt.colorbar()
        if PLOT_LOCAL_PDP:
            plt.scatter(X_d, predictions,c='k')
        plt.title(f"d={d}")
        plt.show()



# def plot_sinewaves_from_synthetic(X_dataset, y):






def plot_all_raw_PDPs(X, y, full_readable_labels):
    PLOT_LOCAL_PDP = False
    PLOT_LOCAL_PDP = True

    for d in full_readable_labels:
        feat_d_info = full_readable_labels[d]
        label_d = feat_d_info['label']
        print(d,label_d)
        
        startdim = feat_d_info['startdim']
        numdims  = feat_d_info['numdims']
        X_d_full = X[:,startdim:startdim+numdims]

        if numdims>1:
            X_d = np.matmul(np.arange(numdims),X_d_full.T)
        else:
            X_d = X_d_full[:,0]

        if PLOT_LOCAL_PDP:
            X_d=X_d[:,None]
            knn_regressor = KNeighborsRegressor(n_neighbors=300)
            knn_regressor.fit(X_d,y)
            predictions = knn_regressor.predict(X_d)
            X_d=X_d[:,0]

        print("X_d",X_d.shape,"y",y.shape)
        plt.title(label_d)
        # plt.scatter(X_d, y)
        plt.scatter(X_d, y, c=y, cmap='Spectral')
        if PLOT_LOCAL_PDP:
            plt.scatter(X_d, predictions,c='k')
        # plt.ylim(-10,10)
        if 'sublabels' in feat_d_info:
            sublabels_d = feat_d_info['sublabels']
            plt.xticks(np.arange(numdims),sublabels_d) #TODO: need to allow for shift as well, maybe fine for categorical usually
        plt.show()