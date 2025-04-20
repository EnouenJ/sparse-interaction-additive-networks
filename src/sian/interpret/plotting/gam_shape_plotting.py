







import numpy as np
import torch
import matplotlib.pyplot as plt







cts_list = ["continuous", "semicontinuous"]
disc_list = ["discrete", "semidiscrete"]

# 04/13/2025
# cts_list = ["cts.raw",]
cts_list = ["cts.raw", "cts.positive"]
disc_list = ["disc.ordinal", "disc.onehot",]









def scatter_plot_shape_function_v2(ind, valX, shape_fn, full_readable_labels, ylim_per_shape, kk):
    # plt.figure(figsize=(12,7))
    plt.figure(figsize=(8,4.5))
    # plt.title(str(ind))
    # plt.title(str(ind)+" ~~ "+" x ".join([readable_labels[i] for i in ind]))
    
    plt.title(str(ind)+" ~~ "+" x ".join([full_readable_labels[i]['label'] for i in ind]))
    
                        
    if len(ind)==1:
        i=ind[0]
        # dtype_i = datatype_labels[i][0]
        feat_i_info = full_readable_labels[i]
        dtype_i = feat_i_info['encoding']

        if True:
            startdim = feat_i_info['startdim']
            numdims  = feat_i_info['numdims']
            X_i_full = valX[:,startdim:startdim+numdims]
            if numdims>1: #onehot case
                X_i = np.matmul(np.arange(numdims),X_i_full.T)
            else:
                X_i = X_i_full[:,0]
            X = X_i

        plot_type = None
        if dtype_i in cts_list:
            plot_type = "c"
        elif dtype_i in disc_list:
            plot_type = "d" 
        
        if plot_type=="c":
            # X = valX[:,i]
            plt.scatter(X,shape_fn)
            B=ylim_per_shape[kk]
            plt.ylim(-B,B)
            # plt.show()
        elif plot_type=="d":
            # X = valX[:,i]
            plt.scatter(X+np.random.rand(X.shape[0]),shape_fn)
            B=ylim_per_shape[kk]
            plt.ylim(-B,B)
            # plt.show()
            # plt.bar(X,all_shapes[:,ii]) #BAD FOR FULL VALX
            # plt.show()
        
    elif len(ind)==2:
        i=ind[0]
        j=ind[1]
        
        
        # dtype_i = datatype_labels[i][0]
        # dtype_j = datatype_labels[j][0]
        feat_i_info = full_readable_labels[i]
        feat_j_info = full_readable_labels[j]
        dtype_i = feat_i_info['encoding']
        dtype_j = feat_j_info['encoding']

        plot_type = None
        if dtype_i in cts_list and dtype_j in cts_list:
            plot_type = "cc"
        elif dtype_i in cts_list and dtype_j in disc_list:
            plot_type = "cd" 
        elif dtype_i in disc_list and dtype_j in cts_list:
            temp=i
            i=j
            j=temp
            feat_i_info = full_readable_labels[i]
            feat_j_info = full_readable_labels[j]
            print('dc') #dc -> cd finisehd converting
            plot_type = "cd" 
            
            # Iy = datatype_labels[j][2]
            Iy = feat_j_info['count'] #TODO: check
            if Iy>10:
                plot_type = "cD"
                plot_type = "cc"
        elif dtype_i in disc_list and dtype_j in disc_list:
            plot_type = "dd" 
            # Ix = datatype_labels[i][2]
            # Iy = datatype_labels[j][2]
            Ix = feat_i_info['count'] #TODO: check
            Iy = feat_j_info['count'] #TODO: check

            if Iy>Ix: #swap for rectangle
                temp=i
                i=j
                j=temp
                feat_i_info = full_readable_labels[i] #needed this as well
                feat_j_info = full_readable_labels[j]
                
            # Ix = datatype_labels[i][2]
            # Iy = datatype_labels[j][2]
            Ix = feat_i_info['count'] #TODO: check
            Iy = feat_j_info['count'] #TODO: check
            if Ix>20 and Iy<=10:
                plot_type = "Dd"  #TODO: still quite cluttered for Iy around 5~10
                
        
        print(plot_type)

        
        if True:
            ###X = valX[:,i]
            ###Y = valX[:,j]
            startdim = feat_i_info['startdim']
            numdims  = feat_i_info['numdims']
            X_i_full = valX[:,startdim:startdim+numdims]
            if numdims>1: #onehot case
                X_i = np.matmul(np.arange(numdims),X_i_full.T)
            else:
                X_i = X_i_full[:,0]
            startdim = feat_j_info['startdim']
            numdims  = feat_j_info['numdims']
            X_j_full = valX[:,startdim:startdim+numdims]
            if numdims>1: #onehot case
                X_j = np.matmul(np.arange(numdims),X_j_full.T)
            else:
                X_j = X_j_full[:,0]
            X = X_i
            Y = X_j
            
        if plot_type=="cc":
    #         plt.scatter(X,Y,c=all_shapes[:,kk])
            plt.scatter(X,Y,c=shape_fn,cmap='coolwarm')
            # plt.scatter(X,Y,c=all_shapes[:,kk],cmap='coolwarm', s=1000)
            B=ylim_per_shape[kk]
            plt.clim(-B,B)
            plt.colorbar()
            # plt.xlabel(readable_labels[i])
            # plt.ylabel(readable_labels[j])
            plt.xlabel(full_readable_labels[i]['label'])
            plt.ylabel(full_readable_labels[j]['label'])
            
            
        elif plot_type=="cd":
            
            # X = valX[:,i]
            # Y = valX[:,j]
            # Iy = datatype_labels[j][2]
            # Iy0 = datatype_labels[j][1]
            Iy = feat_j_info['count'] #TODO: check
            Iy0 = 0.0 #feat_j_info['min'] #TODO: check
            
            y_arange = np.arange(Iy+1) + Iy0
            y_arange = np.arange(Iy) + Iy0
            
            # plt.scatter(X,shape_fn,c=Y,cmap='Set1')
            # plt.clim(Iy0-0.5,Iy0+10-0.5)
            #plt.scatter(X+unifX,shape_fn,c=Y,cmap='Set1', clim=(Iy0-0.5,Iy0+10-0.5))
            plt.scatter(X,shape_fn,c=Y,cmap='Set1')
            plt.clim(Iy0-0.5,Iy0+9-0.5)
            plt.colorbar()
            B=ylim_per_shape[kk]
            plt.ylim(-B,B)
            # plt.xlabel(readable_labels[i])
            plt.xlabel(full_readable_labels[i]['label'])
            pass
        
        elif plot_type=="Dd":
            
            # X = valX[:,i]
            # Y = valX[:,j]
            # Iy = datatype_labels[j][2]
            # Iy0 = datatype_labels[j][1]
            Iy = feat_j_info['count'] #TODO: check
            Iy0 = 0.0 #feat_j_info['min'] #TODO: check

            y_arange = np.arange(Iy+1) + Iy0
            y_arange = np.arange(Iy) + Iy0
            
            unifX = np.random.rand(X.shape[0])
            # plt.scatter(X+unifX,shape_fn,c=Y,cmap='Set1')
            # plt.clim(Iy0-0.5,Iy0+10-0.5)
            # plt.scatter(X+unifX,shape_fn,c=Y,cmap='Set1', clim=(Iy0-0.5,Iy0+10-0.5))
            # cmap = plt.get_cmap("Set1")
            # C = [cmap((0.5+y-Iy0)/9.0) for y in Y]
            # for iy in range(Iy):
            #     color=cmap((0.5+iy-Iy0)/9.0)
            #     plt.scatter([0],[0],c=[color],label=str(iy))
            # plt.legend()
            # plt.scatter(X+unifX,shape_fn,c=C)
            plt.scatter(X+unifX,shape_fn,c=Y,cmap='Set1')
            plt.clim(Iy0-0.5,Iy0+9-0.5)
            plt.colorbar()
            B=ylim_per_shape[kk]
            plt.ylim(-B,B)
            # plt.xlabel(readable_labels[i])
            plt.xlabel(full_readable_labels[i]['label'])
            
        elif plot_type=="dd":
            # X = valX[:,i]
            # Y = valX[:,j]
            # Ix = datatype_labels[i][2]
            # Iy = datatype_labels[j][2]
            # Ix0 = datatype_labels[i][1]
            # Iy0 = datatype_labels[j][1]
            Ix = feat_i_info['count'] #TODO: check
            Ix0 = 0.0 #feat_i_info['min'] #TODO: check
            Iy = feat_j_info['count'] #TODO: check
            Iy0 = 0.0 #feat_j_info['min'] #TODO: check

            x_arange = np.arange(Ix+1) + Ix0
            y_arange = np.arange(Iy+1) + Iy0
            # array = np.zeros((Ix,Iy))


            for ix in x_arange:
                plt.plot([ix]*(Iy+1),y_arange,c='k')
            for iy in y_arange:
                plt.plot(x_arange,[iy]*(Ix+1),c='k')
                
            
            unifX = np.random.rand(X.shape[0])
            unifY = np.random.rand(Y.shape[0])
            # plt.scatter(X+unifX,Y+unifY,c=all_shapes[:,kk])
            plt.scatter(X+unifX,Y+unifY,c=shape_fn,cmap='coolwarm')
            B=ylim_per_shape[kk]
            plt.clim(-B,B)
            plt.colorbar()
            # plt.xlabel(readable_labels[i])
            # plt.ylabel(readable_labels[j])
            plt.xlabel(full_readable_labels[i]['label'])
            plt.ylabel(full_readable_labels[j]['label'])
            
            
            pass
        
    plt.show()




#TODO TODO: cannot support onehot data inputs yet bc of the way it loads in data
def scatter_plot_shape_function_v1(ind, valX, shape_fn, full_readable_labels, ylim_per_shape, kk):
    # plt.figure(figsize=(12,7))
    plt.figure(figsize=(8,4.5))
    # plt.title(str(ind))
    # plt.title(str(ind)+" ~~ "+" x ".join([readable_labels[i] for i in ind]))
    
    plt.title(str(ind)+" ~~ "+" x ".join([full_readable_labels[i]['label'] for i in ind]))
    
                        
    if len(ind)==1:
        i=ind[0]
        # dtype_i = datatype_labels[i][0]
        feat_i_info = full_readable_labels[i]
        dtype_i = feat_i_info['encoding']

        plot_type = None
        if dtype_i in cts_list:
            plot_type = "c"
        elif dtype_i in disc_list:
            plot_type = "d" 
        
        if plot_type=="c":
            X = valX[:,i]
            plt.scatter(X,shape_fn)
            B=ylim_per_shape[kk]
            plt.ylim(-B,B)
            # plt.show()
        elif plot_type=="d":
            X = valX[:,i]
            plt.scatter(X+np.random.rand(X.shape[0]),shape_fn)
            B=ylim_per_shape[kk]
            plt.ylim(-B,B)
            # plt.show()
            # plt.bar(X,all_shapes[:,ii]) #BAD FOR FULL VALX
            # plt.show()
        
    elif len(ind)==2:
        i=ind[0]
        j=ind[1]
        
        
        # dtype_i = datatype_labels[i][0]
        # dtype_j = datatype_labels[j][0]
        feat_i_info = full_readable_labels[i]
        feat_j_info = full_readable_labels[j]
        dtype_i = feat_i_info['encoding']
        dtype_j = feat_j_info['encoding']

        plot_type = None
        if dtype_i in cts_list and dtype_j in cts_list:
            plot_type = "cc"
        elif dtype_i in cts_list and dtype_j in disc_list:
            plot_type = "cd" 
        elif dtype_i in disc_list and dtype_j in cts_list:
            temp=i
            i=j
            j=temp
            feat_i_info = full_readable_labels[i]
            feat_j_info = full_readable_labels[j]
            print('dc') #dc -> cd finisehd converting
            plot_type = "cd" 
            
            # Iy = datatype_labels[j][2]
            Iy = feat_j_info['count'] #TODO: check
            if Iy>10:
                plot_type = "cD"
                plot_type = "cc"
        elif dtype_i in disc_list and dtype_j in disc_list:
            plot_type = "dd" 
            # Ix = datatype_labels[i][2]
            # Iy = datatype_labels[j][2]
            Ix = feat_i_info['count'] #TODO: check
            Iy = feat_j_info['count'] #TODO: check

            if Iy>Ix: #swap for rectangle
                temp=i
                i=j
                j=temp
                feat_i_info = full_readable_labels[i] #needed this as well
                feat_j_info = full_readable_labels[j]
                
            # Ix = datatype_labels[i][2]
            # Iy = datatype_labels[j][2]
            Ix = feat_i_info['count'] #TODO: check
            Iy = feat_j_info['count'] #TODO: check
            if Ix>20 and Iy<=10:
                plot_type = "Dd"  #TODO: still quite cluttered for Iy around 5~10
                
        
        print(plot_type)
            
        if plot_type=="cc":
            X = valX[:,i]
            Y = valX[:,j]
    #         plt.scatter(X,Y,c=all_shapes[:,kk])
            plt.scatter(X,Y,c=shape_fn,cmap='coolwarm')
            # plt.scatter(X,Y,c=all_shapes[:,kk],cmap='coolwarm', s=1000)
            B=ylim_per_shape[kk]
            plt.clim(-B,B)
            plt.colorbar()
            # plt.xlabel(readable_labels[i])
            # plt.ylabel(readable_labels[j])
            plt.xlabel(full_readable_labels[i]['label'])
            plt.ylabel(full_readable_labels[j]['label'])
            
            
        elif plot_type=="cd":
            
            X = valX[:,i]
            Y = valX[:,j]
            # Iy = datatype_labels[j][2]
            # Iy0 = datatype_labels[j][1]
            Iy = feat_j_info['count'] #TODO: check
            Iy0 = feat_j_info['min'] #TODO: check
            
            y_arange = np.arange(Iy+1) + Iy0
            y_arange = np.arange(Iy) + Iy0
            
            # plt.scatter(X,shape_fn,c=Y,cmap='Set1')
            # plt.clim(Iy0-0.5,Iy0+10-0.5)
            #plt.scatter(X+unifX,shape_fn,c=Y,cmap='Set1', clim=(Iy0-0.5,Iy0+10-0.5))
            plt.scatter(X,shape_fn,c=Y,cmap='Set1')
            plt.clim(Iy0-0.5,Iy0+9-0.5)
            plt.colorbar()
            B=ylim_per_shape[kk]
            plt.ylim(-B,B)
            # plt.xlabel(readable_labels[i])
            plt.xlabel(full_readable_labels[i]['label'])
            pass
        
        elif plot_type=="Dd":
            
            X = valX[:,i]
            Y = valX[:,j]
            # Iy = datatype_labels[j][2]
            # Iy0 = datatype_labels[j][1]
            Iy = feat_j_info['count'] #TODO: check
            Iy0 = feat_j_info['min'] #TODO: check

            y_arange = np.arange(Iy+1) + Iy0
            y_arange = np.arange(Iy) + Iy0
            
            unifX = np.random.rand(X.shape[0])
            # plt.scatter(X+unifX,shape_fn,c=Y,cmap='Set1')
            # plt.clim(Iy0-0.5,Iy0+10-0.5)
            # plt.scatter(X+unifX,shape_fn,c=Y,cmap='Set1', clim=(Iy0-0.5,Iy0+10-0.5))
            # cmap = plt.get_cmap("Set1")
            # C = [cmap((0.5+y-Iy0)/9.0) for y in Y]
            # for iy in range(Iy):
            #     color=cmap((0.5+iy-Iy0)/9.0)
            #     plt.scatter([0],[0],c=[color],label=str(iy))
            # plt.legend()
            # plt.scatter(X+unifX,shape_fn,c=C)
            plt.scatter(X+unifX,shape_fn,c=Y,cmap='Set1')
            plt.clim(Iy0-0.5,Iy0+9-0.5)
            plt.colorbar()
            B=ylim_per_shape[kk]
            plt.ylim(-B,B)
            # plt.xlabel(readable_labels[i])
            plt.xlabel(full_readable_labels[i]['label'])
            
        elif plot_type=="dd":
            X = valX[:,i]
            Y = valX[:,j]
            # Ix = datatype_labels[i][2]
            # Iy = datatype_labels[j][2]
            # Ix0 = datatype_labels[i][1]
            # Iy0 = datatype_labels[j][1]
            Ix = feat_i_info['count'] #TODO: check
            Ix0 = feat_i_info['min'] #TODO: check
            Iy = feat_j_info['count'] #TODO: check
            Iy0 = feat_j_info['min'] #TODO: check

            x_arange = np.arange(Ix+1) + Ix0
            y_arange = np.arange(Iy+1) + Iy0
            # array = np.zeros((Ix,Iy))


            for ix in x_arange:
                plt.plot([ix]*(Iy+1),y_arange,c='k')
            for iy in y_arange:
                plt.plot(x_arange,[iy]*(Ix+1),c='k')
                
            
            unifX = np.random.rand(X.shape[0])
            unifY = np.random.rand(Y.shape[0])
            # plt.scatter(X+unifX,Y+unifY,c=all_shapes[:,kk])
            plt.scatter(X+unifX,Y+unifY,c=shape_fn,cmap='coolwarm')
            B=ylim_per_shape[kk]
            plt.clim(-B,B)
            plt.colorbar()
            # plt.xlabel(readable_labels[i])
            # plt.ylabel(readable_labels[j])
            plt.xlabel(full_readable_labels[i]['label'])
            plt.ylabel(full_readable_labels[j]['label'])
            
            
            pass
        
    plt.show()


# def plot_all_GAM_functions(sian2, valX, readable_labels, datatype_labels):
def plot_all_GAM_functions(sian2, valX, full_readable_labels):

    all_shapes = sian2.forward_shapes(torch.Tensor(valX))
    all_shapes = all_shapes.detach().cpu().numpy()
    print('all_shapes',all_shapes.shape)
    pass
    all_shape_vars = np.sqrt(np.var(all_shapes,axis=0))
    sorted_shapes = np.argsort(-all_shape_vars)
    # print(sorted_shapes)

    if True: #if rescaling somewhat for each y-axis
        ylim_per_shape = np.exp(  (np.max(all_shape_vars,keepdims=True)+all_shape_vars)/2  )

    # for kk in sorted_shapes:
    TOP_K = len(sorted_shapes)
    TOP_K = 10
    # TOP_K = 5
    for kkk in range(TOP_K):
        if kkk<sorted_shapes.shape[0]:
            kk = sorted_shapes[kkk]
            # ind = sian2.gam.all_indices[kk]
            ind = sian2.indices[kk] #04/13/2025 @ 9:00pm -- trying this instead for grouped/ungrouped indices
            print(ind)
            print(all_shape_vars[kk])
            
            shape_fn = all_shapes[:,kk]
            # scatter_plot_shape_function(ind, valX, shape_fn)
            # scatter_plot_shape_function(ind, valX, shape_fn, readable_labels, datatype_labels)
            # scatter_plot_shape_function(ind, valX, shape_fn, readable_labels, datatype_labels, ylim_per_shape, kk)
            
            # scatter_plot_shape_function(ind, valX, shape_fn, full_readable_labels, ylim_per_shape, kk)
            # scatter_plot_shape_function_v1(ind, valX, shape_fn, full_readable_labels, ylim_per_shape, kk)
            scatter_plot_shape_function_v2(ind, valX, shape_fn, full_readable_labels, ylim_per_shape, kk)