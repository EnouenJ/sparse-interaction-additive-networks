import math
import numpy as np
import matplotlib.pyplot as plt

    
import matplotlib.colors as mcolors
GAM_order_color_dict = {
    1   : mcolors.to_rgb('C3'),
    2   : mcolors.to_rgb('royalblue'),
    3   : mcolors.to_rgb('C2'),
    4   : mcolors.to_rgb('blueviolet'),
    5   : mcolors.to_rgb('C1'),
    None: [0.5,0.5,0.5],
}






def log_log_hist_helper(value_list,color="gray",res_scale=2.0):
    min_val = np.min(value_list)
    max_val = np.max(value_list)

    #round to nearest half int
    # res_scale = 2.0
    # res_scale = 4.0
    min_val = math.floor(res_scale*min_val)
    max_val = math.ceil(res_scale*max_val)  
    bins = np.linspace(min_val/res_scale,max_val/res_scale,max_val-min_val+1)

    plt.hist(value_list,color=color,alpha=.34,bins=bins)


def plot_1D_log_log_interaction_histogram(value_list, yscale_str="linear", res_scale=2.0):
    log_log_hist_helper(value_list,color="r",res_scale=res_scale)
    plt.yscale(yscale_str)
    plt.show()

def plot_2D_log_log_interaction_histogram(value_list, yscale_str="linear", res_scale=2.0):
    log_log_hist_helper(value_list,color="b",res_scale=res_scale)
    plt.yscale(yscale_str)
    plt.show()




#TODO: plot the learning curves after learning
# def plotting_learning_curves():
#     # plt.figure(figsize=(10,7))
#     # plt.plot(np.mean(all_losses[:k,:,0],axis=1))
#     # plt.plot(all_val_losses[:k])
#     # plt.plot(np.mean(all_losses[:k,:,1],axis=1))
#     # plt.plot(np.mean(all_losses[:k,:,2],axis=1))
#     # plt.plot([-50,EP+50],[0,0],c='k')
#     # plt.legend(['trn','val','mse','l1'])
#     # plt.xlim(-20,EP+20)
#     # plt.ylim(-.01,.45)
#     # plt.title("DNN/MLP Loss While Training")
#     # plt.show()



















def plot_archipelago_covariates(candidates, scores_arr, score_type_labels=None):
    score_type_labels = ['inc','rem','archipelago']
    SS = scores_arr.shape[1]
    assert SS == len(score_type_labels)
    C = len(candidates)
    
    plt.figure(figsize=(11,3))
    for ss in range(SS):
        plt.scatter(np.arange(C),scores_arr[:,ss])
        plt.plot(np.arange(C),scores_arr[:,ss],label=score_type_labels[ss])
    plt.legend()
    plt.xticks(np.arange(C),candidates,rotation=45)
    plt.show()


def fancy_plot_archipelago_covariances(candidates, scores_arr, score_type_labels=None):
    score_type_labels = ['inc','rem','archipelago']
    SS = scores_arr.shape[1]
    assert SS == len(score_type_labels)
    C = len(candidates)
    colors = np.zeros((C,SS,3))
    for cc,cand in enumerate(candidates):
        color_cc = GAM_order_color_dict[None]
        if len(cand) in GAM_order_color_dict:
            color_cc = GAM_order_color_dict[len(cand)]
        colors[cc,:,] = np.array(color_cc)[None]
        
    maxi = np.max(scores_arr,axis=1);    mini = np.min(scores_arr,axis=1);
    maxi = scores_arr[:,0];              mini = scores_arr[:,1];
    
    if True:
        linestyles = ['-','-','-'] 
        colors[:,0] = colors[:,0]/2 + 1/2
        colors[:,1] = colors[:,1]/2
    
    plt.figure(figsize=(11,3))
    for ss in range(SS):
        plt.plot(np.arange(C),scores_arr[:,ss],c='k',linestyle=linestyles[ss])#,alpha=0.5)
        plt.scatter(np.arange(C),scores_arr[:,ss],c=colors[:,ss],label=score_type_labels[ss])
    plt.fill_between(np.arange(C),mini,maxi,color='gray',alpha=0.3)
    plt.plot([-0.5,C-0.5],[0,0],c='k',linestyle=':')
    plt.legend()
    plt.xticks(np.arange(C),candidates,rotation=45)
    plt.show()
    













