import os
import copy
import time
import pickle
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





from sian.utils import gettimestamp

from sian.models import evaluate_model_on_test_set
from sian.models import either_normal_or_masked___gradient_descent_training #04/12/2025 @ 1:30am

from sian.models import MLP, SIAN, MaskedMLP, InstaSHAPMasked_SIAN
from sian.fis import layerwise_feature_interaction_selection_algorithm
from sian.fis import batchwise_feature_interaction_selection_algorithm

from sian.interpret import JamArchipelago, JamMaskedArchipelago
from sian.interpret.basic_wrappers import MixedModelWrapperTorch, Masked_MixedModelWrapperTorch








def get_loss_type(task_type):
    if task_type=="regression":
        return "mse"
    elif task_type=="binary_classification":
        return "ce"
    elif task_type=="multiclass_classification":
        return "softmax"
    else:
        raise Exception()


def train_mlp_final(my_dataset, training_args):
    exp_folder = training_args.saving_settings.exp_folder #NOTE: if exists
    datetimestr = gettimestamp()
    model_path = exp_folder+datetimestr+"_"
    net_name = training_args.model_config.net_name
    
    training_args.saving_settings.net_name = net_name
    training_args.saving_settings.results_to_save = ['best_net','final_net']
    training_args.saving_settings.results_save_prefix = model_path

    is_masked_mlp = training_args.model_config.is_masked
    D = my_dataset.get_D()
    C = my_dataset.get_C()
    task_type = my_dataset.get_task_type()
    loss_type = get_loss_type(task_type)
    training_args.loss_type = loss_type

    sizes = training_args.model_config.sizes
    sizes[0] = D; sizes[-1] = C;

    if is_masked_mlp:
        mlp = MaskedMLP(sizes)
        mlp, val_tensor, train_accuracy, val_accuracy, total_training_time, _, _, _ = \
            either_normal_or_masked___gradient_descent_training(my_dataset, mlp, training_args)
        final_test_mse = evaluate_model_on_test_set(my_dataset, mlp, training_args, is_masked_model=True)
    else:
        mlp = MLP(sizes)
        mlp, val_tensor, train_accuracy, val_accuracy, total_training_time, _, _, _ = \
            either_normal_or_masked___gradient_descent_training(my_dataset, mlp, training_args)
        final_test_mse = evaluate_model_on_test_set(my_dataset, mlp, training_args, is_masked_model=False)

    mlp_results = {
        "train_mse": train_accuracy,
        "val_mse": val_accuracy,
        # "test_mse": final_test_mse, #hide this for now (bc sweeping over hyperparameters)
        "total_training_time": total_training_time,

        "trained_mlp" : mlp, #TODO - do I need to pass a flag to return final or return best???
        "val_tensor" : val_tensor, #NOTE: find out the type that is returned
    }
    return mlp_results


def train_sian_final(my_dataset, training_args):
    exp_folder = training_args.saving_settings.exp_folder #NOTE: if exists
    datetimestr = gettimestamp()
    model_path = exp_folder+datetimestr+"_"
    net_name = training_args.model_config.net_name
    
    training_args.saving_settings.net_name = net_name
    training_args.saving_settings.results_to_save = ['best_net','final_net']
    training_args.saving_settings.results_save_prefix = model_path

    is_masked_sian = training_args.model_config.is_masked
    D = my_dataset.get_D()
    C = my_dataset.get_C()
    task_type = my_dataset.get_task_type()
    loss_type = get_loss_type(task_type)
    training_args.loss_type = loss_type

    training_args.model_config.feature_groups_dict = my_dataset.get_grouped_feature_dict() #04/13/2025 @ 11:00pm -- maybe better right here inside

    
    sizes = training_args.model_config.sizes
    small_sizes = training_args.model_config.small_sizes
    sizes[0] = D; sizes[-1] = C;
    small_sizes[0] = D; small_sizes[-1] = C;
    FIS_interactions = training_args.model_config.FIS_interactions
    feature_groups_dict = training_args.model_config.feature_groups_dict

    if is_masked_sian:
        insta_sian = InstaSHAPMasked_SIAN(sizes, FIS_interactions, small_sizes=small_sizes, feature_groups_dict=feature_groups_dict)
        trained_sian, val_tensor, train_accuracy, val_accuracy, total_training_time, _, _, _ = \
            either_normal_or_masked___gradient_descent_training(my_dataset, insta_sian, training_args)    
        final_test_mse = evaluate_model_on_test_set(my_dataset, trained_sian, training_args, is_masked_model=True)
    else:
        sian = SIAN(sizes, indices=FIS_interactions, small_sizes=small_sizes, feature_groups_dict=feature_groups_dict)
        trained_sian, val_tensor, train_accuracy, val_accuracy, total_training_time, _, _, _ = \
            either_normal_or_masked___gradient_descent_training(my_dataset, sian, training_args)    
        final_test_mse = evaluate_model_on_test_set(my_dataset, trained_sian, training_args, is_masked_model=False)

    sian_results = {        
        "train_mse": train_accuracy,
        "val_mse": val_accuracy,
        # "test_mse": final_test_mse, #hide this for now (bc sweeping over hyperparameters)
        "total_training_time": total_training_time,

        "trained_sian" : trained_sian,
        "val_tensor" : val_tensor, #NOTE: find out the type that is returned
    }
    return sian_results




def do_the_fis_final(my_FIS_hypers, my_val, AGG_K):
    if my_FIS_hypers.FIS_type=="layerwise":
        FIS_interactions, other_results = layerwise_feature_interaction_selection_algorithm(my_FIS_hypers, my_val, AGG_K)
    elif my_FIS_hypers.FIS_type=="batchwise":
        FIS_interactions, other_results = batchwise_feature_interaction_selection_algorithm(my_FIS_hypers, my_val, AGG_K)
    else:
        raise Exception(f"unrecognised FIS_type={my_FIS_hypers.FIS_type}")
    return FIS_interactions 


def initalize_the_explainer(trained_mlp, my_FID_hypers):
    if my_FID_hypers.is_masked_model:
        jam_arch = JamMaskedArchipelago(trained_mlp, my_FID_hypers)
    else:
        model_wrap_MLP = MixedModelWrapperTorch(trained_mlp, my_FID_hypers.device) 
        jam_arch = JamArchipelago(model_wrap_MLP, my_FID_hypers)
    return jam_arch
