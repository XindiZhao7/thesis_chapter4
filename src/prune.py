#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time, os
import numpy as np
import joblib
from copy import copy
from src.load_data import classification_data
from src.factorized_approximation import FactorizedHierarchicalInvGamma as inference_engine
from src.hs_bnn import HSBnn, fit


# In[2]:


def importance_estimation(model):
    importance_score = []
    engine = model.inference_engine
    _, _, tau_mu_vect, tau_sigma_vect, tau_mu_global, tau_sigma_global, _, _ = engine.unpack_params(model.optimal_elbo_params)
    for layer_id, (tau_mu, tau_sigma) in enumerate(zip(engine.unpack_layer_weight_priors(tau_mu_vect),
                                                       engine.unpack_layer_weight_priors(tau_sigma_vect))):
        if layer_id == len(model.shapes)-1:
            break
        scale_mu = 0.5 * (tau_mu + tau_mu_global[layer_id])
        scale_v = 0.25 * (tau_sigma**2 + tau_sigma_global[layer_id]**2)
        importance_score.append(scale_mu-scale_v)
    importance_score = np.concatenate(importance_score)
    return importance_score



def update_model_mask(m, score, sparsity_ratio):
    new_m = np.zeros_like(m)
    finetune_m = np.ones_like(m)
    indices = np.argsort(score)
    num_nodes = np.sum(m)
    candidate_node_indices_ = np.where(m==1)[0]
    knockout_node_indices = np.where(m==0)[0]
    _, ind, _ = np.intersect1d(indices, candidate_node_indices_, return_indices=True)
    candidate_node_indices = indices[np.sort(ind)]
    num_pruned_nodes = int(sparsity_ratio*num_nodes)
    new_m[candidate_node_indices[num_pruned_nodes:]] = 1
    new_m[knockout_node_indices]=1
    finetune_m[candidate_node_indices[:num_pruned_nodes]] = 0
    finetune_m[knockout_node_indices]=0
    return new_m,finetune_m



def update_gradient_mask(model_mask, layer_sizes):
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    w_mask = list()
    
    tau_mask = model_mask[:-layer_sizes[-1]]
    tau_global_mask = np.ones(len(shapes)-1)
    tau_op_mask = np.ones(1)
    for i, j in shapes:
        m = np.ones((i, j))
        m = (model_mask[:j] * m).reshape(-1)
        w_mask.append(m)
        n = np.ones(j)
        n = model_mask[:j] * n
        w_mask.append(n)
        model_mask = model_mask[j:]
        
    w_mask = np.concatenate(w_mask)
    
    gmask = np.concatenate([w_mask, w_mask, tau_mask, tau_mask, tau_global_mask, tau_global_mask, tau_op_mask, tau_op_mask])
    
    return gmask


def update_task_specific_params_only(model_mask, layer_sizes):
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    w_mask = list()
    
    tau_mask = np.zeros_like(model_mask[:-layer_sizes[-1]])
    tau_global_mask = np.ones(len(shapes)-1)
    tau_op_mask = np.ones(1)
    for i, j in shapes[:-1]:
        m = np.zeros((i, j)).reshape(-1)
        w_mask.append(m)
        n = np.zeros(j)
        w_mask.append(n)
    i, j = shapes[-1]
    m, n = np.ones((i,j)).reshape(-1), np.ones(j)
    w_mask.append(m)
    w_mask.append(n)
    w_mask = np.concatenate(w_mask)
    
    gmask = np.concatenate([w_mask, w_mask, tau_mask, tau_mask, tau_global_mask, tau_global_mask, tau_op_mask, tau_op_mask])
    
    return gmask    



def subnetwork_selection(task_id, model, model_mask, base_mask, score):
    ratios = [1-0.8**n for n in range(1, 22)]
    if base_mask is not None:
        temporary_mask_ = model_mask+base_mask
    else:
        temporary_mask_ = model_mask
    _, err_baseline = model.inference_engine.compute_test_ll(task_id, model.variational_params, model.X, model.y,\
                                                             temporary_mask_, num_samples=5)
    print(err_baseline)
    for i in range(len(ratios)):
        r = ratios[i]
        print(r)
        new_mask, finetune_mask = update_model_mask(model_mask, score, r)
        if base_mask is not None:
            temporary_mask_ = finetune_mask + base_mask
        else:
            temporary_mask_ = finetune_mask
        _, err = model.inference_engine.compute_test_ll(task_id, model.variational_params, model.X, model.y,\
                                                        temporary_mask_, num_samples=5)
        print(err)
        if np.abs(err-err_baseline) > 0.15:
            break
    if i == 0:
        r = 0
        print("The nodes have run out")
        new_mask = model_mask+base_mask
        finetune_mask = model_mask
    else:
        r = ratios[i-1]
        new_mask, finetune_mask = update_model_mask(model_mask, score, r)
    if base_mask is not None:
        task_specific_mask = finetune_mask + base_mask
    else:
        task_specific_mask = finetune_mask
    return new_mask,finetune_mask, task_specific_mask




