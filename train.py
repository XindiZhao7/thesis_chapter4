#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time, os
import numpy as np
import joblib
from src.load_data import classification_data
from src.factorized_approximation import FactorizedHierarchicalInvGamma as inference_engine
from src.hs_bnn import HSBnn, fit, teacher_network_selection
from src.prune import importance_estimation, update_gradient_mask, subnetwork_selection, update_task_specific_params_only


#### ARCH Params ##########
num_hidden_layers = 2
num_nodes = 256

#### LEARNING Params ###########
batch_size = 100
learning_rate = 0.0001
num_epochs = 50
num_nodes_list = [num_nodes for i in np.arange(num_hidden_layers)]     
layer_sizes = [784] + num_nodes_list + [2]
teacher_no = 100
base_mask = None
# In[2]:


num_tasks = 5

for t in range(num_tasks):
    print("Task {0}".format(t))
    x_train, y_train, x_test, y_test = classification_data(t)
    if t == 0:
        last_mlp = None
        mask = np.ones(np.sum(layer_sizes[1:]))
        gradient_mask = update_gradient_mask(mask, layer_sizes)
        print("Num Epochs {0} {1}".format(num_epochs, x_train.shape[0]))
        mlp = HSBnn(t, last_mlp, layer_sizes, x_train, y_train, x_test, y_test, inference_engine, mask, batch_size)
        mlp = fit(mlp, n_epochs=num_epochs, mask=gradient_mask, l_rate=learning_rate)
        score = importance_estimation(mlp)
        new_mask,finetune_mask, task_specific_mask = subnetwork_selection(t, mlp, mask, base_mask, score)
        mlp.mask = finetune_mask
        mlp.task_specific_mask = task_specific_mask
        new_gradient_mask = update_gradient_mask(finetune_mask, layer_sizes)
        mlp = fit(mlp, n_epochs=50, mask=new_gradient_mask, l_rate=learning_rate)
        mlp.mask = new_mask
    else:
        last_mlp = joblib.load("./results/Classification/task_{0}/hsbnn_{1}.pkl".format(t-1, num_nodes))
        mask = 1-last_mlp.mask
        gradient_mask = update_gradient_mask(mask, layer_sizes)
        print("Num Epochs {0} {1}".format(num_epochs, x_train.shape[0]))
        mlp = HSBnn(t, last_mlp, layer_sizes, x_train, y_train, x_test, y_test, inference_engine, mask, batch_size)
        teacher = teacher_network_selection(t, last_mlp, x_train, y_train)
        if teacher is not None:
            base_mask, teacher_no = teacher[0], teacher[1]
            mlp.mask = base_mask + mask
        mlp = fit(mlp, n_epochs=num_epochs, mask=gradient_mask, l_rate=learning_rate, t_id=teacher_no)
        score = importance_estimation(mlp)
        new_mask,finetune_mask, task_specific_mask = subnetwork_selection(t, mlp, mask, base_mask, score)
        mlp.mask = task_specific_mask
        mlp.task_specific_mask = task_specific_mask
        new_gradient_mask = update_gradient_mask(finetune_mask, layer_sizes)
        mlp = fit(mlp, n_epochs=50, mask=new_gradient_mask, l_rate=learning_rate)
        mlp.mask = new_mask
    save_path = "./results/Classification/task_{0}".format(t)
    os.makedirs(save_path, exist_ok=True)
    save_name = "{0}/hsbnn_{1}.pkl".format(save_path, num_nodes)    
    joblib.dump(mlp, save_name)
    
    nodes_num = np.sum(finetune_mask)-2
    print("{0} nodes are used for task {1}".format(nodes_num, t))

    for tt in range(t+1):
        _, _, x_test, y_test = classification_data(tt)
        _, err = mlp.compute_optimal_test_ll(tt, x_test, y_test)
        print(tt)
        print(err)

    del mlp
    