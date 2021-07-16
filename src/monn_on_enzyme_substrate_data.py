from collections import defaultdict
import os
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt
import random
# from pdbbind_utils.py import batch_data_process

data_dir = "/Users/sdas/ria-code/enzyme-datasets/data/processed/"
data_files = list(filter(lambda x : ".csv" in x, os.listdir(data_dir)))

def random_pairwise_pred(data_file="", max_iters=500):
    if data_file == "":
        data_file = random.choice(data_files)

    substrate_dict_file_name = data_file[:-4] + "_SUBSTRATES.pickle"
    with open(data_dir + substrate_dict_file_name, 'rb') as handle:
        substrate_dict = pickle.load(handle)

    seq_dict_file_name = data_file[:-4] + "_SEQS.pickle"
    with open(data_dir + seq_dict_file_name, 'rb') as handle:
        seq_dict = pickle.load(handle)

    net = load_model()
    
    iters = 0
    pred = None 
    while iters < max_iters:
        sub_input = substrate_dict[random.choice(list(substrate_dict.keys()))]
        sequence_input = seq_dict[random.choice(list(seq_dict.keys()))]

        _, pairwise_pred = predict(net, sub_input, seq_input)
        if np.count_nonzero(pairwise_pred > 0.5) > 0:
            pred = pairwise_pred
            break
    
    return pred 

def load_model(setting="new_compound"):
    # define hyperparameters
    GNN_depth, inner_CNN_depth, DMA_depth = 4, 2, 2
    if setting == 'new_compound':
        n_fold = 5
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 2, 7, 128, 128
    elif setting == 'new_protein':
        n_fold = 5
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 1, 5, 128, 128
    elif setting == 'new_new':
        n_fold = 9
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 1, 7, 128, 128
    params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]

    # load model
    init_A, init_B, init_W = loading_emb(measure)
    net = Net(init_A, init_B, init_W, params)
    net.cuda()
    net.load_state_dict(torch.load("model_rep_0_fold_3"))
    return net

def predict(net, sub_input, seq_input, net, setting="new_compound"):
    fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat = sub_input
    vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process([fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat, sequence_input])

    # make prediction
    affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
    return affinity_pred, pairwise_pred 

def visualize_predicted_pairwise_matrix(pairwise_pred):
    plot = sns.heatmap(pairwise_pred)
    return plot

