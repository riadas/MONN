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
from CPI_model import *

data_dir = "/home/riadas/enzyme-datasets/data/processed/"
processed_data_dir = "/home/riadas/MONN/src/enzyme_substrate_data/"
data_files = list(filter(lambda x : ".csv" in x, os.listdir(data_dir)))


def random_pairwise_pred(data_file="", count=1, threshold=0.5, max_iters=500):
    if data_file == "":
        data_file = random.choice(data_files)

    substrate_dict_file_name = data_file[:-4] + "_SUBSTRATES.pickle"
    with open(processed_data_dir + substrate_dict_file_name, 'rb') as handle:
        substrate_dict = pickle.load(handle)

    seq_dict_file_name = data_file[:-4] + "_SEQS.pickle"
    with open(processed_data_dir + seq_dict_file_name, 'rb') as handle:
        seq_dict = pickle.load(handle)

    net = load_model()
    
    iters = 0
    pred = None 
    while iters < max_iters:
        if iters % 50 == 0:
            print(iters)
        sub = random.choice(list(substrate_dict.keys()))
        sub_input = substrate_dict[sub]
        seq = random.choice(list(seq_dict.keys()))
        seq_input = seq_dict[seq]

        _, pairwise_pred = predict(net, sub_input, seq_input)
        if torch.count_nonzero(pairwise_pred > threshold) >= count:
            print("Found!")
            pred = pairwise_pred
            return pred, sub, seq, data_file
        iters += 1
    
    return pred, "", "", data_file 

def load_model(measure="KIKD", setting="new_compound"):
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
    net.load_state_dict(torch.load("models/new_compound_model"))
    return net

def predict(net, sub_input, seq_input, setting="new_compound"):
    fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat = sub_input
    vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process([[fatoms], [fbonds], [atom_nb], [bond_nb], [num_nbs_mat], torch.Tensor([seq_input])])
    # vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process([fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat, seq_input])

    # make prediction
    affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
    return affinity_pred, pairwise_pred 

def visualize_predicted_pairwise_matrix(pairwise_pred):
    plot = sns.heatmap(pairwise_pred[0])
    return plot

if __name__ == "__main__":
    pred, sub, seq = random_pairwise_pred()
    print(pred)
    print(sub)
    print(seq)
