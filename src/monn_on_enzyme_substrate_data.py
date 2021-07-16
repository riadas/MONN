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
for data_file in data_files:
    substrate_dict_file_name = data_file[:-4] + "_SUBSTRATES.pickle"
    seq_dict_file_name = data_file[:-4] + "_SEQS.pickle"
    with open(data_dir + substrate_dict_file_name, 'rb') as handle:
        substrate_dict = pickle.load(handle)

    with open(data_dir + seq_dict_file_name, 'rb') as handle:
        seq_dict = pickle.load(handle)
    
    

# convert RDKit mol into graph form for input into model
fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat = Mol2Graph(substrate_mol)
sequence_input = Protein2Sequence(seq, ngram=1)

final form of model input data
vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process([fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat, sequence_input])

# load trained MONN model 
measure = "KIKD"
setting = "new_compound"

## define hyperparameters
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
    #k_head, kernel_size, hidden_size1, hidden_size2 = 1, 7, 128, 128
para_names = ['GNN_depth', 'inner_CNN_depth', 'DMA_depth', 'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2']
params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]

init_A, init_B, init_W = loading_emb(measure)
net = Net(init_A, init_B, init_W, params)
net.cuda()
net.load_state_dict(torch.load("model_rep_0_fold_3"))

# make prediction
affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)