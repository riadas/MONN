from collections import defaultdict
import os
import pickle
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt
import random
from preprocessing_and_clustering import Mol2Graph, Protein2Sequence 
from pdbbind_utils.py import batch_data_process

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
word_dict = defaultdict(lambda: len(word_dict))

data_dir = "/Users/sdas/ria-code/enzyme-datasets/data/processed/"
data_files = list(filter(lambda x : ".csv" in x, os.listdir(data_dir)))

data_file = "aminotransferase_binary.csv"
df = pd.read_csv(os.path.join(data_dir, data_file), index_col=0)
print(df.columns)

substrates = list(pd.unique(df["SUBSTRATES"]))
seqs = list(pd.unique(df["SEQ"]))

substrate = random.choice(substrates)
seq = random.choice(seqs)

# get RDKit mol
substrate_mol = Chem.MolFromSmiles(substrate)

# convert RDKit mol into graph form for input into model
fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat = Mol2Graph(substrate_mol)
sequence_input = Protein2Sequence(seq, ngram=1)

# final form of model input data
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