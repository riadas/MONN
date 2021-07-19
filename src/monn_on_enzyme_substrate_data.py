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
from collections import defaultdict

data_dir = "/home/riadas/enzyme-datasets/data/processed/"
reference_structure_dir = "/home/riadas/enzyme-datasets/data/processed/structure_references/"
processed_data_dir = "/home/riadas/MONN/src/enzyme_substrate_data/"
data_files = list(filter(lambda x : ".csv" in x, os.listdir(data_dir)))

# from preprocessing_and_clustering
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6

word_dict = defaultdict(lambda: len(word_dict))
for aa in aa_list:
    word_dict[aa]
word_dict['X']

def Protein2Sequence(sequence, ngram=1):
    # convert sequence to CNN input
    sequence = sequence.upper()
    word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
    output = []
    for word in word_list:
        if word not in aa_list:
            output.append(word_dict['X'])
        else:
            output.append(word_dict[word])
    if ngram == 3:
        output = [-1]+output+[-1] # pad
    return np.array(output, np.int32)

# end from preprocessing_and_clustering

def random_pairwise_pred(data_file="", count=1, threshold=0.5, max_iters=500):
    if data_file == "":
        data_file = random.choice(data_files)

    substrate_dict_file_name = data_file[:-4] + "_SUBSTRATES.pickle"
    with open(processed_data_dir + substrate_dict_file_name, 'rb') as handle:
        substrate_dict = pickle.load(handle)

    seq_dict_file_name = data_file[:-4] + "_SEQS.pickle"
    with open(processed_data_dir + seq_dict_file_name, 'rb') as handle:
        seq_dict = pickle.load(handle)

    net = load_model(setting="new_new")
    
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
            pred = torch.round(pairwise_pred)
            return pred, sub, seq, data_file, (pred[0] > 0).nonzero()
        iters += 1
    
    return pred, "", "", data_file, [] # last return elt is list of nonzero tuples

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
    net.load_state_dict(torch.load("models/"+setting+"_model"))
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

def check_interaction_pred_validity():
    net = load_model("KIKD", "new_new")
    results = {}
    for data_file in data_files:
        print(data_file)
        # ----- find proteins for which active site shells have been computed -----
        ## get protein sequence at top of every reference structure file
        data_file_prefix = data_file.split("_")[0]
        reference_structure_files = [file_name for file_name in os.listdir(reference_structure_dir) if data_file_prefix in file_name]
        reference_seqs = list(set([open(reference_structure_dir+f, "r").readlines()[1] for f in reference_structure_files]))

        ## get substrate and sequence featurization dicts
        dict_file_prefix = ""
        if "_SUBSTRATES.pickle" in data_file:
            dict_file_prefix = data_file.replace("_SUBSTRATES.pickle", "")
        else:
            dict_file_prefix = data_file.replace("_SEQS.pickle", "")

        substrate_dict_file_name = dict_file_prefix + "_SUBSTRATES.pickle"
        with open(processed_data_dir + substrate_dict_file_name, 'rb') as handle:
            substrate_dict = pickle.load(handle)

        seq_dict_file_name = dict_file_prefix + "_SEQS.pickle"
        with open(processed_data_dir + seq_dict_file_name, 'rb') as handle:
            seq_dict = pickle.load(handle)

        # ## only use protein sequences that are already present in sequence featurization dict
        # ## check if seq is in seq_dict or if (seq - padding) is in seq_dict
        # reference_seqs = [s for s in reference_seqs if (s in seq_dict) or functools.reduce(lambda a,b: a or b, list(map(lambda k: k in s, list(seq_dict.keys()))))]

        # for i in range(len(reference_seqs)):
        #     seq = reference_seqs[i]
        #     if seq not in seq_dict:
        #         new_seq = list(filter(lambda k: k in seq, list(seq_dict.keys())))[0]
        #         reference_seqs[i] = new_seq

        # print(len(reference_seqs))
        # use net to predict the interaction matrix for each protein and each substrate in the dataset
        results[data_file_prefix] = []
        for seq in reference_seqs:
            seq_input = None 
            if seq in seq_dict:
              seq_input = seq_dict[seq]
            else:
              seq_input = Protein2Sequence(seq, ngram=1)
            for sub in substrate_dict:
                sub_input = substrate_dict[sub]
                affinity_pred, pairwise_pred = predict(net, sub_input, seq_input)
                positions = (torch.round(pairwise_pred)[0] > 0).nonzero()
                for angstroms in range(1, 50):
                    angstrom_file = open(reference_structure_dir+data_file_prefix+"_reference_" + str(angstroms) + ".txt", "r")
                    lines = angstrom_file.readlines()[2:]
                    true_positions = list(map(lambda l: int(l.split(",")[1:]), lines))
                    intersection = set(positions).intersection(set(true_positions))
                    if len(intersection) > 0:
                        results[data_file_prefix].append((angstroms, intersection))
                        print((angstroms, intersection))
        return results
    
if __name__ == "__main__":
    pred, sub, seq = random_pairwise_pred()
    print(pred)
    print(sub)
    print(seq)
