from argparse import ArgumentError
import os
import sys
import time
import psutil
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
from torch import optim

from batch_sampler import CellSampler, CellSamplerPool
from train_utils import get_kl_weight, save_embeddings, clustering, \
    get_train_instance_name, draw_embeddings, entropy_batch_mixing
from datasets import available_datasets, process_dataset
from arg_parser import parser
from model import scETM

parser.add_argument('--model_path', type=str, default='/home/mcb/users/ssue1/scETM/results/unfiltered/train_set_scETM_trnGeneEmbDim0_batchScaling_normCells_time03_14-21_20_54/model-800', help='path to model')

# parser.add_argument('--cellsig_genes', type=str, default='/home/mcb/users/ssue1/DECON/data/pathways/top_rho.csv', help='path to rho')

args = parser.parse_args()
if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        
train_instance_name = get_train_instance_name(args)
args.ckpt_dir = os.path.join(args.ckpt_dir, train_instance_name)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#Load dataset
adata = ad.read_h5ad(args.h5ad_path)
print('Bulk adata has shape (n_obs x n_vars)', adata.shape)
adata = process_dataset(adata, args)

if args.parameters:
    with open(args.parameters_path, 'rb') as f:
        best_parameters = pickle.load(f)
        logging.info(f'Best Parameters: {best_parameters}')

#Load model
model = scETM(adata, args, best_parameters.parameters)
if torch.cuda.is_available():
    model = model.to(torch.device('cuda:0'))

state_dict = torch.load(args.model_path)
del state_dict['gene_bias']
del state_dict['global_bias']
#for k in list(state_dict.keys()):
#    if k.startswith('cell_type_clf'):
#        del state_dict[k]

model.load_state_dict(state_dict)

is_sparse = isinstance(adata.X, csr_matrix)
if is_sparse:
    library_size = adata.X.sum(1)
else:
    library_size = adata.X.sum(1, keepdims=True)

cells  = adata.X
cells = torch.FloatTensor(cells.todense() if is_sparse else cells)
library_size = torch.FloatTensor(library_size)

normed_cells = cells / library_size if args.norm_cells else cells
normed_cells = normed_cells.to(device)

with torch.no_grad():
    model.eval()

    q_delta = model.q_delta(normed_cells)
    mu_q_delta = model.mu_q_delta(q_delta)
    theta = F.softmax(mu_q_delta, dim=-1)
    print(f'Theta: {theta}')
    decon = theta @ model.alpha
    print(f'Alpha: {model.alpha}')
    p = F.softmax(decon, dim=1)
    print(f'Deconvolved proportions: {p}')

    new_dict = dict(
            alpha = model.alpha.detach().cpu().numpy(),
            theta = theta)
    import pickle
    with open(os.path.join(args.ckpt_dir, 'new_embeddings.pkl'), 'wb') as f:
        pickle.dump(new_dict, f)
    embeddings = model.get_cell_emb_weights(adata)
    for emb_name, emb in embeddings.items():
        adata.obsm[emb_name] = emb
    save_embeddings(model, adata, embeddings, args)
