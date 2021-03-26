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

#Load dataset
adata = ad.read_h5ad(args.h5ad_path)
print('Bulk adata has shape (n_obs x n_vars)', adata.shape)
adata = process_dataset(adata, args)

#Load rho matrix to get intersection of genes
# mat = pd.read_csv(args.cellsig_genes, index_col=0)
# genes = sorted(list(set(mat.index).intersection(adata.var_names)))
# adata = adata[:, genes]
# print('After filtering out genes based on rho, bulk adata has shape (n_obs x n_vars) ', adata.shape)


#Load model
model = scETM(adata, args)
if torch.cuda.is_available():
    model = model.to(torch.device('cuda:0'))
    
state_dict = torch.load(args.model_path)
del state_dict['gene_bias']
    
model.load_state_dict(state_dict)

model.eval()

embeddings = model.get_cell_emb_weights()
for emb_name, emb in embeddings.items():
    adata.obsm[emb_name] = emb

save_embeddings(model, adata, embeddings, args)
