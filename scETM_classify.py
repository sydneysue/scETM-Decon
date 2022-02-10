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
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from scipy.sparse import csr_matrix

from batch_sampler import CellSampler, CellSamplerPool
from train_utils import get_kl_weight, save_embeddings, clustering, \
    get_train_instance_name, draw_embeddings, entropy_batch_mixing
from datasets import available_datasets, process_dataset
from arg_parser import parser
from model import scETM
from model.classifier import classifier

parser.add_argument('--model-path', type=str, default='/home/mcb/users/ssue1/scETM/results/unfiltered/train_set_scETM_trnGeneEmbDim0_batchScaling_normCells_time03_14-21_20_54/model-800', help='path to model')

args = parser.parse_args()
if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

args.dataset_str = Path(args.h5ad_path).stem

train_instance_name = "classifier_" + get_train_instance_name(args)
args.ckpt_dir = os.path.join(args.ckpt_dir, train_instance_name)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#Load datasets
train_adata = ad.read_h5ad(args.h5ad_path)
print('Training data has shape (n_obs x n_vars)', train_adata.shape)
train_adata = process_dataset(train_adata, args)


test_adata = ad.read_h5ad(args.test_path)
print('Testing adata has shape (n_obs x n_vars)', test_adata.shape)
test_adata = process_dataset(test_adata, args)

if args.parameters:
    with open(args.parameters_path, 'rb') as f:
        best_parameters = pickle.load(f)
        logging.info(f'Best Parameters: {best_parameters}')

#Load model
model = scETM(train_adata, args) 
if torch.cuda.is_available():
    model = model.to(torch.device('cuda:0'))

state_dict = torch.load(args.model_path)
del state_dict['gene_bias']
#del state_dict['global_bias']

model.load_state_dict(state_dict)


def train_model(classifier_model, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    val_acc_history = []

    #best_model_wts = copy.deepcopy(classifier_model.state_dict())
    best_acc = 0.0
    
    train_acc = []
    train_loss = []
    
    test_acc = []
    test_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                
                running_loss = 0.0
                running_corrects = 0
                
                classifier_model.train()  # Set model to training mode
                
                is_sparse = isinstance(train_adata.X, csr_matrix)
                if is_sparse:
                    library_size = train_adata.X.sum(1)
                else:
                    library_size = train_adata.X.sum(1, keepdims=True)

                cells  = train_adata.X
                cells = torch.FloatTensor(cells.todense() if is_sparse else cells)
                library_size = torch.FloatTensor(library_size)

                normed_cells = cells / library_size if args.norm_cells else cells
                normed_cells = normed_cells.to(device)
                
                cell_types = sorted(list(train_adata.obs.cell_types.unique()))
                cell_labels = torch.tensor(train_adata.obs.cell_types.apply(lambda x: cell_types.index(x)))
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    out = classifier_model(normed_cells).cpu()
                    _, preds = torch.max(out, 1)
                    loss = criterion(out, cell_labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * normed_cells.size(0)
                running_corrects += torch.sum(preds == cell_labels)
                
                epoch_train_loss = running_loss / len(normed_cells)
                epoch_train_acc = running_corrects.double() / len(normed_cells)
                
                train_loss.append(epoch_train_loss)
                train_acc.append(epoch_train_acc)
          
            else:
                classifier_model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                classifier_model.eval()   # Set model to evaluate mode
                
                is_sparse = isinstance(test_adata.X, csr_matrix)
                if is_sparse:
                    library_size = test_adata.X.sum(1)
                else:
                    library_size = test_adata.X.sum(1, keepdims=True)

                cells  = test_adata.X
                cells = torch.FloatTensor(cells.todense() if is_sparse else cells)
                library_size = torch.FloatTensor(library_size)

                normed_cells = cells / library_size if args.norm_cells else cells
                normed_cells = normed_cells.to(device)
                
                cell_types = sorted(list(test_adata.obs.cell_types.unique()))
                cell_labels = torch.tensor(test_adata.obs.cell_types.apply(lambda x: cell_types.index(x)))
                
                with torch.set_grad_enabled(phase == 'train'):
                    out = classifier_model(normed_cells).cpu()
                    _, preds = torch.max(out, 1)
                    loss = criterion(out, cell_labels)
                    
                running_loss += loss.item() * normed_cells.size(0)
                running_corrects += torch.sum(preds == cell_labels)
                
                epoch_test_loss = running_loss / len(normed_cells)
                epoch_test_acc = running_corrects.double() / len(normed_cells)
                
                test_loss.append(epoch_test_loss)
                test_acc.append(epoch_test_acc)
                

            epoch_loss = running_loss / len(normed_cells)
            epoch_acc = running_corrects.double() / len(normed_cells)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

#             # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(classifier_model.state_dict(), os.path.join(args.ckpt_dir, f'model-{epoch}'))
                #best_model_wts = copy.deepcopy(classifier_model.state_dict())

#         print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

    #model.load_state_dict(best_model_wts)
    return preds #model,val_acc_history

classifier_model = classifier(model, train_adata, args)
classifier_model = classifier_model.to(torch.device('cuda:0'))

#cell_types = sorted(list(adata.obs.cell_types.unique()))
#cell_labels = torch.tensor(adata.obs.cell_types.apply(lambda x: cell_types.index(x)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr)

hist = train_model(classifier_model, criterion, optimizer, args.n_epochs)

#save loss/accuracy
save_dict = dict(train_loss, train_acc, test_loss, test_acc)
import pickle
with open(os.path.join(args.ckpt_dir, 'log.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)
