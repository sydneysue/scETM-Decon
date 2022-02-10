import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from scipy.sparse import csr_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class classifier(nn.Module):
    def __init__(self, model, adata: ad.AnnData, args,
                device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        
        # Define hyperparameters
        self.device = device
        self.num_celltypes = len(adata.obs.cell_types.unique()) 
        self.num_topics = args.n_topics
        self.model = model
        self.hidden_dim = args.hidden_sizes[-1]
        
        # Add final layer -> output for each cell type
        self.final = nn.Linear(self.hidden_dim, self.num_celltypes)
    
        #Run model
    def forward(self, normed_cells):
        q_delta = self.model.q_delta(normed_cells).to(self.device) #.cpu()
        #mu_q_delta = self.model.mu_q_delta(q_delta).cpu()
        out = self.final(q_delta).to(self.device)
        return out
