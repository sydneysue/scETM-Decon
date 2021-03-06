import anndata
import scanpy as sc
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score


class BaseCellModel(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.trainable_gene_emb_dim = args.trainable_gene_emb_dim
        if not args.no_eval:
            self.n_labels = adata.obs.cell_types.nunique()
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.input_batch_id = args.input_batch_id
        self.batch_scaling = args.batch_scaling
        self.batch_size = args.batch_size
        self.mask_ratio = args.mask_ratio
        if self.mask_ratio < 0 or self.mask_ratio > 0.5:
            raise ValueError("Mask ratio should be between 0 and 0.5.")

        self.is_sparse = isinstance(adata.X, csr_matrix)
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)
        self.X = adata.X
        self.batch_indices = adata.obs.batch_indices.astype(int)

    def mask_gene_expression(self, cells):
        if self.mask_ratio > 0:
            return cells * (torch.rand_like(cells, device=self.device, dtype=torch.float32) * (1 - 2 * self.mask_ratio))
        else:
            return cells

    def get_cell_emb_weights(self, adata, weight_names):
        self.eval()

        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)

        if isinstance(weight_names, str):
            weight_names = [weight_names]

        if adata.n_obs > self.batch_size:
            weights = {name: [] for name in weight_names}
            for start in range(0, adata.n_obs, self.batch_size):
                X = adata.X[start: start + self.batch_size, :]
                if self.is_sparse:
                    X = X.todense()
                cells = torch.FloatTensor(X).to(self.device)
                library_size = torch.FloatTensor(self.library_size[start: start + self.batch_size]).to(self.device)
                data_dict = dict(cells=cells, library_size=library_size,
                    cell_indices=torch.arange(start, min(start + self.batch_size, adata.n_obs), device=self.device))
                if self.input_batch_id or self.batch_scaling:
                    batch_indices = torch.LongTensor(self.batch_indices[start: start + self.batch_size]).to(self.device)
                    data_dict['batch_indices'] = batch_indices
                fwd_dict = self(data_dict, dict(val=True))
                for name in weight_names:
                    weights[name].append(fwd_dict[name].detach().cpu())
            weights = {name: torch.cat(weights[name], dim=0).numpy() for name in weight_names}
        else:
            X = adata.X.todense() if self.is_sparse else adata.X
            cells = torch.FloatTensor(X).to(self.device)
            library_size = torch.FloatTensor(self.library_size).to(self.device)
            data_dict = dict(cells=cells, library_size=library_size, cell_indices=torch.arange(self.n_cells, device=self.device))
            if self.input_batch_id or self.batch_scaling:
                batch_indices = torch.LongTensor(self.batch_indices).to(self.device)
                data_dict['batch_indices'] = batch_indices
            fwd_dict = self(data_dict, dict(val=True))
            weights = {name: fwd_dict[name].detach().cpu().numpy() for name in weight_names}
        return weights

    @staticmethod
    def get_fully_connected_layers(n_input, hidden_sizes, args):
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(n_input, size))
            layers.append(nn.ReLU())
            if not args.no_bn:
                layers.append(nn.BatchNorm1d(size))
            if args.dropout_prob:
                layers.append(nn.Dropout(args.dropout_prob))
            n_input = size
        return nn.Sequential(*layers)
    
    @staticmethod
    def get_kl(mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)
