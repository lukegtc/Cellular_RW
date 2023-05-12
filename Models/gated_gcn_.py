import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 

"""
    GatedGCN-LSPE
    
"""
# from layers.gatedgcn_layer import GatedGCNLayer
from gatedgcn_lspe_layer import GatedGCNLSPELayer
from mlp import MLPReadout

class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.pos_enc_dim = net_params['pos_enc_dim']
        
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # LSPE
        self.layers = nn.ModuleList([ GatedGCNLSPELayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, residual=self.residual) for _ in range(self.n_layers-1) ]) 
        self.layers.append(GatedGCNLSPELayer(hidden_dim, out_dim, dropout, self.batch_norm, residual=self.residual))
        
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        

        self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
        self.Whp = nn.Linear(out_dim+self.pos_enc_dim, out_dim)
    
        self.g = None              # For util; To be accessed in loss() function

        
    def forward(self, g, h, p, e, snorm_n):

        # node feature embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # positional embedding
        p = self.embedding_p(p) 
        
        # edge embedding
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)

    
        # convnets
        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)
            
        g.ndata['h'] = h
        
        if self.pe_init == 'rand_walk':
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            p = self.p_out(p)
            g.ndata['p'] = p
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p']**2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms)            
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p
        
            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'],g.ndata['p']),dim=-1))
            g.ndata['h'] = hp
        
        # readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        self.g = g # For util; To be accessed in loss() function
        
        return self.MLP_layer(hg), g
        
    def loss(self, scores, targets):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = nn.L1Loss()(scores, targets)
        
        
        return loss
