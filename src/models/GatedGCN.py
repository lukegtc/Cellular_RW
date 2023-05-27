import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import linalg
from torch_geometric.utils import unbatch
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool


class GatedGCN_LSPE(nn.Module):
    """ MP-GNN model with learnable structural and positional embeddings. """
    def __init__(self, feat_in, pos_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.e_embed = nn.Linear(edge_feat_in, num_hidden)
        self.p_embed = nn.Linear(pos_in, num_hidden)
        self.layers = nn.ModuleList([GatedGCN_LSPELayer(num_hidden) for _ in range(num_layers)])
        self.readout = nn.Sequential(nn.Linear(2*num_hidden, num_hidden), nn.ReLU(),nn.Linear(num_hidden , num_hidden//2),nn.ReLU(), nn.Linear(num_hidden // 2, 1))

    def forward(self, h, e, p, edge_index,batch):

        h = self.h_embed(h)
        e = self.e_embed(e)
        p = self.p_embed(p)

        for layer in self.layers:
            h, e, p = layer(h, e, p, edge_index)

        h_agg = global_add_pool(h, batch)
        p_agg = global_add_pool(p, batch)
        hep = torch.cat((h_agg, p_agg), dim=1)
        out = self.readout(hep).squeeze()
        return out,p_agg


class GatedGCN_LSPELayer(nn.Module):
    """ MP-GNN layer handling structural and positional embeddings. """
    def __init__(self, num_hidden):
        super().__init__()

        self.bn = nn.BatchNorm1d(num_hidden)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_hidden, num_hidden)
        self.hp_send_layer = nn.Linear(2*num_hidden, num_hidden)
        self.hp_rec_layer = nn.Linear(2*num_hidden, num_hidden)
        self.p_layer_1 = nn.Linear(num_hidden, num_hidden)
        self.p_layer_2 = nn.Linear(num_hidden, num_hidden)

    def forward(self, h, e, p, edge_index):

        send, rec = edge_index
        eta = torch.sigmoid(self.linear(h[send]) + self.linear(h[rec])+self.linear(e))
        eta_new = eta/(eta.sum(dim=1, keepdim=True))
        hp_send = self.hp_send_layer(torch.cat((h[send], p[send]), dim=1))
        hp_rec = self.hp_rec_layer(torch.cat((h[rec], p[rec]), dim=1))

        h[send] = h[send]+self.relu(self.bn(hp_send + scatter_add(hp_rec*eta_new,rec, dim=0, dim_size=hp_rec.shape[0])))
        e = e +self.relu(self.bn(eta))
        linear_p = self.p_layer_2(p[rec])
        p[send] = p[send]+F.tanh(self.p_layer_1(p[send])+scatter_add(linear_p*eta_new,rec, dim=0, dim_size=linear_p.shape[0]))


        return h, e, p
