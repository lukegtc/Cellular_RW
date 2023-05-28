import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool


class GatedGCN(nn.Module):
    """ GatedGCN model. """
    def __init__(self, feat_in, num_hidden, num_layers, **kwargs):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.layers = nn.ModuleList([GatedGCNLayer(num_hidden) for _ in range(num_layers)])
        self.readout = nn.Sequential(nn.Linear(num_hidden, num_hidden//2),
                                     nn.ReLU(),
                                     nn.Linear(num_hidden // 2, 1))

    def forward(self, h, edge_index, batch):
        h = self.h_embed(h)

        for layer in self.layers:
            h = h + F.relu(layer(h, edge_index))

        h_agg = global_add_pool(h, batch)
        return self.readout(h_agg).squeeze()


class GatedGCNLayer(nn.Module):
    """ MP-GNN layer handling structural and positional embeddings. """
    def __init__(self, num_hidden):
        super().__init__()
        self.edge_gate_linear = nn.Linear(2*num_hidden, num_hidden)
        self.hp_send_layer = nn.Linear(num_hidden, num_hidden)
        self.hp_rec_layer = nn.Linear(num_hidden, num_hidden)

    def forward(self, h, edge_index):
        send, rec = edge_index
        eta = torch.sigmoid(self.edge_gate_linear(torch.cat([h[rec], h[send]], dim=1)))
        h_messages = eta * self.hp_send_layer(h[send])
        h_messages_agg = scatter_add(h_messages, rec, dim=0, dim_size=h.shape[0])
        h = self.hp_rec_layer(h) + h_messages_agg
        return h


class GatedGCNLSPE(nn.Module):
    """ MP-GNN model with learnable structural and positional embeddings. """
    def __init__(self, feat_in, pos_in, edge_feat_in, num_hidden, num_layers, **kwargs):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.e_embed = nn.Linear(edge_feat_in, num_hidden)
        self.p_embed = nn.Linear(pos_in, num_hidden)
        self.layers = nn.ModuleList([GatedGCNLSPELayer(num_hidden) for _ in range(num_layers)])
        self.readout = nn.Sequential(nn.Linear(2*num_hidden, num_hidden),
                                     nn.ReLU(),
                                     nn.Linear(num_hidden, 1))

    def forward(self, h, e, p, edge_index, batch):
        h = self.h_embed(h)
        e = self.e_embed(e)
        p = self.p_embed(p)

        for layer in self.layers:
            h_new, e_new, p_new = layer(h, e, p, edge_index)
            h = h + F.relu(h_new)
            e = e + F.relu(e_new)
            p = p + F.tanh(p_new)

        h_agg = global_add_pool(h, batch)
        p_agg = global_add_pool(p, batch)
        return self.readout(torch.cat((h_agg, p_agg), dim=1)).squeeze()


class GatedGCNLSPELayer(nn.Module):
    """ MP-GNN layer handling structural and positional embeddings. """
    def __init__(self, num_hidden):
        super().__init__()

        # self.bn = nn.BatchNorm1d(num_hidden)
        self.edge_gate_linear = nn.Linear(3*num_hidden, num_hidden)  # one matrix instead of three
        self.hp_send_layer = nn.Linear(2*num_hidden, num_hidden)
        self.hp_rec_layer = nn.Linear(2*num_hidden, num_hidden)
        self.p_send_layer = nn.Linear(num_hidden, num_hidden)
        self.p_rec_layer = nn.Linear(num_hidden, num_hidden)

    def forward(self, h, e, p, edge_index):
        send, rec = edge_index
        eta_hat = torch.sigmoid(self.edge_gate_linear(torch.cat([h[send], h[rec], e], dim=1)))
        eta = eta_hat/(eta_hat.sum(dim=1, keepdim=True))

        hp = torch.cat((h, p), dim=1)
        h_messages = eta * self.hp_send_layer(hp[send])
        h_messages_agg = scatter_add(h_messages, rec, dim=0, dim_size=h.shape[0])
        h = self.hp_rec_layer(hp) + h_messages_agg

        e = eta_hat

        p_messages = eta * self.p_send_layer(p[send])
        p_messages_agg = scatter_add(p_messages, rec, dim=0, dim_size=h.shape[0])
        p = self.p_rec_layer(h) + p_messages_agg
        return h, e, p
