import torch
import torch.nn as nn
from torch_geometric.datasets import ZINC
from torch_geometric.nn import global_add_pool
from torch_geometric.transforms import AddRandomWalkPE
from torch_scatter import scatter_add
import torch
import torch.nn as nn
import torch.optim as op
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
import pytorch_lightning as pl

from metrics import accuracy_TU
from transform import AddRandomWalkPE

class MPGNN(nn.Module):
    def __init__(self, feat_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.embed = nn.Linear(feat_in, num_hidden)
        self.edge_embed = nn.Linear(edge_feat_in, num_hidden)
        self.layers = nn.ModuleList([MPGNNLayer(num_hidden) for _ in range(num_layers)])

    def forward(self, h, e, edge_index, batch):
        h = self.embed(h)
        e = self.edge_embed(e)

        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        h = global_add_pool(h, batch)
        return h


class MPGNN_PE(nn.Module):
    def __init__(self, feat_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.embed = nn.Linear(feat_in, num_hidden)
        self.edge_embed = nn.Linear(edge_feat_in, num_hidden)
        self.layers = nn.ModuleList([MPGNNLayer(num_hidden) for _ in range(num_layers)])

    def forward(self, graph):
        h, e, edge_index, batch, p = graph.x, graph.edge_attr, graph.edge_index, graph.batch, graph.p
        h = self.embed(torch.cat(h,p), dim=1)
        e = self.edge_embed(e)

        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        h = global_add_pool(h, batch)
        return h


class MPGNNLayer(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.message_mlp = nn.Linear(3 * num_hidden, num_hidden)
        self.h_update = nn.Linear(2 * num_hidden, num_hidden)
        self.e_update = nn.Linear(3 * num_hidden, num_hidden)

    def forward(self, h, e, edge_index):
        send, rec = edge_index
        h_send, h_rec = h[send], h[rec]
        messages = self.message_mlp(torch.cat((h_send, h_rec, e), dim=1))

        # pass dim_size to include nodes with no edges going in (they only send messages, never receive)
        messages_agg = scatter_add(messages, rec, dim=0, dim_size=h.shape[0])

        h = self.h_update(torch.cat((h, messages_agg), dim=1))
        e = self.e_update(torch.cat((h_send, h_rec, e), dim=1))

        return h, e


class MPGNNHead(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.predict = nn.Linear(num_hidden, 1)

    def forward(self, h):
        final_prediction = self.predict(h)
        return final_prediction.squeeze(1)


class LSPE_MPGNN(nn.Module):
    def __init__(self, feat_in, pos_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.e_embed = nn.Linear(edge_feat_in, num_hidden)
        self.p_embed = nn.Linear(pos_in, num_hidden)
        self.layers = nn.ModuleList([LSPE_MPGNNLayer(num_hidden) for _ in range(num_layers)])
        self.predict = nn.Linear(2*num_hidden, 1)

    def forward(self, h, e, p, edge_index, batch):
        h = h.float()
        h = self.h_embed(h)
        # h = self.h_embed(torch.cat(h,p), dim =1)
        e = e.unsqueeze(1).float()
        e = self.e_embed(e)

        p = self.p_embed(p)

        for layer in self.layers:
            h, e, p = layer(h, e, p, edge_index)

        h = global_add_pool(h, batch)
        p = global_add_pool(p, batch)

        return h, p


class LSPE_MPGNNLayer(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.h_message_mlp = nn.Linear(3 * num_hidden, num_hidden)
        self.h_update = nn.Linear(2 * num_hidden, num_hidden)
        self.e_update = nn.Linear(3 * num_hidden, num_hidden)
        self.p_update = nn.Linear(2 * num_hidden, num_hidden)
        self.p_message_mlp = nn.Linear(3 * num_hidden, num_hidden)

    def forward(self, h, e, p, edge_index):
        send, rec = edge_index

        hp_send = torch.cat((h[send], p[send]), dim=1)
        hp_rec = torch.cat((h[rec], p[rec]), dim=1)
        h_messages = self.h_message_mlp(torch.cat((hp_send, hp_rec, e), dim=1))
        h_messages_agg = scatter_add(h_messages, rec, dim=0, dim_size=h.shape[0])
        h = self.h_update(torch.cat((h, h_messages_agg), dim=1))

        e = self.e_update(torch.cat((h[send], h[rec], e), dim=1))

        p_messages = self.p_message_mlp(torch.cat((p[send], p[rec], e), dim=1))
        p_messages_agg = scatter_add(p_messages, rec, dim=0, dim_size=p.shape[0])
        p = self.p_update(torch.cat((p, p_messages_agg), dim=1))

        return h, e, p


class LSPE_MPGNNHead(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.predict = nn.Linear(2*num_hidden, 1)

    def forward(self, h, p):
        final_prediction = self.predict(torch.cat((h, p), dim=1))
        return final_prediction.squeeze(1)
