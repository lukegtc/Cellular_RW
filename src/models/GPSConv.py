from torch_geometric.nn.conv import GPSConv, GINEConv
import torch.nn as nn
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch_geometric.nn import global_add_pool
from typing import Any, Dict, Optional




# https://arxiv.org/abs/2205.12454
class GPSConvWrapper(nn.Module):
    def __init__(self, feat_in, num_hidden: int,num_layers: int,edge_feat_in: int = 0):
        super().__init__()
        channels = num_hidden
        self.h_embed = nn.Linear(feat_in, channels)


        self.convs = ModuleList()
        for _ in range(num_layers):
            nn_layer = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            gineconv = False
            if gineconv:
                conv_1 = GINEConv(nn_layer)
            else:
                conv_1 = None
            conv = GPSConv(channels, conv_1, heads=4)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )


    def forward(self, h, edge_index, batch):

        x = self.h_embed(h)


        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, batch=batch)
        x = global_add_pool(x, batch)
        return self.mlp(x)

class GPSConvLSPEWrapper(nn.Module):
    def __init__(self, feat_in, pos_in,num_hidden: int,num_layers: int,edge_feat_in: int = 0):
        super().__init__()
        channels = num_hidden
        self.h_embed = nn.Linear(feat_in, channels)
        self.e_embed = nn.Linear(edge_feat_in, num_hidden)
        self.p_embed = nn.Linear(pos_in, num_hidden)



        self.convs = ModuleList()
        for _ in range(num_layers):
            nn_layer = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            gineconv = True
            if gineconv:
                conv_1 = GINEConv(nn_layer)
            else:
                conv_1 = None
            conv = GPSConv(channels, conv_1, heads=4)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

    def forward(self, h, e,p, edge_index, batch):

        x = self.h_embed(h)

        e = self.e_embed(e)

        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, batch=batch, edge_attr=e)
        x = global_add_pool(x, batch)
        return self.mlp(x)



