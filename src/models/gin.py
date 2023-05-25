import torch.nn as nn
from torch_scatter import scatter_add


class GIN(nn.Module):
    """ GIN model. """
    def __init__(self, feat_in, num_hidden, num_layers):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.layers = nn.ModuleList([GINLayer(num_hidden) for _ in range(num_layers)])
        self.predict = nn.Linear(num_hidden, 1)
        self.bn = nn.BatchNorm1d(num_hidden)

    def forward(self, h, edge_index):
        h = self.h_embed(h)

        for layer in self.layers:
            out = layer(h, edge_index)
            out = nn.functional.relu(self.bn(out))
            h = out

        return h


class GINLayer(nn.Module):
    """ GIN-0 layer. """
    def __init__(self, num_hidden):
        super().__init__()
        self.h_update = nn.Linear(num_hidden, num_hidden)

    def forward(self, h, edge_index):
        send, rec = edge_index

        # h_messages = h
        h_messages = h[send]
        h_messages_agg = scatter_add(h_messages, rec, dim=0, dim_size=h.shape[0])
        h = self.h_update(h + h_messages_agg)

        return h
