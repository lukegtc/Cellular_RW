import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool



class GIN(nn.Module):
    """ GIN model. """
    def __init__(self, feat_in, num_hidden, num_layers):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.layers = nn.ModuleList([GINLayer(num_hidden) for _ in range(num_layers)])
        self.readout = nn.Sequential(nn.Linear(num_hidden, num_hidden // 2), nn.ReLU(), nn.Linear(num_hidden // 2, 1))

    def forward(self, h, edge_index, batch):
        h = self.h_embed(h)

        for layer in self.layers:
            h = h + F.relu(layer(h, edge_index))

        h_agg = global_add_pool(h, batch)

        return self.readout(h_agg).squeeze()


class GINLayer(nn.Module):
    """ GIN-0 layer. """
    def __init__(self, num_hidden):
        super().__init__()
        self.h_update = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden))

    def forward(self, h, edge_index):
        send, rec = edge_index

        h_messages = h[send]
        h_messages_agg = scatter_add(h_messages, rec, dim=0, dim_size=h.shape[0])
        h = self.h_update(h + h_messages_agg)

        return h



# class GIN(nn.Module):
#     """ GIN model. """
#     def __init__(self, feat_in, num_hidden, num_layers):
#         super().__init__()
#         self.h_embed = nn.Linear(feat_in, num_hidden)
#         self.layers = nn.ModuleList([GINLayer(num_hidden) for _ in range(num_layers)])
#         self.mlp = nn.Linear(num_hidden, num_hidden)
#         self.predict = nn.Linear(num_hidden, 1)
#         self.bn = nn.BatchNorm1d(num_hidden)
#         self.bn2 = nn.BatchNorm1d(num_hidden)
#
#     def forward(self, h, edge_index):
#         h = self.h_embed(h)
#
#         hidden_states = [h]
#
#         for layer in self.layers:
#             out = layer(h, edge_index)
#             out = nn.functional.relu(self.bn(out))
#             out = nn.functional.relu(self.bn2(self.mlp(out)))
#             # out = layer(out, edge_index)
#             # out = self.bn(out)
#             h = h + out
#             hidden_states.append(h)
#
#         return hidden_states
#
#
# class GINLayer(nn.Module):
#     """ GIN-0 layer. """
#     def __init__(self, num_hidden):
#         super().__init__()
#         self.h_update = nn.Linear(num_hidden, num_hidden)
#
#     def forward(self, h, edge_index):
#         send, rec = edge_index
#
#         h_messages = h[send]
#         h_messages_agg = scatter_add(h_messages, rec, dim=0, dim_size=h.shape[0])
#         h = self.h_update(h + h_messages_agg)
#
#         return h
#
#
# class GINHead(nn.Module):
#     """ Handles readout and final prediction from standard MP-GNN model. """
#     def __init__(self, hidden_dim, num_hidden_states):
#         super().__init__()
#         self.mlp = nn.Linear(hidden_dim * num_hidden_states, hidden_dim)
#         self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.predict = nn.Linear(hidden_dim, 1)
#
#     def forward(self, hidden_states, h_batch):
#         intermediate_graph_reprs = []
#         for h in hidden_states:
#             intermediate_graph_reprs.append(global_add_pool(h, h_batch))
#         h = torch.cat(intermediate_graph_reprs, dim=1)
#
#         h = nn.functional.relu(self.bn1(self.mlp(h)))# one hidden layer
#         out = nn.functional.relu(self.bn2(self.mlp2(h)))
#         h = h + out
#         final_prediction = self.predict(h)  # predict
#         return final_prediction.squeeze(1)
