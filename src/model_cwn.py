import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import unbatch
from torch_scatter import scatter_add

from typing import List


class CWN(nn.Module):
    """ CW network model. """
    def __init__(self,
                 initial_cell_dims: List[int],
                 num_hidden: int,
                 num_layers: int):
        super().__init__()
        self.embed = [nn.Linear(cell_in, num_hidden) for cell_in in initial_cell_dims]
        self.layers = nn.ModuleList([CWNLayer(num_hidden) for _ in range(num_layers)])

    def forward(self,
                cell_features: List[torch.Tensor],
                boundary_index: List[torch.Tensor],
                upper_adj_index: List[torch.Tensor]):
        cell_features = [embed_layer(c) for c, embed_layer in zip(cell_features, self.embed)]

        cell_dims = []
        for cell_dim, c in enumerate(cell_features):
            cell_dims.extend([cell_dim] * c.shape[0])
        cell_dims = torch.tensor(cell_dims, dtype=torch.long, device=cell_features[0].device)

        for layer in self.layers:
            cell_features = layer(cell_features, cell_dims, boundary_index, upper_adj_index)

        return cell_features


class CWNLayer(nn.Module):
    """ CW network layer. """
    def __init__(self, num_hidden):
        super().__init__()
        self.boundary_message_mlp = nn.Linear(2 * num_hidden, num_hidden)
        self.upper_adj_message_mlp = nn.Linear(3 * num_hidden, num_hidden)
        self.h_update = nn.Linear(3 * num_hidden, num_hidden)

    def forward(self,
                cell_features: List[torch.Tensor],
                cell_dims: torch.Tensor,
                boundary_index: List[torch.Tensor],
                upper_adj_index: List[torch.Tensor]):
        try:
            assert len(cell_features) != 3
            assert len(boundary_index) == 2
            assert len(upper_adj_index) == 2
            node_features, edge_features, cycle_features = cell_features
        except AssertionError:
            raise NotImplementedError("CWN only supports cellular complexes with exactly three dimensions: "
                                      "(0d - vertices, "
                                      "1d - edges, "
                                      "2d - cycles). \n"
                                      "Provide a list of cell matrices for each cell dimension. "
                                      "There should be at least one cell in each dimension.")

        update_rows = []

        # --- node receiving messages ---
        # boundary
        node_boundary_messages = torch.zeros((node_features.shape[0], node_features.shape[1])).to(node_features.device)

        # upper adj
        rec, send, common = upper_adj_index[0]
        messages = torch.cat((node_features[rec], node_features[send], edge_features[common]), dim=1)
        node_upper_adj_messages = self.upper_adj_message_mlp(messages, dim=1)
        node_upper_adj_messages = scatter_add(node_upper_adj_messages, rec, dim=0, dim_size=node_features.shape[0])

        # prepare row for update transform
        node_messages = torch.cat((node_features, node_boundary_messages, node_upper_adj_messages), dim=1)
        update_rows.append(node_messages)

        # --- edge receiving messages ---
        # boundary
        rec, send = boundary_index[0]
        edge_boundary_messages = self.boundary_message_mlp(torch.cat((edge_features[rec], node_features[send]), dim=1))
        edge_boundary_messages = scatter_add(edge_boundary_messages, rec, dim=0, dim_size=edge_features.shape[0])

        # upper_adj
        rec, send, common = upper_adj_index[1]
        messages = torch.cat((edge_features[rec], edge_features[send], cycle_features[common]), dim=1)
        edge_upper_adj_messages = self.upper_adj_message_mlp(messages, dim=1)
        edge_upper_adj_messages = scatter_add(edge_upper_adj_messages, rec, dim=0, dim_size=edge_features.shape[0])

        # prepare row for update transform
        edge_messages = torch.cat((edge_features, edge_boundary_messages, edge_upper_adj_messages), dim=1)
        update_rows.append(edge_messages)

        # --- cycle receiving messages ---
        # boundary
        rec, send = boundary_index[1]
        cycle_boundary_messages = self.boundary_message_mlp(torch.cat((cycle_features[rec], edge_features[send]), dim=1))
        cycle_boundary_messages = scatter_add(cycle_boundary_messages, rec, dim=0, dim_size=cycle_features.shape[0])

        # upper_adj
        cycle_upper_adj_messages = torch.zeros((cycle_features.shape[0], cycle_features.shape[1])).to(cycle_features.device)

        # prepare row for update transform
        cycle_messages = torch.cat((cycle_features, cycle_boundary_messages, cycle_upper_adj_messages), dim=1)
        update_rows.append(cycle_messages)

        # --- update ---
        update_rows = torch.cat(update_rows, dim=0)
        cell_features_batched = self.h_update(update_rows)
        cell_features = unbatch(cell_features_batched, cell_dims)
        return cell_features


class CWNHead(nn.Module):
    """ Handles readout and final prediction for graph regression task from CWN model. """
    def __init__(self, num_hidden):
        super().__init__()
        self.predict = nn.Linear(2 * num_hidden, 1)

    def forward(self,
                cell_features: List[torch.Tensor],
                cell_batches: List[torch.Tensor]):
        all_cell_features = torch.cat(cell_features, dim=0)
        all_cell_batches = torch.cat(cell_batches, dim=0)
        graph_reprs = global_add_pool(all_cell_features, all_cell_batches)
        final_prediction = self.predict(graph_reprs)
        return final_prediction.squeeze(1)
