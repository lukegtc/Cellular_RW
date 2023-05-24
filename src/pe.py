from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_self_loop_attr,
    scatter,
    to_edge_index,
    to_torch_coo_tensor
)
import networkx as nx

from cellular import CellularComplex


class AddRandomWalkPE(BaseTransform):
    def __init__(self, walk_length: int, attr_name: Optional[str] = None):
        self.walk_length = walk_length
        self.attr_name = 'random_walk_pe' if attr_name is None else attr_name

    def __call__(self, data: Data) -> Data:
        adj = self.compute_rw_matrix(data.edge_index, data.edge_weight)
        out = adj
        pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=data.num_nodes)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_self_loop_attr(*to_edge_index(out), num_nodes=data.num_nodes))
        pe = torch.stack(pe_list, dim=-1)
        data.pe_features = pe
        return data

    @staticmethod
    def compute_rw_matrix(edge_index, edge_weight):
        # we need to assert that all edge weights are positive
        edge_weights = edge_weight
        assert torch.all(edge_weights > 0)

        # weighted in-degree of each node
        edge_indices = edge_index
        rec = edge_indices[0]
        # we assume that the graph has no isolated vertices
        num_nodes = max(rec) + 1
        node_deg = scatter(edge_weights, rec, dim_size=num_nodes, reduce='sum')
        pos_idx = torch.where(node_deg > 0)
        node_deg[pos_idx] = 1.0 / node_deg[pos_idx]

        # we used to_torch_csr_tensor before, but it gives Runtime Error if you don't have MKL installed
        # I couldn't install MKL bc it's for Intel processors, so I changed it to COO tensor
        # The sparse tensor is unpacked at every step of the loop, so it should give the same result
        adj = to_torch_coo_tensor(edge_indices, edge_weights, size=(num_nodes, num_nodes))

        return adj


class AddCellularRandomWalkPE(AddRandomWalkPE):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """

    def __init__(self, walk_length: int,
                 attr_name: Optional[str] = None):
        super().__init__(walk_length)
        self.attr_name = 'cellular_random_walk_pe' if attr_name is None else attr_name

    def __call__(self, data: Data):
        eg_edge_index, eg_edge_weight = self.construct_extended_graph(data)
        new_data = Data(edge_index=eg_edge_index,
                        edge_weight=eg_edge_weight)
        data_with_pe = super().__call__(new_data)
        pe = data_with_pe.__getattr__(self.attr_name)
        data.__setattr__(self.attr_name, pe)
        return data

    @staticmethod
    def construct_extended_graph(data: Data):
        edge_index = []
        edge_weight = []

        # add boundary connections as new edges
        # replace edge ids for each boundary connection (vertex ids are right)
        eg_edge_boundary_ids = data.boundary_index[1]
        eg_edge_ids = torch.tensor(range(data.num_nodes, data.num_nodes + data.num_edges), dtype=torch.long)
        eg_edge_boundary_ids[0, :] = eg_edge_ids[eg_edge_boundary_ids[0, :]]
        edge_index.append(eg_edge_boundary_ids)
        edge_weight.append(torch.ones(eg_edge_boundary_ids.shape[1]))

        # replace cycle ids and edge ids in boundary connections
        eg_cycle_boundary_ids = data.boundary_index[2]
        eg_cycle_ids = torch.tensor(
            range(data.num_nodes + data.num_edges, data.num_nodes + data.num_edges + data.num_cycles),
            dtype=torch.long)
        eg_cycle_boundary_ids[0, :] = eg_cycle_ids[eg_cycle_boundary_ids[0, :]]
        eg_cycle_boundary_ids[1, :] = eg_edge_ids[eg_cycle_boundary_ids[1, :]]
        edge_index.append(eg_cycle_boundary_ids)
        edge_weight.append(torch.ones(eg_cycle_boundary_ids.shape[1]))

        edge_index = torch.cat(edge_index, dim=1).T
        edge_weight = torch.cat(edge_weight, dim=0).reshape(-1, 1)

        return edge_index, edge_weight