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

from .cellular import CellularComplexData
import networkx as nx


class AddRandomWalkPE(BaseTransform):
    def __init__(self, walk_length: int,
                 attr_name: Optional[str] = None, **kwargs):
        self.walk_length = walk_length
        self.attr_name = 'random_walk_pe' if attr_name is None else attr_name

    def __call__(self, data: Data) -> Data:
        if data.edge_weight is None:
            data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32,
                                          device=data.edge_index.device)

        adj = self.compute_rw_matrix(data.num_nodes, data.edge_index, data.edge_weight)
        out = adj
        pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=data.num_nodes)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_self_loop_attr(*to_edge_index(out), num_nodes=data.num_nodes))
        pe = torch.stack(pe_list, dim=1)
        data[self.attr_name] = pe
        
        lap = self.normalized_laplacian(data)
        data['normalized_lap'] = lap
        return data

    @staticmethod
    def compute_rw_matrix(num_nodes, edge_index, edge_weight):
        # we need to assert that all edge weights are positive
        edge_weights = edge_weight
        assert torch.all(edge_weights > 0)

        # weighted in-degree of each node
        edge_indices = edge_index
        rec = edge_indices[0]
        node_deg = scatter(edge_weights, rec, dim_size=num_nodes, reduce='sum')
        pos_idx = torch.where(node_deg > 0)
        node_deg[pos_idx] = 1.0 / node_deg[pos_idx]

        # we used to_torch_csr_tensor before, but it gives Runtime Error if you don't have MKL installed
        # I couldn't install MKL bc it's for Intel processors, so I changed it to COO tensor
        # The sparse tensor is unpacked at every step of the loop, so it should give the same result
        adj = to_torch_coo_tensor(edge_indices, edge_weights, size=(num_nodes, num_nodes))

        # normalize the adjacency matrix
        adj = adj * node_deg.resize(num_nodes, 1)
        return adj
     
    def normalized_laplacian(self, data):
        nx_graph = nx.Graph()

        # Add the edges to the NetworkX graph
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            nx_graph.add_edge(i.item(), j.item())

        # get the normalized laplacian
        lap = nx.normalized_laplacian_matrix(nx_graph)
        
        return lap


class AddCellularRandomWalkPE(BaseTransform):
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
                 attr_name: Optional[str] = None,
                 traverse_type: str = "boundary",
                 use_node_features: bool = False, **kwargs):
        self.walk_length = walk_length
        self.attr_name = 'cc_random_walk_pe' if attr_name is None else attr_name
        self.traverse_type = traverse_type
        self.use_node_features = use_node_features

    def __call__(self, data: CellularComplexData) -> CellularComplexData:
        if self.traverse_type == "boundary":
            new_data = Data(num_nodes=data.num_cells,
                            edge_index=torch.cat((data.boundary_index, data.coboundary_index), dim=1),
                            edge_weight=torch.ones(data.boundary_index.shape[1]+data.coboundary_index.shape[1], dtype=torch.float32,))
        elif self.traverse_type == "upper_adj":
            new_data = Data(num_nodes=data.num_cells,
                            edge_index=data.upper_adj_index[:2, :])
        elif self.traverse_type == "lower_adj":
            new_data = Data(num_nodes=data.num_cells,
                            edge_index=data.lower_adj_index[:2, :])
        elif self.traverse_type == "upper_lower":
            edge_index = torch.cat([data.lower_adj_index[:2, :], data.upper_adj_index[:2, :]], dim=1)
            new_data = Data(num_nodes=data.num_cells,
                            edge_index=edge_index)
        elif self.traverse_type == "upper_lower_boundary":
            edge_index = torch.cat([data.lower_adj_index[:2, :],
                                    data.upper_adj_index[:2, :],
                                    data.boundary_index], dim=1)
            new_data = Data(num_nodes=data.num_cells,
                            edge_index=edge_index)
        else:
            raise Exception("traverse_type illegal")
        add_rwpe = AddRandomWalkPE(self.walk_length, attr_name='tmp_rwpe')
        pe = add_rwpe(new_data).tmp_rwpe

        # aggregation
        if self.traverse_type in ["lower_adj", "upper_adj", "upper_lower"]:
            # these methods do not change dimension of cells during random walk
            # that means nodes jump to nodes, edges to edges, cycles to cycles
            # so the information don't mix and we need to aggregate it to nodes only
            # we first propagate cycles to edges
            mask = (data.cell_dims[data.upper_adj_index[0, :]] == 1).squeeze()
            pe[data.upper_adj_index[0, mask], :] += pe[data.upper_adj_index[2, mask], :]
            # second propagate edges to nodes
            mask = (data.cell_dims[data.upper_adj_index[0, :]] == 0).squeeze()
            pe[data.upper_adj_index[0, mask], :] += pe[data.upper_adj_index[2, mask], :]

        if self.use_node_features:
            pe = pe[data.cell_dims.squeeze() == 0, :]

        # lap = self.normalized_laplacian(data)
        # data['normalized_lap'] = lap
        data[self.attr_name] = pe
        return data

    def normalized_laplacian(self, data):
        nx_graph = nx.Graph()

        # Add the edges to the NetworkX graph
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            nx_graph.add_edge(i.item(), j.item())

        # get the normalized laplacian
        lap = nx.normalized_laplacian_matrix(nx_graph)
        
        return lap


class AppendRWPE(BaseTransform):
    def __init__(self,
                 h_name: str = 'x',
                 pe_name: str = 'random_walk_pe'):
        self.h_name = h_name
        self.pe_name = pe_name

    def __call__(self, data):
        data[self.h_name] = torch.cat((data[self.h_name], data[self.pe_name]), dim=1)
        return data


class AppendCCRWPE(BaseTransform):
    def __init__(self,
                 cell_features_name: str = 'cell_features',
                 pe_name: str = 'cc_random_walk_pe',
                 use_node_features: bool = False):
        self.pe_name = pe_name
        self.cell_features_name = cell_features_name
        self.use_node_features = use_node_features

    def __call__(self, data):
        cf = data[self.cell_features_name]
        pe = data[self.pe_name]

        if self.use_node_features:
            data.x = torch.cat((data.x, pe[:data.num_nodes]), dim=1)
        else:
            data[self.cell_features_name] = torch.cat((cf, pe), dim=1)

        return data

