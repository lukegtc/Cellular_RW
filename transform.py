from typing import Any, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
)
import networkx as nx


def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


class AddRandomWalkPE(BaseTransform):
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
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_pe',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        row, col = data.edge_index
        N = data.num_nodes
        
        cycle_indices, num_added_nodes = self.create_graph_index(data)
        combined_indices = torch.hstack([data.edge_index, cycle_indices]).type(data.edge_index.dtype)


        # value = data.edge_weight
        # if value is None:
            # value = torch.ones(data.num_edges, device=row.device)
        num_combined_nodes = data.num_nodes + num_added_nodes
        value = torch.ones(combined_indices.size()[1])
        # value = scatter(value, combined_indices[0], dim_size=N, reduce='sum').clamp(min=1)[row]
        value = scatter(value, combined_indices[0], dim_size=num_combined_nodes, reduce='sum').clamp(min=1)[combined_indices[0]]
        value = 1.0 / value

        # we used to_torch_csr_tensor before, but it gives Runtime Error if you don't have MKL installed
        # I couldn't install MKL bc it's for Intel processors, so I changed it to COO tensor
        # The sparse tensor is unpacked at every step of the loop, so it should give the same result
        adj = to_torch_coo_tensor(combined_indices, value, size=(num_combined_nodes,num_combined_nodes))

        out = adj
        pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=num_combined_nodes)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_self_loop_attr(*to_edge_index(out), num_nodes=num_combined_nodes))
        pe = torch.stack(pe_list, dim=-1)
        pe_nodes = pe[:data.num_nodes, :]
        data = add_node_attr(data, pe_nodes, attr_name=self.attr_name)
        return data

    def create_graph_index(self, data):
        nx_graph = nx.Graph()

        # Add the edges to the NetworkX graph
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            nx_graph.add_edge(i.item(), j.item())

        cycles = self.get_simple_cycles(nx_graph)
        largest_cycle = max([len(cycle) for cycle in cycles])
        cycles = [cycle for cycle in cycles if len(cycle) > 2]
        idx = self.get_cycle_index(nx_graph)

        first_index, second_index = [], []
        num_added_nodes = 0
        for i in idx:
            for j in idx[i]:
                first_index.append(i)
                second_index.append(j)
                first_index.append(j)
                second_index.append(i)
            num_added_nodes += 1
        graph_node_index = [first_index, second_index]
        return(torch.Tensor(graph_node_index),num_added_nodes)

    def get_simple_cycles(self, graph):
        digraph = graph.to_directed()
        cycles = [cycle for cycle in nx.simple_cycles(digraph)]
        return cycles

    def get_cycle_index(self, graph):
        digraph = nx.DiGraph(graph)
        cycles = list(nx.simple_cycles(digraph))
        cycle_index = {}
        node_set = set(graph.nodes())
        max_node = max(graph.nodes())

        for i, cycle in enumerate([c for c in cycles if(len(c)>2)]):
            cycle_nodes = set(cycle)
            assert cycle_nodes <= node_set, "Cycle nodes not in graph"
            cycle_index[max_node+i] = [node for node in cycle]
        return cycle_index