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
        
        cycle_indices, num_added_nodes, lap = self.compute_graph_stats(data)
        add_node_attr(data, lap, 'normalized_lap')

        combined_indices = torch.hstack([data.edge_index, cycle_indices]).type(data.edge_index.dtype)


        value = data.edge_weight
        if value is None:
            value = torch.ones(combined_indices.size()[1])
        else:
            # Get the value of cycles by averaging the edge weights of the edges in the cycle from cycle indices
            max_node = max(data.nodes())
            value = torch.cat([value, torch.mean(value[cycle_indices[0] == max_node + torch.arange(num_added_nodes)])])

        num_combined_nodes = data.num_nodes + num_added_nodes
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

        # add cell features
        cell_features = self.get_cell_features(data)
        data = add_node_attr(data, cell_features, attr_name='cell_features')
        
        # add boundary index
        boundary_index = self.boundary_index(data)
        data = add_node_attr(data, cell_features, attr_name='boundary_index')

        # add upper adjacency idx
        upper_adj_index = self.upper_adjacency(data)
        data = add_node_attr(data, upper_adj_index, attr_name='upper_adj_index')
        return data

    def compute_graph_stats(self, data):
        nx_graph = nx.Graph()

        # Add the edges to the NetworkX graph
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            nx_graph.add_edge(i.item(), j.item())

        # get the normalized laplacian
        lap = nx.normalized_laplacian_matrix(nx_graph)

        cycles = self.get_simple_cycles(nx_graph)
        # largest_cycle = max([len(cycle) for cycle in cycles])
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
        return torch.Tensor(graph_node_index), num_added_nodes, lap
    
    def make_graph(self, data):
        nx_graph = nx.Graph()

        # Add the edges to the NetworkX graph
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            nx_graph.add_edge(i.item(), j.item())
        return nx_graph
    
    def get_simple_cycles(self, graph):
        digraph = graph.to_directed()
        cycles = [cycle for cycle in nx.simple_cycles(digraph)]
        return cycles

    def get_cycle_index(self, graph):
        # unique cycle_index
        digraph = nx.DiGraph(graph)
        cycles = list(nx.simple_cycles(digraph))
        cycle_index = {}
        node_set = set(graph.nodes())
        edge_index = self.get_edge_index(graph)
        max_node = max(graph.nodes()) + len(edge_index.keys())

        for i, cycle in enumerate([c for c in cycles if(len(c)>2)]):
            cycle_nodes = set(cycle)
            assert cycle_nodes <= node_set, "Cycle nodes not in graph"
            cycle_index[max_node+i] = [node for node in cycle]
        return cycle_index
    
    def get_edge_index(self, graph):
        # unique edge_index
        edge_index = {}
        max_node = max(graph.nodes())
        for i, edge in enumerate(graph.edges):
            i += 1
            edge_index[max_node+i] = [node for node in edge]
        return edge_index
    
    def boundary_index(self, data):
        graph = self.make_graph(data)
        cycle_idx = self.get_cycle_index(graph)
        edge_idx = self.get_edge_index(graph)
        first_index, second_index = [], []

        for edge_id, nodes in edge_idx.items():
            for node in nodes:
                first_index.append(edge_id)
                second_index.append(node)

        for cycle_id, edge_list in cycle_idx.items(): 
            for edge in edge_list:
                edge_id = next((key for key, value in edge_idx.items() if value == list(edge)), None)
                if edge_id:
                    first_index.append(cycle_id)
                    second_index.append(edge_id)
        cycle_boundary = [first_index, second_index]
        return torch.Tensor(cycle_boundary)
    
    def upper_adjacency(self, data):
        nx_graph = nx.Graph()

        # Add the edges to the NetworkX graph
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            nx_graph.add_edge(i.item(), j.item())

        first = []
        second = []
        third = []
        edge_index = self.get_edge_index(nx_graph)
        cycle_edge_idx = self.get_cycle_edges(nx_graph)
        for edge_id, edge in edge_index.items():
                first.append(edge_id)
                second.append(edge[0])
                third.append(edge[1])

        for cell_id, edges in cycle_edge_idx.items():
            for i in range(len(edges) - 1):
                for j in range(i + 1, len(edges)):
                    edge_id_1 = next((key for key, value in edge_index.items() if value == list(edges[i])), None)
                    edge_id_2 = next((key for key, value in edge_index.items() if value == list(edges[j])), None)
                    if edge_id_1 and edge_id_2:
                        first.append(edge_id_1)
                        second.append(edge_id_2)
                        third.append(cell_id)

        up_adj = [first, second, third]
        return torch.Tensor(up_adj)
    
    def get_cycle_edges(self, graph):
        # dictionary cycle and edges forming the cycle
        cycle_bound_idx = {}
        cycles = self.get_simple_cycles(graph)
        for j, cycle in enumerate(cycles):
            for i in range(len(cycle)):
                current_node = cycle[i]
                next_node = cycle[(i+1) % len(cycle)] 

                if graph.has_edge(current_node, next_node):
                    if j in cycle_bound_idx.keys():
                        cycle_bound_idx[j].append([current_node, next_node])
                    else:
                        cycle_bound_idx[j] = [[current_node, next_node]]
    
    def get_cell_features(self, data):
        nx_graph = nx.Graph()

        # Add the edges to the NetworkX graph
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            nx_graph.add_edge(i.item(), j.item())

        idx = self.get_cycle_index(nx_graph)
        num_cycles = len(idx)
        cell_features = torch.zeros(num_cycles)
        for i in range(cell_features.shape[0]):
            for node in list(idx.values())[i]:
                cell_features[i] += data.x[node].item()
        return cell_features
