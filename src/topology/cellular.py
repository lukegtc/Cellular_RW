from typing import Optional, List, Tuple, Dict
from itertools import combinations

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class CellularComplex:
    def __init__(self,
                 cells: List[Dict[int, Tuple]],
                 boundary_index: torch.Tensor,
                 coboundary_index: torch.Tensor):
        """
        Describes cellular complex.

        Args:
            cells: List of dictionaries, where each dictionary contains cells of dimension k.
            The dictionary is in format {cell_id: cell}. Cell is a tuple of cell ids of dimension k-1.
            We assume that cells are ordered by dimension, and indexing is contiguous, so
            indices are from 0 to N-1, where N is the total number of cells.

            boundary_index:

        This way we have clearly defined C_k spaces and boundary index, which describes the cellular complex.
        """
        # basic objects to define a cellular complex
        self.cells = cells
        self.boundary_index = boundary_index
        self.coboundary_index = coboundary_index

        # possible extra stuff
        self.upper_adj_index: Optional[torch.Tensor] = None
        self.lower_adj_index: Optional[torch.Tensor] = None
        self.cell_features: Optional[torch.Tensor] = None

    @property
    def num_cells(self):
        return sum(len(k_cells) for k_cells in self.cells)

    @property
    def cell_dims(self):
        cdims = []
        for k, cells in enumerate(self.cells):
            cdims.extend([k] * len(cells))
        return torch.tensor(cdims, dtype=torch.long)

    @classmethod
    def from_nx_graph(cls, graph: nx.Graph):
        cells = [{},{},{}]
        boundary_cols = []
        coboundary_cols = []

        # ------- NODES ---------
        cells[0] = {}
        for node_id in graph.nodes:
            cells[0][node_id] = ()

        # ------- EDGES ---------
        nodes2edge = {}
        cells[1] = {}
        for edge_id, edge in enumerate(graph.edges, start=len(graph.nodes)):
            cells[1][edge_id] = edge

            nodes2edge[edge[0], edge[1]] = edge_id  # we'll need that for recovering cycle edge ids
            for node_id in edge:
                boundary_cols.append([edge_id, node_id])
                coboundary_cols.append([node_id, edge_id])

        # ------- CYCLES ---------
        digraph = graph.to_directed()
        cycles = [cycle for cycle in nx.simple_cycles(digraph) if len(cycle) > 2]

        def to_edge_set(cycle):
            edges = []
            n_edges = len(cycle)
            for node_id, node in enumerate(cycle):
                next_node = cycle[(node_id + 1) % n_edges]
                assert graph.has_edge(node, next_node)
                try:
                    edge = nodes2edge[node, next_node]
                except KeyError:
                    edge = nodes2edge[next_node, node]
                edges.append(edge)
            return cycle

        cells[2] = {}
        for cycle_id, cycle in enumerate(cycles, start=len(graph.nodes) + len(graph.edges)):
            cycle = to_edge_set(cycle)
            cells[2][cycle_id] = cycle

            for edge_id in cycle:
                boundary_cols.append([cycle_id, edge_id])
                coboundary_cols.append([edge_id, cycle_id])

        boundary_index = torch.tensor(boundary_cols, dtype=torch.long).T
        coboundary_index = torch.tensor(coboundary_cols, dtype=torch.long).T

        cc = cls(cells=cells,
                 boundary_index=boundary_index,
                 coboundary_index=coboundary_index)
        cc.compute_upper_adj_index()
        cc.compute_lower_adj_index()
        cc.cell_features = torch.zeros((cc.num_cells, 1), dtype=torch.long)
        return cc

    def compute_upper_adj_index(self):
        vertex_upper_adj = [[], [], []]
        for edge_id, edge in self.cells[1].items():  # unique_id for edge id:(node1, node2)
            vertex_upper_adj[0].append(edge[0])
            vertex_upper_adj[1].append(edge[1])
            vertex_upper_adj[2].append(edge_id)

            vertex_upper_adj[0].append(edge[1])
            vertex_upper_adj[1].append(edge[0])
            vertex_upper_adj[2].append(edge_id)

        edge_upper_adj = [[], [], []]

        for cycle_id, cycle in self.cells[2].items():
            for edge_id_1, edge_id_2 in combinations(cycle, 2):
                edge_upper_adj[0].append(edge_id_1)
                edge_upper_adj[1].append(edge_id_2)
                edge_upper_adj[2].append(cycle_id)

                edge_upper_adj[0].append(edge_id_2)
                edge_upper_adj[1].append(edge_id_1)
                edge_upper_adj[2].append(cycle_id)

        self.upper_adj_index = torch.cat([torch.Tensor(vertex_upper_adj),
                                          torch.Tensor(edge_upper_adj)],
                                         dim=1).long()

    def compute_lower_adj_index(self):
        # Lower adj is from edge node to edge node using a normal node and from cycle node to cycle node using an edge node
        # edge_lower_adj = [[], [], []]
        # index1 = edge1_id, index2 = edge2_id, index3 = node_id
        edge_ids = self.boundary_index[0, :]
        node_ids = self.boundary_index[1, :]
        edge_lower_adj = []
        for i in range(len(node_ids)):
            for j in range(i+1, len(node_ids)):
                if node_ids[i] == node_ids[j]:
                    edge_lower_adj.append([edge_ids[i], edge_ids[j], node_ids[i]])
        edge_lower_adj = torch.tensor(edge_lower_adj).T.long()
        self.lower_adj_index = edge_lower_adj

class CellularComplexData(Data):
    @classmethod
    def from_data_cc_pair(cls, data: Data, cc: CellularComplex):
        data['num_cells'] = cc.num_cells
        data['cell_features'] = cc.cell_features
        data['cell_dims'] = cc.cell_dims
        data['cell_batch'] = torch.zeros(cc.num_cells, dtype=torch.long)
        data['boundary_index'] = cc.boundary_index
        data['coboundary_index'] = cc.coboundary_index
        data['upper_adj_index'] = cc.upper_adj_index
        data['lower_adj_index'] = cc.lower_adj_index
        mapping = data.items()._mapping
        return cls(**mapping)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'boundary_index':
            return self.num_cells
        if key == 'upper_adj_index':
            return self.num_cells
        if key == 'lower_adj_index':
            return self.num_cells
        return super().__inc__(key, value, *args, **kwargs)


class LiftGraphToCC(BaseTransform):
    def __call__(self, data: Data) -> Data:
        graph = nx.Graph()
        graph.add_nodes_from(range(data.num_nodes))
        graph.add_edges_from(data.edge_index.T.tolist())
        cc = CellularComplex.from_nx_graph(graph)
        new_data = CellularComplexData.from_data_cc_pair(data, cc)
        return new_data



