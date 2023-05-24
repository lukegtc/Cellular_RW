import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add
from torch_geometric.utils import unbatch
import scipy.sparse as sp
import scipy.sparse.linalg as linalg


class MPGNN(nn.Module):
    """ Standard MP-GNN model. """
    def __init__(self, feat_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.embed = nn.Linear(feat_in, num_hidden)
        self.edge_embed = nn.Linear(edge_feat_in, num_hidden)
        self.layers = nn.ModuleList([MPGNNLayer(num_hidden) for _ in range(num_layers)])

    def forward(self, h, e, edge_index):
        h = self.embed(h)
        e = self.edge_embed(e)

        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        return h


class MPGNNLayer(nn.Module):
    """ Standard MP-GNN layer. """
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
    """ Handles readout and final prediction from standard MP-GNN model. """
    def __init__(self, num_hidden):
        super().__init__()
        self.predict = nn.Linear(num_hidden, 1)

    def forward(self, h, h_batch):
        graph_reprs = global_add_pool(h, h_batch)
        final_prediction = self.predict(graph_reprs)
        return final_prediction.squeeze(1)


class LSPE_MPGNN(nn.Module):
    """ MP-GNN model with learnable structural and positional embeddings. """
    def __init__(self, feat_in, pos_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.e_embed = nn.Linear(edge_feat_in, num_hidden)
        self.p_embed = nn.Linear(pos_in, num_hidden)
        self.layers = nn.ModuleList([LSPE_MPGNNLayer(num_hidden) for _ in range(num_layers)])
        self.predict = nn.Linear(2*num_hidden, 1)

    def forward(self, h, e, p, edge_index):
        h = self.h_embed(h)
        e = self.e_embed(e)
        p = self.p_embed(p)

        for layer in self.layers:
            h, e, p = layer(h, e, p, edge_index)

        return h, p


class LSPE_MPGNNLayer(nn.Module):
    """ MP-GNN layer handling structural and positional embeddings. """
    def __init__(self, num_hidden):
        super().__init__()
        self.h_message_mlp = nn.Linear(5 * num_hidden, num_hidden)
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
    """ Handles readout and final prediction for graph regression task from LSPE-MP-GNN model. """
    def __init__(self, num_hidden):
        super().__init__()
        self.predict = nn.Linear(2*num_hidden, 1)

    def forward(self, h, p, h_batch):
        h = global_add_pool(h, h_batch)
        p = global_add_pool(p, h_batch)  # we can use batch indices for nodes because we have positional encoding for each node
        graph_reprs = torch.cat((h, p), dim=1)
        final_prediction = self.predict(graph_reprs, dim=1)
        return final_prediction.squeeze(1)


class LapEigLoss(nn.Module):
    """ One part of loss function for LSPE-MP-GNN model. """
    def __init__(self, frobenius_norm_coeff, pos_enc_dim):
        super().__init__()
        self.coeff = frobenius_norm_coeff
        self.pos_enc_dim = pos_enc_dim

    def forward(self, p, normalized_laplacian, p_batch):
        # p is dense
        # we assume that laplacian is also dense
        loss1 = torch.trace(p.T @ normalized_laplacian @ p)

        p_unbatched = unbatch(p.detach(), p_batch)
        p_block = sp.block_diag(p_unbatched)

        # # Conversion to torch tensor
        # indices = torch.LongTensor(torch.vstack((p_block.row, p_block.col)))
        # values = torch.FloatTensor(p_block.data)
        # shape = p_block.shape
        #
        # p_block = torch.sparse_coo_tensor(indices, values, torch.Size(shape))

        PTP_In = p_block.T * p_block - sp.eye(p_block.shape[1])
        loss2 = torch.tensor(linalg.norm(PTP_In, 'fro') ** 2)

        batch_size = len(p_unbatched)
        n = normalized_laplacian.shape[0]
        loss = (loss1 + self.coeff * loss2) / (self.pos_enc_dim * batch_size * n)
        return loss


class GIN(nn.Module):
    """ GIN model. """
    def __init__(self, feat_in, num_hidden, num_layers):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.layers = nn.ModuleList([GINLayer(num_hidden) for _ in range(num_layers)])
        self.predict = nn.Linear(num_hidden, 1)

    def forward(self, h, edge_index):
        h = self.h_embed(h)

        for layer in self.layers:
            h = layer(h, edge_index)

        return h


class GINLayer(nn.Module):
    """ GIN-0 layer. """
    def __init__(self, num_hidden):
        super().__init__()
        self.h_update = nn.Linear(num_hidden, num_hidden)

    def forward(self, h, edge_index):
        send, rec = edge_index

        h_messages = h
        h_messages_agg = scatter_add(h_messages, rec, dim=0, dim_size=h.shape[0])
        h = self.h_update(h + h_messages_agg)

        return h
