import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add
from torch_geometric.utils import unbatch
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

class MPGNN_LSPE(nn.Module):
    """ Standard MP-GNN model. """
    def __init__(self, feat_in, pos_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.e_embed = nn.Linear(edge_feat_in, num_hidden)
        self.p_embed = nn.Linear(pos_in, num_hidden)
        self.layers = nn.ModuleList([MPGNN_LSPELayer(num_hidden) for _ in range(num_layers)])
        self.readout = nn.Sequential(nn.Linear(2*num_hidden, num_hidden), nn.ReLU(),nn.Linear(num_hidden , num_hidden//2),nn.ReLU(), nn.Linear(num_hidden // 2, 1))

    def forward(self, h, e, p, edge_index,batch):
        h = self.h_embed(h)
        e = self.e_embed(e)
        p = self.p_embed(p)

        for layer in self.layers:
            h, e,p = layer(h, e, p,edge_index)

        h_agg = global_add_pool(h, batch)
        p_agg = global_add_pool(p, batch)
        hep = torch.cat((h_agg, p_agg), dim=1)
        out = self.readout(hep).squeeze()
        return out,p

class MPGNN_LSPELayer(nn.Module):
    """ Standard MP-GNN layer. """
    def __init__(self, num_hidden):
        super().__init__()

        self.h_update = nn.Linear(5*num_hidden, num_hidden)
        self.e_update = nn.Linear(3*num_hidden, num_hidden)
        self.p_update = nn.Linear(3*num_hidden, num_hidden)
        self.h_message_agg_update = nn.Linear(2*num_hidden, num_hidden)
        self.p_message_agg_update = nn.Linear(2*num_hidden, num_hidden)

    def forward(self, h, e, p, edge_index):
        send, rec = edge_index
        hp_send = torch.cat((h[send], p[send]), dim=1)
        hp_rec = torch.cat((h[rec], p[rec]), dim=1)
        h_messages = self.h_update(torch.cat((hp_send, hp_rec, e), dim=1))
        h_messages_agg = scatter_add(h_messages, rec, dim=0, dim_size=h.shape[0])
        h = self.h_message_agg_update(torch.cat((h, h_messages_agg), dim=1))

        e = self.e_update(torch.cat((h[send], h[rec], e), dim=1))

        p_messages = self.p_update(torch.cat((p[send], p[rec], e), dim=1))
        p_messages_agg = scatter_add(p_messages, rec, dim=0, dim_size=p.shape[0])
        p = self.p_message_agg_update(torch.cat((p, p_messages_agg), dim=1))

        return h, e,p

class LapEigLoss(nn.Module):
    """ One part of loss function for LSPE-MP-GNN model. """
    def __init__(self, frobenius_norm_coeff, pos_enc_dim):
        super().__init__()
        self.coeff = frobenius_norm_coeff
        self.pos_enc_dim = pos_enc_dim

    def forward(self, p, normalized_laplacian, p_batch):
        # p is dense
        # we assume that laplacian is also dense

        loss1 = torch.trace(p.T @ normalized_laplacian.to('cuda') @ p)

        p_unbatched = unbatch(p.detach().to('cpu'), p_batch.to('cpu'))
        p_block = sp.block_diag(p_unbatched)

        PTP_In = p_block.T * p_block - sp.eye(p_block.shape[1])
        loss2 = torch.tensor(linalg.norm(PTP_In, 'fro') ** 2)

        batch_size = len(p_unbatched)
        n = normalized_laplacian.shape[0]
        loss = (loss1 + self.coeff * loss2) / (self.pos_enc_dim * batch_size * n)
        return loss