import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import linalg
from torch_geometric.utils import unbatch
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool


class GatedGCN_LSPE(nn.Module):
    """ MP-GNN model with learnable structural and positional embeddings. """
    def __init__(self, feat_in, pos_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.h_embed = nn.Linear(feat_in, num_hidden)
        self.e_embed = nn.Linear(edge_feat_in, num_hidden)
        self.p_embed = nn.Linear(pos_in, num_hidden)
        self.layers = nn.ModuleList([GatedGCN_LSPELayer(num_hidden) for _ in range(num_layers)])
        self.predict = nn.Linear(2*num_hidden, 1)
        self.readout = nn.Sequential(nn.Linear(2*num_hidden, num_hidden), nn.ReLU(),nn.Linear(num_hidden , num_hidden//2),nn.ReLU(), nn.Linear(num_hidden // 2, 1))

    def forward(self, h, e, p, edge_index,batch):

        h = self.h_embed(h)
        e = self.e_embed(e)
        p = self.p_embed(p)

        for layer in self.layers:
            h, e, p = layer(h, e, p, edge_index)

        h_agg = global_add_pool(h, batch)
        p_agg = global_add_pool(p, batch)
        hep = torch.cat((h_agg, p_agg), dim=1)
        out = self.readout(hep).squeeze()
        return out


class GatedGCN_LSPELayer(nn.Module):
    """ MP-GNN layer handling structural and positional embeddings. """
    def __init__(self, num_hidden):
        super().__init__()

        self.bn = nn.BatchNorm1d(num_hidden)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_hidden, num_hidden)
        self.hp_send_layer = nn.Linear(2*num_hidden, num_hidden)
        self.hp_rec_layer = nn.Linear(2*num_hidden, num_hidden)
        self.p_layer_1 = nn.Linear(num_hidden, num_hidden)
        self.p_layer_2 = nn.Linear(num_hidden, num_hidden)

    def forward(self, h, e, p, edge_index):

        send, rec = edge_index
        eta = torch.sigmoid(self.linear(h[send]) + self.linear(h[rec])+self.linear(e))
        eta_new = eta/(eta.sum(dim=1, keepdim=True))
        hp_send = self.hp_send_layer(torch.cat((h[send], p[send]), dim=1))
        hp_rec = self.hp_rec_layer(torch.cat((h[rec], p[rec]), dim=1))

        h[send] = h[send]+self.relu(self.bn(hp_send + scatter_add(hp_rec*eta_new,rec, dim=0, dim_size=hp_rec.shape[0])))
        e = e +self.relu(self.bn(eta))
        linear_p = self.p_layer_2(p[rec])
        p[send] = p[send]+F.tanh(self.p_layer_1(p[send])+scatter_add(linear_p*eta_new,rec, dim=0, dim_size=linear_p.shape[0]))


        return h, e, p





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
