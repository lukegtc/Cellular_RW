import torch
import torch.nn as nn
import torch.optim as op
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
import pytorch_lightning as pl

from transform import AddRandomWalkPE
from src.model_cwn import CWN, CWNHead
from src.config import parse_train_args

from typing import List


class ZINCModel(nn.Module):
    """
    We combine here the argument preprocessing, CWN and the head.
    We should define a separate model for each dataset, because the attribute names need not be consistent between datasets.
    We unpack the attributes and make all necessary calls like .float() in the dedicated function extract_gnn_args.
    """
    def __init__(self, cwn_params, head_params, pe_params):
        super().__init__()
        self.gnn = CWN(**cwn_params)
        self.head = CWNHead(**head_params)
        self.pe_params = pe_params
        self.device = cwn_params['device']

    def extract_gnn_args(self, graph):
        cell_features: List[torch.Tensor] = graph.cell_features
        boundary_index: List[torch.Tensor] = graph.boundary_index
        upper_adj_index: List[torch.Tensor] = graph.upper_adj_index
        cell_batches = graph.batch

        # should that be here on in transform.py?
        for i in range(len(cell_features)):
            cell_features[i] = cell_features[i].float()
        for i in range(len(boundary_index)):
            boundary_index[i] = boundary_index[i].long()
        for i in range(len(upper_adj_index)):
            upper_adj_index[i] = upper_adj_index[i].long()

        if self.pe_params['use_pe']:
            if self.pe_params['use_cells'] == True:
                initial_pos_enc: List[torch.Tensor] = graph.random_walk_pe_with_cells
            if self.pe_params['use_cells'] == False:
                initial_pos_enc: List[torch.Tensor] = graph.random_walk_pe
            cell_features = cell_features.reshape(-1, 1)
            cell_features = torch.cat((cell_features, initial_pos_enc), dim=1)
            # cell_features = [torch.cat((h, p), dim=1) for h, p in zip(cell_features, initial_pos_enc)]

        return cell_features, boundary_index, upper_adj_index, cell_batches

    def forward(self, graph):
        cell_features, boundary_index, upper_adj_index, cell_batches = self.extract_gnn_args(graph)
        cell_features = self.gnn(cell_features, boundary_index, upper_adj_index)
        out = self.head(cell_features, cell_batches)
        return out


class LitZINCModel(pl.LightningModule):
    def __init__(self, cwn_params, head_params, training_params):
        super().__init__()
        self.save_hyperparameters()
        self.model = ZINCModel(cwn_params, head_params, training_params['pe_params'])
        self.criterion = nn.L1Loss(reduce='sum')
        self.training_params = training_params

    def training_step(self, batch, batch_idx):
        label = batch.y
        out = self.model(batch)
        loss = self.criterion(out, label)
        self.log("train_loss", loss)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        label = batch.y
        out = self.model(batch)
        loss = self.criterion(out, label)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        # get last train epoch loss
        train_loss = self.trainer.callback_metrics['train_loss']
        print(f'\nCurrent train loss {train_loss}')
        # get last validation epoch loss
        val_loss = self.trainer.callback_metrics['val_loss']
        print(f'Current val loss {val_loss}')

    def configure_optimizers(self):
        optimizer = op.Adam(model.parameters(), lr=self.training_params['lr'])
        scheduler = op.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.training_params['lr_decay'],
                                                      patience=self.training_params['patience'],
                                                      min_lr=self.training_params['min_lr'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


if __name__ == '__main__':
    args = parse_train_args()
    if args.use_cells:
        attr_name = 'random_walk_pe_with_cells'
    else:
        attr_name = 'random_walk_pe'
    transform = AddRandomWalkPE(walk_length=args.walk_length, attr_name=attr_name)
    data_train = ZINC('datasets/ZINC_basic',subset=args.subset, split='train', pre_transform=transform)  # QM9('datasets/QM9', pre_transform=transform)
    data_val = ZINC('datasets/ZINC_basic', subset=args.subset, split='val', pre_transform=transform)  # QM9('datasets/QM9', pre_transform=transform)

    train_loader = DataLoader(data_train[:10000], batch_size=32)
    val_loader = DataLoader(data_val[:1000], batch_size=32)

    gnn_params = {
        # 'feat_in': args.feat_in,
        'initial_cell_dims' : [1,1,1],
        # 'edge_feat_in': 1,
        'num_hidden': 32,
        'num_layers': 16,
        'device': "cuda:0"
    }

    head_params = {
        'num_hidden': 32,
    }
    pe_params = {
        # 'use_pe': args.use_pe,
        'use_pe': True,

        'use_cells': args.use_cells,



    }

    training_params = {
        'lr': 1e-3,
        'lr_decay': 0.5,
        'patience': 25,
        'min_lr': 1e-6,
        'pe_params': pe_params,
    }

    model = LitZINCModel(gnn_params, head_params, training_params)

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator=args.accelerator,
                         devices=args.devices,
                         log_every_n_steps=10,
                         default_root_dir=args.trainer_root_dir)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
