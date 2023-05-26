import torch
import torch.nn as nn
import torch.optim as op
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from torch_geometric.transforms import Compose

from topology.cellular import LiftGraphToCC
from topology.pe import AddRandomWalkPE, AddCellularRandomWalkPE, AppendCCRWPE, AppendRWPE
from models.mpgnn import MPGNN, MPGNNHead
from config import parse_train_args


class ZINCModel(nn.Module):
    """
    We combine here the argument preprocessing, GNN and the head.
    We should define a separate model for each dataset, because the attribute names need not be consistent between datasets.
    We unpack the attributes and make all necessary calls like .float() in the dedicated function extract_gnn_args.
    """
    def __init__(self, gnn_params, head_params, use_pe=False):
        super().__init__()
        self.gnn = MPGNN(**gnn_params)
        self.head = MPGNNHead(**head_params)
        self.use_pe = use_pe

    def extract_gnn_args(self, graph):
        h, edge_index, e, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        h = h.float()
        e = e.unsqueeze(1).float()

        if self.use_pe:
            p = graph.random_walk_pe
            h = torch.cat((h, p), dim=1)

        return h, e, edge_index, batch

    def forward(self, graph):
        h, e, edge_index, batch = self.extract_gnn_args(graph)
        out = self.gnn(h, e, edge_index)
        out = self.head(out, batch)
        return out


class LitZINCModel(pl.LightningModule):
    def __init__(self, gnn_params, head_params, training_params):
        super().__init__()
        self.save_hyperparameters()
        # self.gnn = MPGNN(**gnn_params)
        # self.head = MPGNNHead(**head_params)

        self.model = ZINCModel(gnn_params, head_params, training_params['use_pe'])
        self.criterion = nn.L1Loss(reduce='sum')
        self.training_params = training_params

    def training_step(self, batch, batch_idx):
        # h, edge_index, e, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        # h = h.float()
        # e = e.unsqueeze(1).float()
        # h, edge_index = batch.x, batch.edge_index
        # h = h.float()
        # out = self.gnn(h, e, edge_index)
        # out = self.head(out, b)

        label = batch.y
        out = self.model(batch)
        loss = self.criterion(out, label)
        self.log("train_loss", loss)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        # h, edge_index, e, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        # h = h.float()
        # e = e.unsqueeze(1).float()
        # # h, edge_index = batch.x, batch.edge_index
        # # h = h.float()
        # out = self.gnn(h, e, edge_index)
        # out = self.head(out, b)

        label = batch.y
        out = self.model(batch)
        loss = self.criterion(out, label)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # h, edge_index, e, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        # h = h.float()
        # e = e.unsqueeze(1).float()
        # # h, edge_index = batch.x, batch.edge_index
        # # h = h.float()
        # out = self.gnn(h, e, edge_index)
        # out = self.head(out, batch)

        label = batch.y
        out = self.model(batch)
        loss = self.criterion(out, label)
        self.log("test_loss", loss)
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

    # GIN does not use edge features, so we don't need to create any feature initialization transform.
    transform = None
    if args.use_pe is not None:
        if args.use_pe == 'rw':
            transform = Compose([
                AddRandomWalkPE(walk_length=args.walk_length),
                AppendRWPE()
            ])
        elif args.use_pe == 'ccrw':
            transform = Compose([
                LiftGraphToCC(),
                AddCellularRandomWalkPE(walk_length=args.walk_length),
                AppendCCRWPE(use_node_features=True)
            ])
        else:
            raise ValueError('Invalid PE type')

    data_train = ZINC(args.zinc_folder, subset=True, split='train', pre_transform=transform)  # QM9('datasets/QM9', pre_transform=transform)
    data_val = ZINC(args.zinc_folder, subset=True, split='val', pre_transform=transform)  # QM9('datasets/QM9', pre_transform=transform)
    data_test = ZINC(args.zinc_folder, subset=True, split='test', pre_transform=transform)

    train_loader = DataLoader(data_train, batch_size=32)
    val_loader = DataLoader(data_val, batch_size=32)
    test_loader = DataLoader(data_test, batch_size=32)

    gnn_in_features = args.feat_in
    if args.use_pe is not None:
        gnn_in_features += 2 * args.walk_length 

    gnn_params = {
        'feat_in': gnn_in_features,
        'edge_feat_in': 1,
        'num_hidden': 32,
        'num_layers': 16
    }

    head_params = {
        'num_hidden': 32,
    }

    training_params = {
        'lr': 1e-3,
        'lr_decay': 0.5,
        'patience': 25,
        'min_lr': 1e-6,
        'use_pe': args.use_pe,
    }

    model = LitZINCModel(gnn_params, head_params, training_params)

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator=args.accelerator,
                         devices=args.devices,
                         log_every_n_steps=10,
                         default_root_dir=args.trainer_root_dir)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
