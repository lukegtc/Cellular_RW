import os
import torch
import torch.nn as nn
import torch.optim as op
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.transforms import Compose

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.models.GPSConv import GPSConvWrapper, GPSConvLSPEWrapper
from src.models.gin import GIN, GINLSPE
from src.models.GatedGCN import GatedGCN, GatedGCNLSPE
from src.config import parse_train_args
from src.topology.cellular import LiftGraphToCC
from src.topology.pe import AddRandomWalkPE, AddCellularRandomWalkPE, AppendCCRWPE, AppendRWPE
import wandb

class LitGNNModel(pl.LightningModule):
    model_classes = {
        'gin': GIN,
        'gin_lspe': GINLSPE,
        'gated_gcn': GatedGCN,
        'gated_gcn_lspe': GatedGCNLSPE,
        'gps': GPSConvWrapper,
        'gps_lspe': GPSConvLSPEWrapper,
    }

    def __init__(self, model_name, model_params, training_params, learnable_pe=False):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.gnn = self.model_classes[model_name](**model_params)
        self.criterion = nn.L1Loss(reduce='mean')
        self.training_params = training_params
        self.learnable_pe = learnable_pe

        # we initialize the predict function here to avoid the overhead of checking the model name
        # in every forward pass
        if self.model_name.startswith('gin'):
            if self.learnable_pe:
                def predict(gnn, batch):
                    h, p, edge_index = batch.x, batch.random_walk_pe, batch.edge_index
                    h = h.float()
                    return gnn(h, p, edge_index, batch.batch)
            else:
                def predict(gnn, batch):
                    h, edge_index = batch.x, batch.edge_index
                    h = h.float()
                    return gnn(h, edge_index, batch.batch)
        elif self.model_name.startswith('gated_gcn'):
            if self.learnable_pe:
                def predict(gnn, batch):
                    h, e, p, edge_index = batch.x, batch.edge_attr, batch.random_walk_pe, batch.edge_index
                    e = e.float().reshape(-1, 1)
                    h = h.float()
                    return gnn(h, e, p, edge_index, batch.batch)
            else:
                def predict(gnn, batch):
                    h, edge_index = batch.x, batch.edge_index
                    h = h.float()
                    return gnn(h, edge_index, batch.batch)

        elif self.model_name.startswith('gps'):
            if self.learnable_pe:

                def predict(gnn, batch):
                    h, e, p, edge_index = batch.x, batch.edge_attr, batch.random_walk_pe, batch.edge_index
                    e = e.float().reshape(-1, 1)
                    h = h.float()
                    return gnn(h, e, p, edge_index, batch.batch)

            else:
                def predict(gnn, batch):
                    h, edge_index = batch.x, batch.edge_index
                    h = h.float()
                    return gnn(h, edge_index, batch.batch)


        else:
            raise ValueError(f'Unknown model name: {self.model_name}')

        self._predict = predict

    def predict(self, batch):
        return self._predict(self.gnn, batch)

    def training_step(self, batch, batch_idx):
        out = self.predict(batch)
        label = batch.y
        loss = self.criterion(out, label)
        self.log("train_loss", loss)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.predict(batch)
        label = batch.y
        loss = self.criterion(out, label)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        out = self.predict(batch)
        label = batch.y
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

    def on_test_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        test_loss = self.trainer.callback_metrics['test_loss']
        print(f'\nCurrent test loss {test_loss}')

    def test_dataloader(self):
        return super().test_dataloader()

    def configure_optimizers(self):
        optimizer = op.Adam(self.parameters(), lr=self.training_params['lr'])
        scheduler = op.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.training_params['lr_decay'],
                                                      patience=self.training_params['patience'],
                                                      min_lr=self.training_params['min_lr'],
                                                      verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


def load_zinc(zinc_folder, use_pe, learnable=False, **pe_params):
    # GIN does not use edge features, so we don't need to create any feature initialization transform.
    pre_transforms = []
    transforms = []
    if use_pe is not None:
        if use_pe == 'rw':
            pre_transforms.append(AddRandomWalkPE(**pe_params))
            if not learnable:
                transforms.append(AppendRWPE(pe_name=pe_params['attr_name']))
        elif use_pe == 'ccrw':
            pre_transforms.extend([
                LiftGraphToCC(),
                AddCellularRandomWalkPE(**pe_params),
            ])
            if not learnable:
                transforms.append(AppendCCRWPE(pe_name=pe_params['attr_name'], use_node_features=True))
        else:
            raise ValueError('Invalid PE type')
    pre_transform = Compose(pre_transforms)
    transform = Compose(transforms)

    zinc_folder = zinc_folder
    if use_pe is not None:
        zinc_folder = zinc_folder + '_' + use_pe + '_' + str(pe_params['walk_length'])
        if use_pe == 'ccrw':
            zinc_folder = zinc_folder + '_' + pe_params['traverse_type']

    data_train = ZINC(zinc_folder, subset=True, split='train',
                      pre_transform=pre_transform,
                      transform=transform)  # QM9('datasets/QM9', pre_transform=transform)
    data_val = ZINC(zinc_folder, subset=True, split='val',
                    pre_transform=pre_transform,
                    transform=transform)  # QM9('datasets/QM9', pre_transform=transform)
    data_test = ZINC(zinc_folder, subset=True, split='test',
                     pre_transform=pre_transform,
                     transform=transform)

    return data_train, data_val, data_test


if __name__ == '__main__':
    args, pe_params = parse_train_args()
    pe_params['attr_name'] = 'random_walk_pe'
    pe_params['use_node_features'] = True

    wandb.init(
        name=f"{args.model}_{args.use_pe}_learnable_pe={args.learnable_pe}",
        project="CRW-GNN",
        notes="This is a test run",
        tags=[f'{args.model}', f'{args.use_pe}', f'learnable_pe={args.learnable_pe}'],
        entity="crw-gnn",
        config=args,
    )
    wandb_logger = WandbLogger(log_model='all')






    data_train, data_val, data_test = load_zinc(args.zinc_folder, args.use_pe, args.learnable_pe, **pe_params)

    train_loader = DataLoader(data_train, batch_size=128)
    val_loader = DataLoader(data_val, batch_size=128)
    test_loader = DataLoader(data_test, batch_size=128)

    feat_in = 1

    # default GIN-LSPE 500k params
    num_hidden = 78
    num_layers = 16

    # GatedGCN-LSPE 500k params
    if args.model == 'gated_gcn':
        num_hidden = 60
        num_layers = 16

    if args.model == 'gps':
        num_hidden=64
        num_layers=10


    model_params = {
        'feat_in': feat_in,
        'edge_feat_in': 1,
        'num_hidden': num_hidden,
        'num_layers': num_layers
    }

    training_params = {
        'lr': 1e-3,
        'lr_decay': 0.5,
        'patience': 20,
        'min_lr': 1e-5
    }

    # setup dependent on using learnable PE
    model_name = args.model
    if args.use_pe is not None:
        pe_features = args.walk_length
        if args.learnable_pe:
            model_name = model_name + '_lspe'
            model_params['pos_in'] = pe_features
            training_params['pe_name'] = pe_params['attr_name']
        else:
            model_params['feat_in'] += pe_features

    model = LitGNNModel(model_name, model_params, training_params, learnable_pe=args.learnable_pe)
    wandb.watch(model, log='all')
    if(args.use_pe is not None):
        logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir, name=model_name+args.use_pe+args.traverse_type)
        csv_logger = pl.loggers.CSVLogger(save_dir=args.log_dir, name=model_name+args.use_pe+args.traverse_type)
    else:
        logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir, name=model_name+args.traverse_type)
        csv_logger = pl.loggers.CSVLogger(save_dir=args.log_dir, name=model_name+args.traverse_type)
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator=args.accelerator,
                         devices=args.devices,
                         log_every_n_steps=10,
                         default_root_dir=args.trainer_root_dir,
                         logger=[logger, csv_logger, wandb_logger])

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    trainer.test(model, ckpt_path="best", dataloaders=test_loader)
    wandb.finish()
