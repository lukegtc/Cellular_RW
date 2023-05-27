import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.optim as op
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
import pytorch_lightning as pl

from src.models.GatedGCN import GatedGCN_LSPE
from src.models.MPGNN_LSPE import MPGNN_LSPE, LapEigLoss
from src.topology.cellular import LiftGraphToCC
from src.topology.pe import AddRandomWalkPE, AddCellularRandomWalkPE, AppendCCRWPE, AppendRWPE
from src.models.gin import GIN
from src.config import parse_train_args


class LitMPGNN_LSPEModel(pl.LightningModule):
    def __init__(self, gnn_params, training_params):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = MPGNN_LSPE(**gnn_params)
        self.criterion = nn.L1Loss(reduce='mean')
        self.training_params = training_params
        self.pos_enc_loss = LapEigLoss(frobenius_norm_coeff=self.training_params['lspe_lambda'],
                                       pos_enc_dim=gnn_params['pos_in'])

    def training_step(self, batch, batch_idx):

        h, e,p, edge_index = batch.x,batch.edge_attr, batch.cc_random_walk_pe,batch.edge_index
        h = h.float()
        e = e.float()
        p = p.float()
        e = torch.unsqueeze(e,1)
        out,p_agg = self.gnn(h, e, p, edge_index, batch.batch)

        label = batch.y
        normalized_laplacians = batch.normalized_lap
        lap = sp.block_diag(normalized_laplacians)
        lap = torch.from_numpy(lap.todense()).float()
        loss = self.criterion(out, label) + self.pos_enc_loss(p, lap, batch.batch)
        self.log("train_loss", loss)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):

        h, e,p, edge_index = batch.x,batch.edge_attr, batch.cc_random_walk_pe,batch.edge_index
        h = h.float()
        e = e.float()
        p = p.float()
        e = torch.unsqueeze(e,1)
        out,p_agg = self.gnn(h, e, p, edge_index, batch.batch)

        label = batch.y
        normalized_laplacians = batch.normalized_lap
        lap = sp.block_diag(normalized_laplacians)
        lap = torch.from_numpy(lap.todense()).float()
        loss = self.criterion(out, label) + self.pos_enc_loss(p, lap, batch.batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):

        h, e,p, edge_index = batch.x,batch.edge_attr, batch.cc_random_walk_pe,batch.edge_index
        h = h.float()
        e = e.float()
        p = p.float()
        e = torch.unsqueeze(e,1)
        out,p_agg = self.gnn(h, e, p, edge_index, batch.batch)

        label = batch.y
        normalized_laplacians = batch.normalized_lap
        lap = sp.block_diag(normalized_laplacians)
        lap = torch.from_numpy(lap.todense()).float()
        loss = self.criterion(out, label) + self.pos_enc_loss(p, lap, batch.batch)
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

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

    def configure_optimizers(self):
        optimizer = op.Adam(self.parameters(), lr=self.training_params['lr'])
        scheduler = op.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.training_params['lr_decay'],
                                                      patience=self.training_params['patience'],
                                                      min_lr=self.training_params['min_lr'],
                                                      verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


if __name__ == '__main__':
    args = parse_train_args()

    # GIN does not use edge features, so we don't need to create any feature initialization transform.
    transform = None
    transform_str = ''
    if args.use_pe is not None:
        if args.use_pe == 'rw':
            transform = Compose([
                AddRandomWalkPE(walk_length=args.walk_length),
                AppendRWPE()
            ])
            transform_str = 'rw'
        elif args.use_pe == 'ccrw':
            transform = Compose([
                LiftGraphToCC(),
                AddCellularRandomWalkPE(walk_length=args.walk_length),
                AppendCCRWPE(use_node_features=True)
            ])
            transform_str = 'ccrw'
        else:
            raise ValueError('Invalid PE type')

    data_train = ZINC(args.zinc_folder + transform_str, subset=True, split='train',
                      pre_transform=transform)  # QM9('datasets/QM9', pre_transform=transform)
    data_val = ZINC(args.zinc_folder + transform_str, subset=True, split='val',
                    pre_transform=transform)  # QM9('datasets/QM9', pre_transform=transform)
    data_test = ZINC(args.zinc_folder + transform_str, subset=True, split='test', pre_transform=transform)

    train_loader = DataLoader(data_train, batch_size=128)
    val_loader = DataLoader(data_val, batch_size=128)
    test_loader = DataLoader(data_test, batch_size=128)

    num_hidden = 128
    num_gnn_layers = 4
    gnn_in_features = args.feat_in
    if args.use_pe is not None:
        gnn_in_features += args.walk_length

    gnn_params = {
        'feat_in': gnn_in_features,
        'num_hidden': num_hidden,
        'pos_in': args.walk_length,
        'edge_feat_in': 1,
        'num_layers': num_gnn_layers
    }

    # head_params = {
    #     'hidden_dim': num_hidden,
    #     'num_hidden_states': num_gnn_layers + 1,
    # }

    training_params = {
        'lr': 1e-3,
        'lr_decay': 0.5,
        'patience': 20,
        'min_lr': 1e-5,
        'lspe_lambda': 1e-1,
        'lspe_alpha': 1
    }

    model = LitMPGNN_LSPEModel(gnn_params, training_params)

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator=args.accelerator,
                         devices=args.devices,
                         log_every_n_steps=10,
                         default_root_dir=args.trainer_root_dir)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)

    trainer.test(model, ckpt_path="best", dataloaders=test_loader)
    # trainer.test(model)