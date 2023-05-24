import argparse


def parse_train_args():
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument('--zinc_path', type=str, default='datasets/ZINC')
    parser.add_argument('--subset', type=bool, default=True)

    # model params are mostly omitted here
    # they are set directly in training script corresponding to particular model
    # this one corresponds to basic MPGNN
    parser.add_argument('--feat_in', type=int, default=1)

    # training params
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--trainer_root_dir', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)

    # pe params
    parser.add_argument('--walk_length', type=int, default=20)  # PE random walk length
    parser.add_argument('--use_pe', action='store_true')
    parser.add_argument('--pe_max_cell_dim')

    args = parser.parse_args()
    return args
