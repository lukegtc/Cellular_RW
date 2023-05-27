import argparse


def parse_train_args():
    parser = argparse.ArgumentParser()

    # ZINC params
    parser.add_argument('--zinc_folder', type=str, default='datasets/ZINC')
    parser.add_argument('--subset', type=bool, default=True)

    # PE params
    parser.add_argument('--use_pe', type=str, default=None, choices=['rw', 'ccrw'])  # PE type
    parser.add_argument('--learnable_pe', type=bool, default=False)  # whether to learn PE or not

    # random walk based PE params
    pe_params = parser.add_argument_group('pe_params')
    pe_params.add_argument('--walk_length', type=int, default=20)  # PE random walk length
    pe_params.add_argument('--traverse_type', type=str, default='upper_adj',
                           choices=['boundary', 'upper_adj', 'lower_adj', 'upper_lower', 'upper_lower_boundary'])

    # gnn name (we set params inside train script)
    parser.add_argument('--model', type=str, default='gin', choices=['gin'])

    # training params
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--trainer_root_dir', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)

    args = parser.parse_args()

    # extract pe params
    pe_params = {}
    for g in parser._action_groups:
        title = g.title.lower()
        if 'pe_params' in title:
            for a in g._group_actions:
                pe_params[a.dest] = getattr(args, a.dest)

    return args, pe_params
