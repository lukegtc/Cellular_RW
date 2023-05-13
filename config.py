import argparse


def parse_train_args():
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument('--walk_length', type=int, default=4)  # PE random walk length

    # training params
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--accelerator', type=str, default='cpu')
    parser.add_argument('--trainer_root_dir', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)

    args = parser.parse_args()
    return args
