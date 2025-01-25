import argparse
import torch
import ast
import shlex

def arg_parser():
    parser = argparse.ArgumentParser(description="Config for training")

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size')
    parser.add_argument('--fold', type=str, help='Fold indices', default='[]')
    parser.add_argument('--early_stop', type=ast.literal_eval, default=False, help='Enable early stopping')
    parser.add_argument('--pre_training', type=ast.literal_eval, default=False, help='Enable early stopping')
    parser.add_argument('--dataset_name', type=str, default='', help='Name of the dataset')
    parser.add_argument('--test_dataset', type=str, default='', help='Name of the dataset')
    parser.add_argument('--seed', type=ast.literal_eval, default=False, help='Reproducibility')
    parser.add_argument('--folder_name', type=str, default='', help='Output directory')
    return parser


def read_args_from_file(file_path):
    with open(file_path, 'r') as file:
        args = shlex.split(file.read())
    return args

def parse_args(config_file, remaining_args=None):
    parser = arg_parser()
    file_args = read_args_from_file(config_file)
    all_args = file_args + (remaining_args if remaining_args is not None else [])
    args = parser.parse_args(all_args)

    # Parse fold argument from string to list
    args.fold = ast.literal_eval(args.fold)
    args.freezed = ast.literal_eval(args.freezed)

    return args