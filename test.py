import argparse
import os
import torch
import warnings

from feature_processor import generate_branch1_pred_masks

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Branch1 prediction masks for the test split')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root, e.g. ./data/vessel/')
    parser.add_argument('--pretrained_weight', type=str, required=True, help='Path to pretrained Branch1 weights')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory for thresholded prediction masks')
    return parser.parse_args()


def test_with_pretrained():
    args = parse_args()

    data_path = args.data_path
    pretrained_weight = args.pretrained_weight
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    output_dir = args.output_dir

    print(f'#----------Loading pretrained weights from {pretrained_weight}----------#')
    print(f'#----------Using device: {device}----------#')
    print(f'#----------Saving thresholded prediction masks to {output_dir}----------#')
    print(f'#----------Saving probability maps to {output_dir}_prob----------#')

    os.makedirs(output_dir, exist_ok=True)
    generate_branch1_pred_masks(
        data_path=data_path,
        branch1_model_path=pretrained_weight,
        device=device,
        splits=('test',),
        output_dirs={'test': output_dir},
    )

    print('Prediction masks generated successfully!')


if __name__ == '__main__':
    test_with_pretrained()
