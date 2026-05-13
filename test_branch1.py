import argparse
import os
import torch
import warnings

from feature_processor import generate_branch1_pred_masks

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Branch1 prediction masks for the test split')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root, e.g. ./data/vessel/')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Branch1 checkpoint')
    parser.add_argument('--output_dir', type=str, default='./data/vessel/test/pred_masks', help='Output directory for thresholded prediction masks')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    gpu_id = int(args.gpu_id)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    print(f'#----------Loading pretrained weights from {args.checkpoint_path}----------#')
    print(f'#----------Using device: {device}----------#')
    print(f'#----------Saving thresholded prediction masks to {args.output_dir}----------#')
    print(f'#----------Saving probability maps to {args.output_dir}_prob----------#')

    os.makedirs(args.output_dir, exist_ok=True)
    generate_branch1_pred_masks(
        data_path=args.data_path,
        branch1_model_path=args.checkpoint_path,
        device=device,
        splits=('test',),
        output_dirs={'test': args.output_dir},
    )

    print('Prediction masks generated successfully!')
