import argparse
import os
import torch
import warnings
from torch.utils.data import DataLoader

from configs.config_setting import setting_config
from dataset import Branch2_datasets
from engine_branch2 import test_one_epoch
from feature_processor import generate_branch1_pred_masks, process_images, validate_branch2_inputs
from models.vmunet.samvmnet import SAMVMNet
from utils import get_logger, log_config_info, set_seed

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Branch2 on a selected split')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root, e.g. ./data/vessel/')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Branch2 checkpoint')
    parser.add_argument('--work_dir', type=str, default='./result_branch2_test/', help='Directory for logs and output masks')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--medsam_path', type=str, required=True, help='Path to MedSAM checkpoint')
    parser.add_argument('--branch1_model_path', type=str, required=True, help='Path to Branch1 checkpoint for generating pred_masks')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test', help='Dataset split to evaluate')
    return parser.parse_args()


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            return checkpoint['model']
    return checkpoint


def _load_branch2_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    raw_state_dict = _extract_state_dict(checkpoint)

    model_state = model.state_dict()
    mapped_state_dict = {}
    for key, value in raw_state_dict.items():
        new_key = key
        if new_key.startswith('vmunet.'):
            new_key = 'samvmnet.' + new_key[len('vmunet.'):]
        mapped_state_dict[new_key] = value

    compatible_state_dict = {
        key: value
        for key, value in mapped_state_dict.items()
        if key in model_state and model_state[key].shape == value.shape and 'total_ops' not in key and 'total_params' not in key
    }
    model_state.update(compatible_state_dict)
    model.load_state_dict(model_state)

    print(
        'Branch2 checkpoint load: total model keys {}, total checkpoint keys {}, update {}'.format(
            len(model_state), len(raw_state_dict), len(compatible_state_dict)
        )
    )


if __name__ == '__main__':
    config = setting_config
    args = parse_args()

    config.work_dir = args.work_dir if args.work_dir.endswith('/') else args.work_dir + '/'
    config.data_path = args.data_path
    config.gpu_id = args.gpu_id
    config.batch_size = 1

    gpu_id = int(config.gpu_id)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print(f'#----------Generating Branch1 {args.split} pred_masks----------#')
    generate_branch1_pred_masks(config.data_path, args.branch1_model_path, device, splits=(args.split,))

    print(f'#----------Generating Branch2 {args.split} features----------#')
    process_images(config.data_path, args.medsam_path, splits=(args.split,))
    validate_branch2_inputs(config.data_path, splits=(args.split,))

    log_dir = os.path.join(config.work_dir, 'log')
    output_dir = os.path.join(config.work_dir, 'outputs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(f'test_branch2_{args.split}', log_dir)
    log_config_info(config, logger)

    test_dataset = Branch2_datasets(
        config.data_path,
        config,
        train=False,
        test=args.split == 'test',
        return_image_name=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    model_cfg = config.model_config
    model = SAMVMNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
    ).to(device)
    _load_branch2_checkpoint(model, args.checkpoint_path)

    criterion = config.criterion
    test_one_epoch(
        test_loader,
        model,
        criterion,
        logger,
        config,
        device,
        test_data_name=args.split,
    )
