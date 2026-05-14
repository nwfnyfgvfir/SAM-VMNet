from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from configs.config_setting import setting_config
from med_sam.medsam_point import medsam_point
from models.vmunet.vmunet import VMUNet


SPLITS = ('train', 'val', 'test')


def _list_files(directory):
    return sorted([
        file_name for file_name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file_name)) and not file_name.endswith(':Zone.Identifier')
    ])


def _collect_named_pairs(split_dir, mask_subdir):
    image_dir = os.path.join(split_dir, 'images')
    mask_dir = os.path.join(split_dir, mask_subdir)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Image directory not found: {image_dir}')
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

    image_files = _list_files(image_dir)
    mask_map = {
        Path(file_name).stem: file_name
        for file_name in _list_files(mask_dir)
    }

    pairs = []
    missing_masks = []
    for image_file in image_files:
        stem = Path(image_file).stem
        mask_file = mask_map.get(stem)
        if mask_file is None:
            missing_masks.append(image_file)
            continue
        pairs.append((
            image_file,
            os.path.join(image_dir, image_file),
            os.path.join(mask_dir, mask_file),
        ))

    if missing_masks:
        raise FileNotFoundError(
            f'Missing {mask_subdir} files for split {Path(split_dir).name}: {missing_masks[:5]}'
        )

    return pairs


def _build_branch1_model(device):
    model_cfg = setting_config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()
    return model.to(device)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            return checkpoint['model']
    return checkpoint


def _load_branch1_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and 'total_ops' not in key and 'total_params' not in key
    }
    print(
        'Branch1 checkpoint load: total model keys {}, total checkpoint keys {}, update {}'.format(
            len(model_state), len(state_dict), len(filtered_state_dict)
        )
    )
    model.load_state_dict(filtered_state_dict, strict=False)


def _save_prediction_outputs(prediction, binary_output_path, probability_output_path, output_size):
    probability_prediction = np.clip(prediction, 0, 1)
    binary_prediction = np.where(probability_prediction > setting_config.threshold, 255, 0).astype(np.uint8)
    probability_image = (probability_prediction * 255).astype(np.uint8)

    Image.fromarray(binary_prediction).resize(output_size, resample=Image.NEAREST).save(binary_output_path)
    Image.fromarray(probability_image).resize(output_size, resample=Image.BILINEAR).save(probability_output_path)


def generate_branch1_pred_masks(data_path, branch1_model_path, device, splits=SPLITS, output_dirs=None):
    model = _build_branch1_model(device)
    _load_branch1_checkpoint(model, branch1_model_path)
    model.eval()

    with torch.no_grad():
        for split in splits:
            split_dir = os.path.join(data_path, split)
            pairs = _collect_named_pairs(split_dir, 'masks')
            output_dir = output_dirs.get(split) if output_dirs and split in output_dirs else os.path.join(split_dir, 'pred_masks')
            probability_output_dir = f'{output_dir}_prob'
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(probability_output_dir, exist_ok=True)

            print(f'Generating Branch1 pred_masks for {split}...')
            prediction_mins = []
            prediction_maxs = []
            prediction_means = []
            prediction_positive_ratios = []
            for image_file, image_path, mask_path in tqdm(pairs, total=len(pairs)):
                output_path = os.path.join(output_dir, image_file)
                probability_output_path = os.path.join(probability_output_dir, image_file)

                image = np.array(Image.open(image_path).convert('RGB'))
                mask = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255
                original_height, original_width = mask.shape[:2]
                transformed_image, _ = setting_config.test_transformer((image, mask))
                transformed_image = transformed_image.unsqueeze(0).to(device, non_blocking=True).float()

                output = model(transformed_image)
                if isinstance(output, tuple):
                    output = output[0]
                prediction = output.squeeze().detach().cpu().numpy()
                prediction_mins.append(float(prediction.min()))
                prediction_maxs.append(float(prediction.max()))
                prediction_means.append(float(prediction.mean()))
                prediction_positive_ratios.append(float((prediction >= setting_config.threshold).mean()))
                _save_prediction_outputs(prediction, output_path, probability_output_path, (original_width, original_height))

            print(
                f'{split} Branch1 predictions: min={min(prediction_mins):.4f}, max={max(prediction_maxs):.4f}, '
                f'mean={np.mean(prediction_means):.4f}, positive_ratio@{setting_config.threshold}={np.mean(prediction_positive_ratios):.6f}'
            )
            print(f'Saved binary masks to {output_dir} and probability maps to {probability_output_dir}')


def _validate_mask_size(image_path, mask_path):
    with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
        if image_file.size != mask_file.size:
            raise ValueError(
                f'Image and pred_mask sizes do not match: {image_path} ({image_file.size}) vs {mask_path} ({mask_file.size})'
            )


def process_images(data_path, model_path, splits=SPLITS):
    for split in splits:
        split_dir = os.path.join(data_path, split)
        pairs = _collect_named_pairs(split_dir, 'pred_masks')
        output_dir = os.path.join(split_dir, 'feature')
        os.makedirs(output_dir, exist_ok=True)

        print(f'Processing {split} Dataset...')
        empty_prompt_count = 0
        for image_file, image_path, mask_path in tqdm(pairs, total=len(pairs)):
            output_file = os.path.join(output_dir, f'{Path(image_file).stem}.pt')

            _validate_mask_size(image_path, mask_path)
            medsam_result, used_prompt = medsam_point(image_path, mask_path, model_path)
            if not used_prompt:
                empty_prompt_count += 1
            torch.save(medsam_result, output_file)

        print(f'{split} MedSAM fallback on {empty_prompt_count}/{len(pairs)} empty pred_masks')


def validate_branch2_inputs(data_path, splits=SPLITS):
    for split in splits:
        split_dir = os.path.join(data_path, split)
        image_dir = os.path.join(split_dir, 'images')
        pred_mask_dir = os.path.join(split_dir, 'pred_masks')
        feature_dir = os.path.join(split_dir, 'feature')

        image_files = _list_files(image_dir)
        pred_mask_map = {Path(file_name).stem: file_name for file_name in _list_files(pred_mask_dir)}
        feature_map = {Path(file_name).stem: file_name for file_name in _list_files(feature_dir)}

        missing_pred_masks = []
        missing_features = []
        for image_file in image_files:
            stem = Path(image_file).stem
            if stem not in pred_mask_map:
                missing_pred_masks.append(image_file)
            if stem not in feature_map:
                missing_features.append(image_file)

        if missing_pred_masks:
            raise FileNotFoundError(f'Missing pred_masks for split {split}: {missing_pred_masks[:5]}')
        if missing_features:
            raise FileNotFoundError(f'Missing feature files for split {split}: {missing_features[:5]}')

        for image_file in image_files:
            stem = Path(image_file).stem
            _validate_mask_size(
                os.path.join(image_dir, image_file),
                os.path.join(pred_mask_dir, pred_mask_map[stem]),
            )
